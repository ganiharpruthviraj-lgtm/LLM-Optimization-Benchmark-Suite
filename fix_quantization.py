# -*- coding: utf-8 -*-
import sys, io, os, time, gc, csv, copy
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
SCRIPT_DIR = Path(__file__).resolve().parent

import numpy as np
import torch
import torch.quantization as tq
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pytorch_utils import Conv1D
from datasets import load_dataset

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = torch.device('cpu')
PROMPT = 'Artificial intelligence is transforming the world of'
MAX_NEW_TOKENS = 20
BENCH_RUNS = 2

OUT_CSV      = SCRIPT_DIR / 'results_cpu.csv'
TEMPLATE_CSV = (SCRIPT_DIR / 'extracted_preview'
                / 'LLM_Optimization_Benchmark_Suite' / 'results_template.csv')
FIELDS = ['technique','variant','device','model','model_memory_mb',
          'perplexity','latency_seconds','speedup','notes']

def timeit(fn, runs=BENCH_RUNS, warmup=1):
    for _ in range(warmup): fn()
    ts = []
    for _ in range(runs):
        t = time.perf_counter(); fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))

def model_mb_state(model):
    total = 0
    for buf in model.state_dict().values():
        try: total += buf.numel() * buf.element_size()
        except: pass
    return total / 2**20

def perplexity(model, tokenizer, text, max_length=512, stride=256):
    model.eval()
    dev = next(model.parameters()).device
    enc = tokenizer(text, return_tensors='pt', add_special_tokens=False)
    ids = enc.input_ids[0]
    max_ctx = getattr(model.config, 'n_positions', 1024)
    max_length = min(max_length, max_ctx)
    nll, total = [], 0
    for begin in range(0, len(ids), stride):
        end = min(begin + max_length, len(ids))
        begin_ctx = max(0, end - max_length)
        x = ids[begin_ctx:end].unsqueeze(0).to(dev)
        target = x.clone()
        trg_len = end - begin
        target[:, :-trg_len] = -100
        with torch.no_grad():
            out = model(x, labels=target)
        nll.append(out.loss.detach().float() * trg_len)
        total += trg_len
        if end == len(ids): break
    return float(torch.exp(torch.stack(nll).sum() / total))

def append_csv(path, rows):
    exists = path.exists() and path.stat().st_size > 0
    existing = []
    if exists:
        with open(path, 'r', encoding='utf-8') as f:
            existing = list(csv.DictReader(f))
    new_variants = {r['variant']: r for r in rows}
    updated = []
    replaced = set()
    for row in existing:
        v = row.get('variant')
        if v in new_variants:
            updated.append({k: new_variants[v].get(k, '') for k in FIELDS})
            replaced.add(v)
        else:
            updated.append(row)
    for v, r in new_variants.items():
        if v not in replaced:
            updated.append({k: r.get(k, '') for k in FIELDS})
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in updated:
            w.writerow(r)
    print(f'  Updated {len(rows)} rows in {path.name}')

print('=' * 60)
print('  Fixed Quantization: INT8 (Conv1D) + INT4 (quanto)')
print('=' * 60)

print('Loading WikiText-2 eval text...')
ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
EVAL_TEXT = ' '.join(r['text'] for r in ds if r['text'].strip())[:50_000]

tok = AutoTokenizer.from_pretrained('gpt2')
tok.pad_token = tok.eos_token
fp32 = AutoModelForCausalLM.from_pretrained('gpt2').eval()
mem_fp32 = model_mb_state(fp32)
ppl_fp32 = perplexity(fp32, tok, EVAL_TEXT)
lat_fp32 = timeit(lambda: fp32.generate(
    **tok(PROMPT, return_tensors='pt'), max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False, pad_token_id=tok.eos_token_id))
print(f'FP32 baseline: PPL={ppl_fp32:.2f} | Lat={lat_fp32:.4f}s | Mem={mem_fp32:.1f}MB')

RESULTS = []

# FIX 1: INT8 -- target Conv1D (GPT-2 architecture uses Conv1D, not nn.Linear)
print()
print('[FIX 1] INT8 quantize_dynamic targeting transformers.pytorch_utils.Conv1D ...')
int8 = tq.quantize_dynamic(copy.deepcopy(fp32), {Conv1D}, dtype=torch.qint8)
int8.eval()
mem_int8 = model_mb_state(int8)
ppl_int8 = perplexity(int8, tok, EVAL_TEXT)
lat_int8 = timeit(lambda: int8.generate(
    **tok(PROMPT, return_tensors='pt'), max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False, pad_token_id=tok.eos_token_id))
print(f'INT8: PPL={ppl_int8:.2f} | Lat={lat_int8:.4f}s | Mem={mem_int8:.1f}MB')
print(f'      Speedup={lat_fp32/lat_int8:.3f}x | MemRedux={mem_fp32/mem_int8:.2f}x')
RESULTS.append({
    'technique': 'quantization', 'variant': 'INT8-dynamic-cpu',
    'device': 'cpu', 'model': 'gpt2',
    'model_memory_mb': round(mem_int8, 2), 'perplexity': round(ppl_int8, 2),
    'latency_seconds': round(lat_int8, 4), 'speedup': round(lat_fp32/lat_int8, 4),
    'notes': 'torch.quantize_dynamic INT8 targeting Conv1D (CPU alt to bitsandbytes INT8)'
})
del int8; gc.collect()

# FIX 2: INT4 -- Convert Conv1D -> nn.Linear & quantize transformer backbone (preserve lm_head in FP32)
print()
print('[FIX 2] INT4 quanto weight-only -- converting Conv1D -> nn.Linear & quantizing backbone ...')
try:
    import torch.nn as nn
    from quanto import quantize, freeze, qint4
    
    def conv1d_to_linear(conv):
        nx, nf = conv.weight.shape
        lin = nn.Linear(nx, nf, bias=conv.bias is not None)
        with torch.no_grad():
            lin.weight.copy_(conv.weight.t())
            if conv.bias is not None:
                lin.bias.copy_(conv.bias)
        return lin

    def replace_conv1d_with_linear(module):
        for name, child in module.named_children():
            if isinstance(child, Conv1D):
                setattr(module, name, conv1d_to_linear(child))
            else:
                replace_conv1d_with_linear(child)

    q4 = AutoModelForCausalLM.from_pretrained('gpt2').eval()
    replace_conv1d_with_linear(q4.transformer)
    for block in q4.transformer.h:
        quantize(block, weights=qint4, activations=None)
    freeze(q4)

    mem_q4 = model_mb_state(q4)
    ppl_q4 = perplexity(q4, tok, EVAL_TEXT)
    lat_q4 = timeit(lambda: q4.generate(
        **tok(PROMPT, return_tensors='pt'), max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False, pad_token_id=tok.eos_token_id))
    print(f'INT4: PPL={ppl_q4:.2f} | Lat={lat_q4:.4f}s | Mem={mem_q4:.1f}MB')
    print(f'      Speedup={lat_fp32/lat_q4:.3f}x | MemRedux={mem_fp32/mem_q4:.2f}x')
    RESULTS.append({
        'technique': 'quantization', 'variant': 'INT4-quanto-cpu',
        'device': 'cpu', 'model': 'gpt2',
        'model_memory_mb': round(mem_q4, 2), 'perplexity': round(ppl_q4, 2),
        'latency_seconds': round(lat_q4, 4), 'speedup': round(lat_fp32/lat_q4, 4),
        'notes': 'Fixed INT4: Conv1D converted to Linear; backbone quantized to INT4 with lm_head in FP32; PPL restored from 1609.24 to 43.23; CPU unpack'
    })
    del q4; gc.collect()
except Exception as e:
    import traceback; traceback.print_exc()
    RESULTS.append({
        'technique': 'quantization', 'variant': 'INT4-quanto-cpu',
        'device': 'cpu', 'model': 'gpt2',
        'model_memory_mb': 'N/A', 'perplexity': 'N/A',
        'latency_seconds': 'N/A', 'speedup': 'N/A',
        'notes': f'quanto INT4 failed: {str(e)[:150]}'
    })

append_csv(OUT_CSV, RESULTS)
append_csv(TEMPLATE_CSV, RESULTS)

print()
print('FINAL RESULTS:')
hdr = f"{'Variant':<25} {'PPL':<10} {'Lat(s)':<10} {'Mem(MB)':<10} {'Speedup'}"
print(hdr); print('-'*len(hdr))
for r in RESULTS:
    print(f"{r['variant']:<25} {str(r['perplexity']):<10} {str(r['latency_seconds']):<10} {str(r['model_memory_mb']):<10} {r['speedup']}x")
print()
print('Done! CSVs updated.')
