# -*- coding: utf-8 -*-
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ.setdefault("HF_HOME", r"D:\hf_cache")

"""
LLM Optimization Benchmark Suite -- CPU-Compatible Alternatives for GPU-Only Experiments
=========================================================================================
EXP A: INT8 / INT4 Quantization
       -> torch.quantization dynamic INT8 + optimum-quanto INT4 weight-only

EXP B: LoRA / PEFT Fine-tuning
       -> peft library (works on CPU, just slower)

EXP C: Efficient Attention (FlashAttention-2 analogue)
       -> PyTorch SDPA / BetterTransformer on CPU
"""

import time, gc, csv, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ---- Config --------------------------------------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE         = torch.device("cpu")
PROMPT         = "Artificial intelligence is transforming the world of"
MAX_NEW_TOKENS = 20
BENCH_RUNS     = 2
RESULTS: list[dict] = []

SCRIPT_DIR   = Path(__file__).parent
OUT_CSV      = SCRIPT_DIR / "results_cpu.csv"
TEMPLATE_CSV = (SCRIPT_DIR / "extracted_preview"
                / "LLM_Optimization_Benchmark_Suite" / "results_template.csv")

# ---- Helpers -------------------------------------------------------
def sep(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def timeit(fn, runs=BENCH_RUNS, warmup=1):
    for _ in range(warmup): fn()
    ts = []
    for _ in range(runs):
        t = time.perf_counter(); fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))

def model_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20

def model_mb_state(model):
    total = 0
    for buf in model.state_dict().values():
        try: total += buf.numel() * buf.element_size()
        except Exception: pass
    return total / 2**20

def load_wikitext_eval(max_chars=50_000):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = " ".join(r["text"] for r in ds if r["text"].strip())
    return text[:max_chars]

def perplexity(model, tokenizer, text, max_length=512, stride=256):
    model.eval()
    dev = next(model.parameters()).device
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids[0]
    max_ctx = getattr(model.config, "n_positions",
                      getattr(model.config, "max_position_embeddings", 1024))
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

# ---- Load eval text ------------------------------------------------
sep("Loading WikiText-2 evaluation text")
EVAL_TEXT = load_wikitext_eval()
print(f"  Loaded {len(EVAL_TEXT)} chars / ~{len(EVAL_TEXT.split())} words")

# ==================================================================
# EXPERIMENT A -- INT8 & INT4 Quantization (CPU-compatible)
# ==================================================================
sep("EXPERIMENT A -- Quantization: Dynamic INT8 + Quanto INT4 (CPU)")

tokA = AutoTokenizer.from_pretrained("gpt2")
tokA.pad_token = tokA.eos_token
fp32 = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE).eval()
mem_fp32 = model_mb(fp32)
ppl_fp32 = perplexity(fp32, tokA, EVAL_TEXT)
lat_fp32 = timeit(lambda: fp32.generate(
    **tokA(PROMPT, return_tensors="pt"), max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False, pad_token_id=tokA.eos_token_id))
print(f"  FP32 baseline -> PPL={ppl_fp32:.2f} | Lat={lat_fp32:.4f}s | Mem={mem_fp32:.1f}MB")

# -- Dynamic INT8 (torch.quantization) --
import copy, torch.quantization as tq
print("\n  Applying torch.quantization.quantize_dynamic (INT8)...")
int8 = tq.quantize_dynamic(copy.deepcopy(fp32).cpu(), {torch.nn.Linear}, dtype=torch.qint8)
int8.eval()
mem_int8 = model_mb_state(int8)
ppl_int8 = perplexity(int8, tokA, EVAL_TEXT)
lat_int8 = timeit(lambda: int8.generate(
    **tokA(PROMPT, return_tensors="pt"), max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False, pad_token_id=tokA.eos_token_id))
print(f"  INT8 -> PPL={ppl_int8:.2f} | Lat={lat_int8:.4f}s | Mem={mem_int8:.1f}MB | "
      f"Speedup={lat_fp32/lat_int8:.3f}x")
RESULTS.append({
    "technique": "quantization", "variant": "INT8-dynamic-cpu",
    "device": "cpu", "model": "gpt2",
    "model_memory_mb": round(mem_int8, 2), "perplexity": round(ppl_int8, 2),
    "latency_seconds": round(lat_int8, 4), "speedup": round(lat_fp32/lat_int8, 4),
    "notes": "torch.quantize_dynamic INT8 (CPU alt to bitsandbytes INT8)"
})

# -- Quanto INT4 weight-only --
print("\n  Applying optimum-quanto INT4 weight-only quantization...")
try:
    from optimum.quanto import quantize, freeze, qint4
    q4 = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE).eval()
    quantize(q4, weights=qint4, activations=None)
    freeze(q4)
    mem_q4  = model_mb_state(q4)
    ppl_q4  = perplexity(q4, tokA, EVAL_TEXT)
    lat_q4  = timeit(lambda: q4.generate(
        **tokA(PROMPT, return_tensors="pt"), max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False, pad_token_id=tokA.eos_token_id))
    print(f"  Quanto-INT4 -> PPL={ppl_q4:.2f} | Lat={lat_q4:.4f}s | Mem={mem_q4:.1f}MB | "
          f"Speedup={lat_fp32/lat_q4:.3f}x")
    RESULTS.append({
        "technique": "quantization", "variant": "INT4-quanto-cpu",
        "device": "cpu", "model": "gpt2",
        "model_memory_mb": round(mem_q4, 2), "perplexity": round(ppl_q4, 2),
        "latency_seconds": round(lat_q4, 4), "speedup": round(lat_fp32/lat_q4, 4),
        "notes": "optimum-quanto INT4 weight-only (CPU alt to bitsandbytes NF4)"
    })
    del q4
except Exception as e:
    print(f"  Quanto INT4 failed: {e}")
    RESULTS.append({
        "technique": "quantization", "variant": "INT4-quanto-cpu",
        "device": "cpu", "model": "gpt2",
        "model_memory_mb": "N/A", "perplexity": "N/A",
        "latency_seconds": "N/A", "speedup": "N/A",
        "notes": f"optimum-quanto INT4 failed on CPU: {str(e)[:100]}"
    })

del fp32, int8; gc.collect()

# ==================================================================
# EXPERIMENT B -- LoRA / PEFT Fine-tuning (CPU via peft library)
# ==================================================================
sep("EXPERIMENT B -- LoRA / PEFT Fine-tuning (CPU via peft library)")
try:
    from peft import LoraConfig, get_peft_model, TaskType

    tokB = AutoTokenizer.from_pretrained("gpt2")
    tokB.pad_token = tokB.eos_token
    base = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
    mem_base = model_mb(base)

    base.eval()
    lat_base = timeit(lambda: base.generate(
        **tokB(PROMPT, return_tensors="pt"), max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False, pad_token_id=tokB.eos_token_id))
    ppl_base = perplexity(base, tokB, EVAL_TEXT)
    print(f"  Base (pre-LoRA) -> PPL={ppl_base:.2f} | Lat={lat_base:.4f}s | Mem={mem_base:.1f}MB")
    RESULTS.append({
        "technique": "LoRA/PEFT", "variant": "full-finetune-baseline",
        "device": "cpu", "model": "gpt2",
        "model_memory_mb": round(mem_base, 2), "perplexity": round(ppl_base, 2),
        "latency_seconds": round(lat_base, 4), "speedup": 1.0,
        "notes": "Full GPT-2 baseline before LoRA; WikiText-2 PPL; CPU"
    })

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32,
        lora_dropout=0.05, target_modules=["c_attn"], bias="none")
    lora_model = get_peft_model(base, lora_cfg)
    lora_model.print_trainable_parameters()
    total_p     = sum(p.numel() for p in lora_model.parameters())
    trainable_p = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)

    print("\n  10-step LoRA training...")
    train_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:1%]")
    train_ds = train_ds.filter(lambda x: x["text"].strip() != "")
    opt = AdamW(filter(lambda p: p.requires_grad, lora_model.parameters()), lr=1e-4)
    lora_model.train()
    for step, row in enumerate(train_ds.select(range(min(10, len(train_ds))))):
        enc  = tokB(row["text"], return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
        out  = lora_model(**enc, labels=enc.input_ids)
        out.loss.backward(); opt.step(); opt.zero_grad()
        print(f"    step {step+1:>2}/10  loss={float(out.loss.detach()):.4f}")

    lora_model.eval()
    ppl_lora = perplexity(lora_model, tokB, EVAL_TEXT)
    lat_lora = timeit(lambda: lora_model.generate(
        **tokB(PROMPT, return_tensors="pt"), max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False, pad_token_id=tokB.eos_token_id))
    mem_lora = model_mb(lora_model)
    print(f"\n  Post-LoRA -> PPL={ppl_lora:.2f} | Lat={lat_lora:.4f}s | Mem={mem_lora:.1f}MB")
    RESULTS.append({
        "technique": "LoRA/PEFT", "variant": "LoRA-r8-cpu",
        "device": "cpu", "model": "gpt2+LoRA(r=8)",
        "model_memory_mb": round(mem_lora, 2), "perplexity": round(ppl_lora, 2),
        "latency_seconds": round(lat_lora, 4), "speedup": round(lat_base/lat_lora, 4),
        "notes": (f"LoRA r=8 alpha=32; trainable={trainable_p:,}/{total_p:,} "
                  f"({100*trainable_p/total_p:.2f}%); 10-step mini-train; CPU")
    })
    del lora_model, base; gc.collect()

except Exception as e:
    import traceback; traceback.print_exc()
    for v in ["full-finetune-baseline", "LoRA-r8-cpu"]:
        RESULTS.append({
            "technique": "LoRA/PEFT", "variant": v,
            "device": "cpu", "model": "gpt2",
            "model_memory_mb": "N/A", "perplexity": "N/A",
            "latency_seconds": "N/A", "speedup": "N/A",
            "notes": f"peft error: {str(e)[:120]}"
        })

# ==================================================================
# EXPERIMENT C -- Efficient Attention: SDPA vs Standard (CPU)
# ==================================================================
sep("EXPERIMENT C -- Efficient Attention: PyTorch SDPA vs Standard (CPU)")
print("  FlashAttention-2 requires CUDA. Using PyTorch SDPA (F.scaled_dot_product_attention).\n")

tokC = AutoTokenizer.from_pretrained("gpt2")
tokC.pad_token = tokC.eos_token

# Standard attention baseline
std = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE).eval()
lat_std = timeit(lambda: std.generate(
    **tokC(PROMPT, return_tensors="pt"), max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False, pad_token_id=tokC.eos_token_id))
mem_std = model_mb(std)
print(f"  Standard attention -> Lat={lat_std:.4f}s | Mem={mem_std:.1f}MB")
RESULTS.append({
    "technique": "flash_attention", "variant": "standard-attn-cpu",
    "device": "cpu", "model": "gpt2",
    "model_memory_mb": round(mem_std, 2), "perplexity": "N/A",
    "latency_seconds": round(lat_std, 4), "speedup": 1.0,
    "notes": "Standard GPT-2 attention baseline; CPU"
})

# SDPA via BetterTransformer
print("\n  Trying BetterTransformer (SDPA)...")
try:
    from optimum.bettertransformer import BetterTransformer
    sdpa = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE).eval()
    sdpa = BetterTransformer.transform(sdpa)
    lat_sdpa = timeit(lambda: sdpa.generate(
        **tokC(PROMPT, return_tensors="pt"), max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False, pad_token_id=tokC.eos_token_id))
    mem_sdpa = model_mb(sdpa)
    sp = lat_std / lat_sdpa
    print(f"  SDPA (BetterTransformer) -> Lat={lat_sdpa:.4f}s | Speedup={sp:.3f}x")
    RESULTS.append({
        "technique": "flash_attention", "variant": "SDPA-BetterTransformer-cpu",
        "device": "cpu", "model": "gpt2",
        "model_memory_mb": round(mem_sdpa, 2), "perplexity": "N/A",
        "latency_seconds": round(lat_sdpa, 4), "speedup": round(sp, 4),
        "notes": "PyTorch SDPA via BetterTransformer (CPU analogue of FlashAttention-2)"
    })
    del sdpa
except Exception as e1:
    print(f"  BetterTransformer failed ({e1}). Using raw F.scaled_dot_product_attention benchmark...")
    try:
        B, H, S, D = 1, 12, 128, 64
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        mask = torch.ones(S, S, dtype=torch.bool).tril()

        def std_attn():
            sc = torch.matmul(q, k.transpose(-2,-1)) * (D**-0.5)
            sc = sc.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            return torch.matmul(F.softmax(sc, dim=-1), v)

        def sdpa_attn():
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        lat_s_raw  = timeit(std_attn,  runs=10)
        lat_sp_raw = timeit(sdpa_attn, runs=10)
        speedup_r  = lat_s_raw / lat_sp_raw
        print(f"  Raw SDPA benchmark (S=128, H=12): std={lat_s_raw*1000:.3f}ms | "
              f"sdpa={lat_sp_raw*1000:.3f}ms | speedup={speedup_r:.3f}x")
        RESULTS.append({
            "technique": "flash_attention", "variant": "SDPA-raw-kernel-cpu",
            "device": "cpu", "model": "tensor-B1-H12-S128-D64",
            "model_memory_mb": "N/A", "perplexity": "N/A",
            "latency_seconds": round(lat_sp_raw, 6), "speedup": round(speedup_r, 4),
            "notes": ("F.scaled_dot_product_attention vs manual attn; CPU kernel; "
                      "BetterTransformer unavailable")
        })
    except Exception as e2:
        print(f"  SDPA fallback failed: {e2}")
        RESULTS.append({
            "technique": "flash_attention", "variant": "SDPA-cpu",
            "device": "cpu", "model": "gpt2",
            "model_memory_mb": "N/A", "perplexity": "N/A",
            "latency_seconds": "N/A", "speedup": "N/A",
            "notes": f"All SDPA attempts failed: {str(e2)[:100]}"
        })

del std; gc.collect()

# ==================================================================
# Write results (append to existing CSVs)
# ==================================================================
sep("SAVING RESULTS")
FIELDS = ["technique", "variant", "device", "model", "model_memory_mb",
          "perplexity", "latency_seconds", "speedup", "notes"]

def append_csv(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists: w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"  Appended {len(rows)} rows -> {path}")

append_csv(OUT_CSV, RESULTS)
append_csv(TEMPLATE_CSV, RESULTS)

print()
hdr = f"{'Technique':<22} {'Variant':<30} {'Mem(MB)':<10} {'PPL':<8} {'Lat(s)':<10} {'Speedup'}"
print(hdr); print("-" * len(hdr))
for r in RESULTS:
    print(f"{r['technique']:<22} {r['variant']:<30} "
          f"{str(r.get('model_memory_mb','')):<10} "
          f"{str(r.get('perplexity','')):<8} "
          f"{str(r.get('latency_seconds','')):<10} "
          f"{r.get('speedup','')}")
print("\nDone!")
