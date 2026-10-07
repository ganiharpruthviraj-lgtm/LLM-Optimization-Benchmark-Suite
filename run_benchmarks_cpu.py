# -*- coding: utf-8 -*-
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
# Pin HuggingFace cache to D: (71 GB free) -- avoids C: drive space issues
os.environ.setdefault("HF_HOME", r"D:\hf_cache")

"""
LLM Optimization Benchmark Suite -- CPU-compatible runner (CORRECTED)
Fixes applied vs v1:
  1. Perplexity now uses WikiText-2 test split (real diverse text), not a
     repeated single sentence. GPT-2 should score ~29-35 PPL on wikitext.
  2. Speculative decoding uses gpt2 (draft) + gpt2-medium (target) -- a
     realistic small/large pair so the draft can actually save work.
  3. Results are written to BOTH results_cpu.csv (full detail) AND the
     original results_template.csv inside the extracted notebook folder.

GPU-only steps (INT8/INT4 bitsandbytes, LoRA/PEFT, FlashAttention-2) are
clearly skipped with explanatory notes.
"""

import os, time, math, gc, csv, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROMPT      = "Artificial intelligence is transforming the world of"
MAX_NEW_TOKENS = 20   # keep low so CPU finishes quickly
BENCH_RUNS     = 2    # median over N timed runs
RESULTS: list[dict] = []

# Output paths
SCRIPT_DIR  = Path(__file__).parent
OUT_CSV     = SCRIPT_DIR / "results_cpu.csv"
TEMPLATE_CSV = SCRIPT_DIR / "extracted_preview" / "LLM_Optimization_Benchmark_Suite" / "results_template.csv"

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def sep(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

def timeit(fn, runs=BENCH_RUNS, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t = time.perf_counter(); fn()
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))

def model_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20

# FIX 1: Use WikiText-2 test split for proper diverse-text perplexity.
# Previously we used a repeated sentence which gave an artificially low PPL.
# Real GPT-2 scores ~29-35 on WikiText-2.
def load_wikitext_eval(max_chars=50_000):
    """Return a single long string from WikiText-2 test split."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = " ".join(row["text"] for row in ds if row["text"].strip())
    return text[:max_chars]   # cap to keep CPU time reasonable

def perplexity(model, tokenizer, text, max_length=512, stride=256):
    """Sliding-window causal-LM perplexity; no padding counted."""
    model.eval()
    dev = next(model.parameters()).device
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids[0]
    max_ctx = getattr(model.config, "n_positions",
                      getattr(model.config, "max_position_embeddings", 1024))
    max_length = min(max_length, max_ctx)
    nll, total = [], 0
    for begin in range(0, len(ids), stride):
        end        = min(begin + max_length, len(ids))
        begin_ctx  = max(0, end - max_length)
        x          = ids[begin_ctx:end].unsqueeze(0).to(dev)
        target     = x.clone()
        trg_len    = end - begin
        target[:, :-trg_len] = -100
        with torch.no_grad():
            out = model(x, labels=target)
        nll.append(out.loss.detach().float() * trg_len)
        total += trg_len
        if end == len(ids):
            break
    return float(torch.exp(torch.stack(nll).sum() / total))

# ------------------------------------------------------------------ #
# Load evaluation text once (shared across experiments)
# ------------------------------------------------------------------ #
sep("Loading WikiText-2 evaluation text")
print("Fetching WikiText-2 test split (first 50 000 chars) ...")
EVAL_TEXT = load_wikitext_eval()
print(f"Eval text length: {len(EVAL_TEXT)} chars / "
      f"~{len(EVAL_TEXT.split())} words")

# ------------------------------------------------------------------ #
# EXPERIMENT 1 -- Quantization: FP32 baseline (CPU)
# ------------------------------------------------------------------ #
sep("EXPERIMENT 1 -- Quantization: FP32 baseline")
print(f"Device: {DEVICE}  |  Model: gpt2")
tok1 = AutoTokenizer.from_pretrained("gpt2")
tok1.pad_token = tok1.eos_token
fp32 = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE).eval()

ppl_fp32 = perplexity(fp32, tok1, EVAL_TEXT)
lat_fp32 = timeit(lambda: fp32.generate(
    **tok1(PROMPT, return_tensors="pt").to(DEVICE),
    max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
    pad_token_id=tok1.eos_token_id))

r = {"technique": "quantization", "variant": "FP32", "device": str(DEVICE),
     "model": "gpt2", "model_memory_mb": round(model_mb(fp32), 2),
     "perplexity": round(ppl_fp32, 2), "latency_seconds": round(lat_fp32, 4),
     "speedup": 1.0, "notes": "FP32 baseline on WikiText-2 test perplexity"}
RESULTS.append(r)
print(f"  FP32  ->  PPL={ppl_fp32:.2f}  |  Latency={lat_fp32:.4f}s  |  Mem={model_mb(fp32):.1f} MB")

for label, reason in [("INT8",     "requires CUDA GPU + bitsandbytes"),
                       ("INT4-NF4", "requires CUDA GPU + bitsandbytes")]:
    note = f"SKIPPED: {reason}"
    print(f"  {label}: {note}")
    RESULTS.append({"technique": "quantization", "variant": label,
                    "device": "N/A", "model": "gpt2",
                    "model_memory_mb": "N/A", "perplexity": "N/A",
                    "latency_seconds": "N/A", "speedup": "N/A", "notes": note})
del fp32; gc.collect()

# ------------------------------------------------------------------ #
# EXPERIMENT 2 -- LoRA / PEFT (GPU only -- skipped)
# ------------------------------------------------------------------ #
sep("EXPERIMENT 2 -- LoRA / PEFT  [SKIPPED -- GPU required]")
print("  LoRA fine-tuning with 8-bit base model requires CUDA + bitsandbytes.")
print("  Run notebook 02_lora_peft_corrected.ipynb on a GPU instance.")
for v in ["full-finetune", "LoRA"]:
    RESULTS.append({"technique": "LoRA/PEFT", "variant": v, "device": "N/A",
                    "model": "gpt2", "model_memory_mb": "N/A", "perplexity": "N/A",
                    "latency_seconds": "N/A", "speedup": "N/A",
                    "notes": "SKIPPED: requires CUDA GPU + bitsandbytes"})

# ------------------------------------------------------------------ #
# EXPERIMENT 3 -- KV Cache (CPU-compatible)
# ------------------------------------------------------------------ #
sep("EXPERIMENT 3 -- KV Cache  (with vs without)")
print("  Note: FlashAttention-2 requires CUDA; skipped here.")
tok3 = AutoTokenizer.from_pretrained("gpt2")
tok3.pad_token = tok3.eos_token
m3   = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE).eval()
inp3 = tok3(PROMPT, return_tensors="pt").to(DEVICE)

lat_kv   = timeit(lambda: m3.generate(
    **inp3, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
    use_cache=True,  pad_token_id=tok3.eos_token_id))
lat_nokv = timeit(lambda: m3.generate(
    **inp3, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
    use_cache=False, pad_token_id=tok3.eos_token_id))

speedup_kv = lat_nokv / lat_kv
print(f"  With KV cache:    {lat_kv:.4f}s")
print(f"  Without KV cache: {lat_nokv:.4f}s")
print(f"  Cache speedup:    {speedup_kv:.3f}x")

RESULTS.append({"technique": "kv_cache", "variant": "with_cache",
                "device": str(DEVICE), "model": "gpt2",
                "model_memory_mb": round(model_mb(m3), 2), "perplexity": "N/A",
                "latency_seconds": round(lat_kv,   4),
                "speedup": round(speedup_kv, 4),
                "notes": "KV cache enabled (default)"})
RESULTS.append({"technique": "kv_cache", "variant": "without_cache",
                "device": str(DEVICE), "model": "gpt2",
                "model_memory_mb": round(model_mb(m3), 2), "perplexity": "N/A",
                "latency_seconds": round(lat_nokv, 4), "speedup": 1.0,
                "notes": "Recomputes K/V every step -- correct baseline"})
RESULTS.append({"technique": "flash_attention", "variant": "FlashAttention-2",
                "device": "N/A", "model": "N/A",
                "model_memory_mb": "N/A", "perplexity": "N/A",
                "latency_seconds": "N/A", "speedup": "N/A",
                "notes": "SKIPPED: requires CUDA + flash-attn package"})
del m3; gc.collect()

# ------------------------------------------------------------------ #
# EXPERIMENT 4 -- Speculative Decoding  (distilgpt2 draft + gpt2 target)
#
# v1 bug: used gpt2 as BOTH target and draft -> draft always matched target,
# so no tokens were ever rejected and we paid 2x compute with zero benefit.
#
# Fix: distilgpt2 (82M, draft) + gpt2 (117M, target).
# Both models are already cached locally (no extra disk space needed).
# distilgpt2 shares GPT-2's BPE vocabulary so token IDs are directly comparable.
# The draft is ~30% smaller/faster; the target verifies in one forward pass.
# On CPU the overhead may still outweigh gains at 20 tokens; the experiment
# is a CORRECTNESS demonstration (token-match check) + latency comparison.
# Real hardware speedup is seen on GPU with longer generation sequences.
# ------------------------------------------------------------------ #
sep("EXPERIMENT 4 -- Speculative Decoding  (distilgpt2 draft + gpt2 target)")
print("  Loading gpt2 as target  (~117 MB, already cached) ...")
tok4   = AutoTokenizer.from_pretrained("gpt2")
tok4.pad_token = tok4.eos_token
target = AutoModelForCausalLM.from_pretrained(
    "gpt2", dtype=torch.float32).to(DEVICE).eval()
print("  Loading distilgpt2 as draft (~82 MB, already cached) ...")
tok4d  = AutoTokenizer.from_pretrained("distilgpt2")
tok4d.pad_token = tok4d.eos_token
draft  = AutoModelForCausalLM.from_pretrained(
    "distilgpt2", dtype=torch.float32).to(DEVICE).eval()

# Re-tokenize with target tokenizer (same BPE vocab for gpt2 family)
ids4 = tok4(PROMPT, return_tensors="pt").input_ids.to(DEVICE)

@torch.no_grad()
def greedy(model, ids, n=MAX_NEW_TOKENS):
    return model.generate(input_ids=ids, max_new_tokens=n,
                          do_sample=False, use_cache=True,
                          pad_token_id=tok4.eos_token_id)

@torch.no_grad()
def speculative_greedy(tgt, dft, ids, max_new_tokens=MAX_NEW_TOKENS, k=5):
    """Greedy speculative decoding with exact token-match verification."""
    generated = ids.clone(); produced = 0
    while produced < max_new_tokens:
        kk = min(k, max_new_tokens - produced)
        cur = generated.clone(); proposals = []
        # Draft proposes k greedy tokens
        for _ in range(kk):
            d_logits = dft(cur, use_cache=False).logits[:, -1, :]
            nxt = d_logits.argmax(-1, keepdim=True)
            proposals.append(nxt); cur = torch.cat([cur, nxt], dim=1)
        proposal = torch.cat(proposals, dim=1)
        # One target forward verifies all proposed positions
        full   = torch.cat([generated, proposal], dim=1)
        logits = tgt(full, use_cache=False).logits
        accepted = []
        for j in range(kk):
            target_best = logits[:, generated.shape[1] - 1 + j, :].argmax(-1, keepdim=True)
            if target_best.item() == proposal[0, j].item():
                accepted.append(proposal[:, j:j+1])
            else:
                accepted.append(target_best); break   # reject from here
        new       = torch.cat(accepted, dim=1)
        generated = torch.cat([generated, new], dim=1)
        produced += new.shape[1]
        if new[0, -1].item() == tok4.eos_token_id:
            break
    return generated

# Correctness check: must match pure greedy on the TARGET model
g_out = greedy(target, ids4)
s_out = speculative_greedy(target, draft, ids4)
match = torch.equal(g_out, s_out)
print(f"\n  Exact token match (greedy == speculative-greedy): {match}")
print(f"  GREEDY:      {tok4.decode(g_out[0], skip_special_tokens=True)}")
print(f"  SPECULATIVE: {tok4.decode(s_out[0], skip_special_tokens=True)}")

lat_g  = timeit(lambda: greedy(target, ids4))
lat_sp = timeit(lambda: speculative_greedy(target, draft, ids4))
speedup_sd = lat_g / lat_sp
print(f"\n  Greedy latency:      {lat_g:.4f}s")
print(f"  Speculative latency: {lat_sp:.4f}s")
print(f"  Speedup:             {speedup_sd:.3f}x")
print("  (CPU speedup is modest; GPU with longer sequences shows 2-3x gains)")

note_sd_base = "target=gpt2 (117M); pure greedy baseline"
note_sd_spec = (f"draft=distilgpt2 (82M) + target=gpt2 (117M); token_match={match}; "
                "real speedup on GPU with larger target & longer seqs")
RESULTS.append({"technique": "speculative_decoding", "variant": "greedy_baseline",
                "device": str(DEVICE), "model": "gpt2",
                "model_memory_mb": round(model_mb(target), 2),
                "perplexity": "N/A", "latency_seconds": round(lat_g, 4),
                "speedup": 1.0, "notes": note_sd_base})
RESULTS.append({"technique": "speculative_decoding", "variant": "speculative_greedy",
                "device": str(DEVICE), "model": "distilgpt2+gpt2",
                "model_memory_mb": round(model_mb(draft) + model_mb(target), 2),
                "perplexity": "N/A", "latency_seconds": round(lat_sp, 4),
                "speedup": round(speedup_sd, 4), "notes": note_sd_spec})
del target, draft; gc.collect()

# ------------------------------------------------------------------ #
# EXPERIMENT 5 -- Knowledge Distillation
# ------------------------------------------------------------------ #
sep("EXPERIMENT 5 -- Knowledge Distillation (GPT-2 teacher -> DistilGPT2 student)")
t_tok = AutoTokenizer.from_pretrained("gpt2");        t_tok.pad_token = t_tok.eos_token
s_tok = AutoTokenizer.from_pretrained("distilgpt2");  s_tok.pad_token = s_tok.eos_token
teacher = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE).eval()
student = AutoModelForCausalLM.from_pretrained("distilgpt2").to(DEVICE).eval()

print(f"  Teacher (gpt2):       {model_mb(teacher):.1f} MB")
print(f"  Student (distilgpt2): {model_mb(student):.1f} MB")
print(f"  Compression:          {model_mb(teacher)/model_mb(student):.2f}x smaller")

lat_t = timeit(lambda: teacher.generate(
    **t_tok(PROMPT, return_tensors="pt").to(DEVICE),
    max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=t_tok.eos_token_id))
lat_s = timeit(lambda: student.generate(
    **s_tok(PROMPT, return_tensors="pt").to(DEVICE),
    max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=s_tok.eos_token_id))

# FIX 1 applied here too: WikiText-2 perplexity
print("  Computing perplexity on WikiText-2 test split ...")
ppl_t = perplexity(teacher, t_tok, EVAL_TEXT)
ppl_s = perplexity(student, s_tok, EVAL_TEXT)
print(f"\n  Teacher -> latency={lat_t:.4f}s | WikiText-2 PPL={ppl_t:.2f}")
print(f"  Student -> latency={lat_s:.4f}s | WikiText-2 PPL={ppl_s:.2f}")
print(f"  Latency speedup: {lat_t/lat_s:.3f}x")

RESULTS.append({"technique": "distillation", "variant": "teacher",
                "device": str(DEVICE), "model": "gpt2",
                "model_memory_mb": round(model_mb(teacher), 2),
                "perplexity": round(ppl_t, 2),
                "latency_seconds": round(lat_t, 4), "speedup": 1.0,
                "notes": "teacher baseline; WikiText-2 test PPL"})
RESULTS.append({"technique": "distillation", "variant": "student_pretrained",
                "device": str(DEVICE), "model": "distilgpt2",
                "model_memory_mb": round(model_mb(student), 2),
                "perplexity": round(ppl_s, 2),
                "latency_seconds": round(lat_s, 4),
                "speedup": round(lat_t / lat_s, 4),
                "notes": "pre-trained distilled student; WikiText-2 test PPL"})

# Mini distillation training loop (10 steps)
print("\n  Running 10-step mini distillation training loop ...")
train_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:1%]")
train_ds = train_ds.filter(lambda x: x["text"].strip() != "")
opt = AdamW(student.parameters(), lr=5e-5); T = 2.0
student.train(); teacher.eval()
for step, row in enumerate(train_ds.select(range(min(10, len(train_ds))))):
    enc = s_tok(row["text"], return_tensors="pt",
                truncation=True, max_length=64).to(DEVICE)
    with torch.no_grad():
        tlog = teacher(**enc).logits
    slog = student(**enc).logits
    kl = F.kl_div(F.log_softmax(slog[:, :-1] / T, dim=-1),
                  F.softmax(tlog[:, 1:]  / T, dim=-1),
                  reduction="batchmean") * (T * T)
    ce   = F.cross_entropy(slog[:, :-1].reshape(-1, slog.size(-1)),
                           enc.input_ids[:, 1:].reshape(-1))
    loss = 0.5 * kl + 0.5 * ce
    loss.backward(); opt.step(); opt.zero_grad()
    print(f"    step {step+1:>2}/10  "
          f"loss={float(loss.detach()):.4f}  "
          f"kl={float(kl.detach()):.4f}  "
          f"ce={float(ce.detach()):.4f}")

student.eval()
ppl_s_post = perplexity(student, s_tok, EVAL_TEXT)
print(f"\n  Pre-distillation  student PPL: {ppl_s:.2f}")
print(f"  Post-distillation student PPL: {ppl_s_post:.2f}")
print("  (10 steps is a demo; real improvement needs thousands of steps)")

RESULTS.append({"technique": "distillation", "variant": "student_post_distill",
                "device": str(DEVICE), "model": "distilgpt2",
                "model_memory_mb": round(model_mb(student), 2),
                "perplexity": round(ppl_s_post, 2),
                "latency_seconds": "N/A", "speedup": "N/A",
                "notes": "after 10-step mini distillation; WikiText-2 test PPL"})

# ------------------------------------------------------------------ #
# FIX 3: Write results to BOTH CSV files
# ------------------------------------------------------------------ #
sep("RESULTS SUMMARY")
FIELDS = ["technique", "variant", "device", "model", "model_memory_mb",
          "perplexity", "latency_seconds", "speedup", "notes"]

def write_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in RESULTS:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"  Written -> {path}")

write_csv(OUT_CSV)
write_csv(TEMPLATE_CSV)

# Pretty print
print()
hdr = f"{'Technique':<22} {'Variant':<25} {'Mem(MB)':<10} {'PPL':<8} {'Lat(s)':<10} {'Speedup'}"
print(hdr)
print("-" * len(hdr))
for r in RESULTS:
    print(f"{r['technique']:<22} {r['variant']:<25} "
          f"{str(r.get('model_memory_mb','')):<10} "
          f"{str(r.get('perplexity','')):<8} "
          f"{str(r.get('latency_seconds','')):<10} "
          f"{r.get('speedup','')}")

print("\nDone! All results saved.")
