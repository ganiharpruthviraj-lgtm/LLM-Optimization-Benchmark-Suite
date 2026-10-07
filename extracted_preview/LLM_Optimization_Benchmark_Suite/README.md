# LLM Optimization Benchmark Suite

Prepared from the supplied CCBP notebooks. The project covers five optimization ideas:

1. **Quantization** — FP32 GPT-2 vs optional bitsandbytes INT8/INT4.
2. **LoRA / PEFT** — full fine-tuning vs low-rank adapters on an 8-bit base.
3. **KV Cache** — same-model decoding with and without KV cache.
4. **FlashAttention-2** — experimental design uses the same model/weights; it is not compared against a different model.
5. **Speculative Decoding** — deterministic greedy verification with exact output-match validation.
6. **Knowledge Distillation** — GPT-2 teacher vs DistilGPT2 student, plus a mini distillation training loop.

## Accuracy rules used

- No benchmark numbers are fabricated. Run the notebooks on the target hardware and report the printed measurements.
- Latency comparisons keep prompt, generated-token count, warmup and decoding settings consistent within each experiment.
- Perplexity uses causal next-token loss and avoids counting padding tokens.
- FlashAttention speed must be measured with the **same model architecture and weights**, changing only the attention implementation.
- BLEU was removed because the supplied notebook compared generated text against manually invented references, which is not a valid evaluation.
- Speculative greedy decoding checks exact token equality with normal target greedy decoding before comparing latency.

## Suggested run order

Run notebooks 01 → 05. For the GPU experiments, use a CUDA GPU and record GPU type, PyTorch version and Transformers version in your final report.

## Important

These are educational benchmarks, not universal claims. Results vary with GPU, CPU, CUDA, driver, Transformers version, batch size, sequence length, prompt, generated tokens and quantization kernels.
