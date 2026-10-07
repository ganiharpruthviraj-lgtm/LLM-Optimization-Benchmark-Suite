# -*- coding: utf-8 -*-
"""
Production FastAPI Model Serving API for LLM Optimization Suite
Author: Pruthviraj Ganihat (B.Tech AI Engineering)

Features:
- REST API endpoint for real-time text generation (/generate)
- Model variants: FP32 baseline, INT8 dynamic, INT4 quanto backbone
- Health check (/health) and system status (/info) endpoints
- Pydantic input validation & error handling
"""

import time
import torch
import torch.nn as nn
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pytorch_utils import Conv1D

# Initialize FastAPI application
app = FastAPI(
    title="LLM Optimization Inference API",
    description="Production REST API serving optimized GPT-2 / DistilGPT2 models with dynamic quantization.",
    version="1.0.0"
)

# Global model & tokenizer cache
MODEL_CACHE = {}
TOKENIZER = None
DEFAULT_MODEL_NAME = "gpt2"


class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000, example="Artificial intelligence is transforming")
    max_new_tokens: int = Field(default=30, ge=1, le=256, description="Maximum new tokens to generate")
    variant: str = Field(default="FP32", description="Optimization variant: 'FP32', 'INT8', or 'INT4'")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="Sampling temperature")
    do_sample: bool = Field(default=False, description="Whether to use greedy or sampling decoding")


class GenerationResponse(BaseModel):
    prompt: str
    generated_text: str
    variant: str
    tokens_generated: int
    latency_seconds: float
    throughput_tokens_per_sec: float


def conv1d_to_linear(module):
    """Converts HuggingFace Conv1D layers to nn.Linear for quanto INT4 compatibility."""
    for name, child in module.named_children():
        if child.__class__.__name__ == "Conv1D":
            in_features, out_features = child.weight.shape
            linear = nn.Linear(in_features, out_features, bias=(child.bias is not None))
            with torch.no_grad():
                linear.weight.copy_(child.weight.t())
                if child.bias is not None:
                    linear.bias.copy_(child.bias)
            setattr(module, name, linear)
        else:
            conv1d_to_linear(child)


def load_model_variant(variant: str):
    """Lazy-loads and caches requested model variant."""
    global TOKENIZER, MODEL_CACHE

    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
        TOKENIZER.pad_token = TOKENIZER.eos_token

    if variant in MODEL_CACHE:
        return MODEL_CACHE[variant], TOKENIZER

    print(f"Loading model variant: {variant} ...")
    if variant == "FP32":
        model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_NAME).eval()

    elif variant == "INT8":
        import torch.quantization as tq
        fp32 = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_NAME).eval()
        model = tq.quantize_dynamic(fp32, {Conv1D}, dtype=torch.qint8)

    elif variant == "INT4":
        from quanto import quantize, freeze, qint4
        model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_NAME).eval()
        conv1d_to_linear(model.transformer)
        for block in model.transformer.h:
            quantize(block, weights=qint4, activations=None)
        freeze(model)

    else:
        raise ValueError(f"Unsupported variant: {variant}. Supported: 'FP32', 'INT8', 'INT4'")

    MODEL_CACHE[variant] = model
    return model, TOKENIZER


@app.on_event("startup")
async def startup_event():
    """Pre-warm baseline FP32 model on startup."""
    print("Pre-warming baseline FP32 model...")
    load_model_variant("FP32")


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Service health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "cached_models": list(MODEL_CACHE.keys())
    }


@app.get("/info", status_code=status.HTTP_200_OK)
def system_info():
    """Return model and environment details."""
    return {
        "base_model": DEFAULT_MODEL_NAME,
        "supported_variants": ["FP32", "INT8", "INT4"],
        "device": "cpu",
        "pytorch_version": torch.__version__
    }


@app.post("/generate", response_model=GenerationResponse, status_code=status.HTTP_200_OK)
def generate_text(req: GenerationRequest):
    """Generate text endpoint given a prompt and configuration."""
    try:
        model, tokenizer = load_model_variant(req.variant)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load variant '{req.variant}': {str(e)}"
        )

    inputs = tokenizer(req.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    start_time = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=req.max_new_tokens,
            do_sample=req.do_sample,
            temperature=req.temperature if req.do_sample else 1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    latency = time.perf_counter() - start_time

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    num_generated_tokens = output_ids.shape[1] - input_ids.shape[1]
    throughput = num_generated_tokens / max(latency, 1e-5)

    return GenerationResponse(
        prompt=req.prompt,
        generated_text=generated_text,
        variant=req.variant,
        tokens_generated=num_generated_tokens,
        latency_seconds=round(latency, 4),
        throughput_tokens_per_sec=round(throughput, 2)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
