"""Baseline benchmark for TinyLlama.

This script intentionally does not implement agents, quantization, pruning, or
dashboard logic. It loads the baseline model, runs one prompt, records core
performance metrics, and writes the result as JSON.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.latency import measure_latency
from benchmark.memory import count_parameters, get_ram_usage_mb
from benchmark.throughput import calculate_tokens_per_second


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "What is machine learning?"
RESULTS_PATH = Path("results/benchmark_results.json")
MAX_NEW_TOKENS = 64


def select_device() -> str:
    """Choose the best available device for the baseline run."""

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(model_name: str, device: str) -> tuple[Any, Any, str]:
    """Load TinyLlama and its tokenizer for the baseline benchmark."""

    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()
    return model, tokenizer, str(dtype).replace("torch.", "")


def generate_response(model: Any, tokenizer: Any, prompt: str, device: str) -> dict[str, Any]:
    """Generate one deterministic response and return token-level details."""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_token_count = int(inputs["input_ids"].shape[-1])

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    if device == "cuda":
        torch.cuda.synchronize()

    generated_ids = output_ids[0][input_token_count:]
    generated_token_count = int(generated_ids.shape[-1])
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return {
        "response": response,
        "input_token_count": input_token_count,
        "generated_token_count": generated_token_count,
    }


def save_results(results: dict[str, Any], output_path: Path) -> None:
    """Persist benchmark results to a JSON file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def run_baseline_benchmark() -> dict[str, Any]:
    """Run the complete baseline benchmark pipeline."""

    device = select_device()
    ram_before_model_load_mb = get_ram_usage_mb()

    model, tokenizer, dtype = load_model_and_tokenizer(MODEL_NAME, device)
    ram_after_model_load_mb = get_ram_usage_mb()
    parameter_count = count_parameters(model)

    if device == "cuda":
        torch.cuda.synchronize()
    generation_result, inference_latency_seconds = measure_latency(
        lambda: generate_response(model, tokenizer, PROMPT, device)
    )

    generated_token_count = generation_result["generated_token_count"]
    tokens_per_second = calculate_tokens_per_second(
        generated_token_count,
        inference_latency_seconds,
    )

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "prompt": PROMPT,
        "response": generation_result["response"],
        "device": device,
        "dtype": dtype,
        "parameter_count": parameter_count,
        "ram_usage_before_model_load_mb": ram_before_model_load_mb,
        "ram_usage_after_model_load_mb": ram_after_model_load_mb,
        "inference_latency_seconds": round(inference_latency_seconds, 4),
        "input_token_count": generation_result["input_token_count"],
        "generated_token_count": generated_token_count,
        "tokens_per_second": tokens_per_second,
    }


def main() -> int:
    """Entry point used when running `python main.py`."""

    results = run_baseline_benchmark()
    save_results(results, RESULTS_PATH)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
