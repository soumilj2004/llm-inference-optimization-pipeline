"""Comparison utilities for baseline and compressed TinyLlama models.

The existing single-model benchmark remains intact. This module adds a small
registry and consistent result schema so future agents can benchmark multiple
compression methods without knowing how each model is prepared.
"""

from __future__ import annotations

import gc
import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.latency import measure_latency
from benchmark.memory import calculate_ram_delta_mb, count_parameters, get_stable_ram_usage_mb, stabilize_memory
from benchmark.throughput import calculate_tokens_per_second


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPT = "What is machine learning?"
DEFAULT_OUTPUT_PATH = Path("results/comparison_results.json")
MAX_NEW_TOKENS = 64
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompressionTarget:
    """Describes one model preparation path in the comparison registry."""

    name: str
    loader: Callable[[str], Any]
    description: str


def select_device(device: str | None = None) -> str:
    """Return the requested device, or choose CUDA when it is available."""

    if device is not None:
        return device
    torch = load_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tinyllama_fp32(device: str) -> Any:
    """Load the uncompressed FP32 comparison model."""

    torch = load_torch()
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    return model


def load_tinyllama_pruned_comparison(device: str) -> Any:
    """Lazy-load pruning support for environment-aware comparisons."""

    from compression.pruning import load_tinyllama_pruned

    return load_tinyllama_pruned(device)


COMPRESSION_TARGETS: dict[str, CompressionTarget] = {
    "fp32": CompressionTarget("fp32", load_tinyllama_fp32, "Uncompressed FP32 reference model"),
    "pruned": CompressionTarget("pruned", load_tinyllama_pruned_comparison, "Unstructured magnitude-pruned model"),
}


def compare_methods(
    methods: Iterable[str] = ("fp32", "pruned"),
    prompt: str = DEFAULT_PROMPT,
    output_path: str | Path | None = DEFAULT_OUTPUT_PATH,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Benchmark multiple compression methods with a consistent result schema."""

    resolved_device = select_device(device)
    tokenizer = load_tokenizer()
    results: list[dict[str, Any]] = []

    for method in methods:
        target = _get_target(method)
        stabilize_memory()
        ram_before_model_load_mb = get_stable_ram_usage_mb()
        model: Any | None = None
        try:
            model = target.loader(resolved_device)
            stabilize_memory()
            ram_after_model_load_mb = get_stable_ram_usage_mb()

            result = benchmark_model(
                method=target.name,
                description=target.description,
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=resolved_device,
                ram_before_model_load_mb=ram_before_model_load_mb,
                ram_after_model_load_mb=ram_after_model_load_mb,
            )
            results.append(result)
        except Exception as exc:
            LOGGER.exception("Comparison method '%s' failed. Continuing.", target.name)
            results.append(
                failed_result(
                    method=target.name,
                    description=target.description,
                    error=exc,
                    device=resolved_device,
                    ram_before_model_load_mb=ram_before_model_load_mb,
                )
            )
        finally:
            if model is not None:
                _release_model(model)

    if output_path is not None:
        save_comparison_results(results, Path(output_path))

    return results


def benchmark_model(
    method: str,
    description: str,
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: str,
    ram_before_model_load_mb: float,
    ram_after_model_load_mb: float,
) -> dict[str, Any]:
    """Benchmark one already-loaded model and return the shared result schema."""

    generation_model = unwrap_model(model)
    parameter_count = count_parameters(generation_model)
    torch = load_torch()

    if device == "cuda":
        torch.cuda.synchronize()
    generation_result, latency_seconds = measure_latency(
        lambda: generate_response(generation_model, tokenizer, prompt, device)
    )

    generated_token_count = generation_result["generated_token_count"]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "status": "completed",
        "error": None,
        "description": description,
        "model_name": MODEL_NAME,
        "prompt": prompt,
        "response": generation_result["response"],
        "device": device,
        "parameter_count": parameter_count,
        "ram_usage_before_model_load_mb": ram_before_model_load_mb,
        "ram_usage_after_model_load_mb": ram_after_model_load_mb,
        "model_load_ram_delta_mb": calculate_ram_delta_mb(
            ram_before_model_load_mb,
            ram_after_model_load_mb,
        ),
        "inference_latency_seconds": round(latency_seconds, 4),
        "input_token_count": generation_result["input_token_count"],
        "generated_token_count": generated_token_count,
        "tokens_per_second": calculate_tokens_per_second(generated_token_count, latency_seconds),
        "quality_score": None,
    }


def failed_result(
    *,
    method: str,
    description: str,
    error: Exception,
    device: str,
    ram_before_model_load_mb: float,
) -> dict[str, Any]:
    """Return a standard failed comparison row."""

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "status": "failed",
        "error": str(error),
        "description": description,
        "model_name": MODEL_NAME,
        "prompt": None,
        "response": None,
        "device": device,
        "parameter_count": None,
        "ram_usage_before_model_load_mb": ram_before_model_load_mb,
        "ram_usage_after_model_load_mb": None,
        "model_load_ram_delta_mb": None,
        "inference_latency_seconds": None,
        "input_token_count": None,
        "generated_token_count": None,
        "tokens_per_second": None,
        "quality_score": None,
    }


def load_tokenizer() -> Any:
    """Load the shared tokenizer used for every comparison method."""

    load_torch()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def generate_response(model: Any, tokenizer: Any, prompt: str, device: str) -> dict[str, Any]:
    """Generate one deterministic response and count prompt/generated tokens."""

    torch = load_torch()
    inputs = tokenizer(prompt, return_tensors="pt").to(_model_device(model, device))
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


def save_comparison_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """Persist comparison results as JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def unwrap_model(model: Any) -> Any:
    """Return the model object that supports generation."""

    if hasattr(model, "generate"):
        return model
    return getattr(model, "model", model)


def _model_device(model: Any, fallback_device: str) -> str:
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return fallback_device


def _get_target(method: str) -> CompressionTarget:
    normalized = method.lower()
    if normalized not in COMPRESSION_TARGETS:
        supported = ", ".join(sorted(COMPRESSION_TARGETS))
        raise ValueError(f"Unsupported method '{method}'. Supported methods: {supported}")
    return COMPRESSION_TARGETS[normalized]


def _release_model(model: Any) -> None:
    del model
    gc.collect()
    torch = load_torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_torch() -> Any:
    """Import PyTorch lazily so dashboard/import paths do not fail early."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required to run experiments. Install dependencies with "
            "`pip install -r requirements.txt` in the same environment used to run the project."
        ) from exc
    return torch
