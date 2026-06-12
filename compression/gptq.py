"""GPTQ 4-bit quantization for TinyLlama.

GPTQ is a post-training quantization method that uses a small calibration
dataset to approximate each layer's weight reconstruction error while reducing
the model to low-bit weights. This module only prepares the compressed model;
benchmarking stays in the benchmark package.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_CALIBRATION_DATA = [
    "What is machine learning?",
    "Explain the difference between latency and throughput.",
    "Summarize why model compression is useful for language models.",
    "Write a short answer about neural networks.",
]


def select_device(device: str | None = None) -> str:
    """Return the requested device, or choose CUDA when it is available."""

    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tinyllama_gptq(
    device: str | None = None,
    calibration_dataset: Sequence[str] | None = None,
    bits: int = 4,
    group_size: int = 128,
) -> Any:
    """Load TinyLlama and apply GPTQ 4-bit quantization.

    GPTQ needs representative calibration text so it can estimate quantization
    error while compressing weights. The default dataset is intentionally tiny
    for a lab smoke test; real experiments should pass a larger calibration set.
    """

    resolved_device = select_device(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gptq_config = GPTQConfig(
        bits=bits,
        dataset=list(calibration_dataset or DEFAULT_CALIBRATION_DATA),
        tokenizer=tokenizer,
        group_size=group_size,
    )

    load_kwargs: dict[str, Any] = {"quantization_config": gptq_config}
    if resolved_device == "cuda":
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
    model.eval()
    return model
