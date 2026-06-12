"""Magnitude pruning for TinyLlama.

Magnitude pruning removes the smallest weights by absolute value. This module
keeps the process simple and calibration-free so the pruned model can be
benchmarked against the FP32 baseline.
"""

from __future__ import annotations

from typing import Any


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_SPARSITY = 0.20


def select_device(device: str | None = None) -> str:
    """Return the requested device, or choose CUDA when it is available."""

    if device is not None:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tinyllama_pruned(
    device: str | None = None,
    sparsity: float = DEFAULT_SPARSITY,
) -> Any:
    """Load TinyLlama and apply unstructured magnitude pruning.

    The pruning pass visits every linear layer and zeros the requested fraction
    of lowest-magnitude weights. prune.remove() makes the mask permanent so the
    returned model behaves like a normal Hugging Face model during benchmarking.
    """

    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be greater than or equal to 0.0 and less than 1.0")

    import torch
    from transformers import AutoModelForCausalLM

    resolved_device = select_device(device)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if resolved_device == "cuda" else torch.float32,
    )
    model.to(resolved_device)
    model.eval()

    apply_magnitude_pruning(model, sparsity)
    model.eval()
    return model


def apply_magnitude_pruning(model: Any, sparsity: float) -> None:
    """Apply L1 unstructured pruning to every linear weight matrix in-place."""

    from torch import nn
    from torch.nn.utils import prune

    for module in model.modules():
        if isinstance(module, nn.Linear):
            # L1 unstructured pruning zeros individual weights with the smallest
            # absolute values, which is the standard magnitude-pruning baseline.
            prune.l1_unstructured(module, name="weight", amount=sparsity)
            prune.remove(module, "weight")
