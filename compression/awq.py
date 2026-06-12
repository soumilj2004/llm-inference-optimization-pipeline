"""AWQ 4-bit quantization for TinyLlama.

AWQ, or Activation-aware Weight Quantization, keeps a small set of important
weights in higher precision while quantizing most weights to 4-bit values. This
module prepares an AWQ-compressed model and leaves benchmarking to the benchmark
package.
"""

from __future__ import annotations

from typing import Any

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
AWQ_QUANT_CONFIG = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}


def select_device(device: str | None = None) -> str:
    """Return the requested device, or choose CUDA when it is available."""

    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tinyllama_awq(device: str | None = None) -> Any:
    """Load TinyLlama and apply AWQ 4-bit quantization.

    AutoAWQ computes activation-aware scales from tokenizer-driven calibration
    samples during model.quantize(). The returned object wraps the underlying
    Hugging Face model and can be used for generation/benchmarking.
    """

    resolved_device = select_device(device)
    model = AutoAWQForCausalLM.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True,
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # AWQ is calibration-based: quantize() observes activations from sample text
    # and preserves the most important weights while packing most weights to 4-bit.
    model.quantize(tokenizer, quant_config=AWQ_QUANT_CONFIG)

    if resolved_device == "cuda" and hasattr(model, "model"):
        model.model.to("cuda")
    if hasattr(model, "model"):
        model.model.eval()

    return model
