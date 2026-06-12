"""INT8 RTN quantization for TinyLlama using Quanto.

This module prepares the first compressed model variant for the lab. It does
not benchmark the model directly; it returns a normal Hugging Face causal LM
that can be passed into the existing benchmark helpers.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def select_device(device: str | None = None) -> str:
    """Return the requested device, or choose CUDA when it is available."""

    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tinyllama_int8_rtn(device: str | None = None) -> Any:
    """Load TinyLlama and apply INT8 round-to-nearest weight quantization.

    Quanto's low-level API first wraps supported layers with quantized
    equivalents. We use INT8 weights and leave activations unquantized so this
    is a simple RTN-style weight-only compression path that needs no calibration
    data. The final freeze step materializes the quantized weights for
    inference, making the returned model ready for benchmarking.
    """

    freeze, qint8, quantize = _load_quanto()
    resolved_device = select_device(device)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if resolved_device == "cuda" else torch.float32,
    )
    model.to(resolved_device)
    model.eval()

    # RTN is a calibration-free path: each supported weight tensor is rounded
    # to the nearest representable INT8 value by Quanto.
    quantize(model, weights=qint8, activations=None)

    # Freeze replaces dynamic quantized weights with inference-ready quantized
    # weights so benchmark runs measure the compressed model path.
    freeze(model)
    model.eval()

    return model


def _load_quanto() -> tuple[Any, Any, Any]:
    """Import Quanto lazily and raise a clear dependency error if unavailable."""

    try:
        from optimum.quanto import freeze, qint8, quantize
    except ModuleNotFoundError as exc:
        if exc.name == "optimum":
            raise ModuleNotFoundError(
                "RTN quantization requires `optimum-quanto`. Install project "
                "dependencies with `pip install -r requirements.txt`."
            ) from exc
        raise

    return freeze, qint8, quantize
