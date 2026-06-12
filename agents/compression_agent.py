"""Compression Agent.

This lightweight agent is a dispatcher around the model-preparation functions in
the compression package. It does not benchmark or analyze results; it only
returns the requested model variant.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompressionMethod:
    """Metadata and loader for one experiment method."""

    name: str
    description: str
    loader: Callable[[str], Any]


class CompressionAgent:
    """Loads baseline or compressed TinyLlama variants on request."""

    def __init__(self) -> None:
        self.unavailable_methods: dict[str, str] = {}
        self.methods: dict[str, CompressionMethod] = {
            "fp32": CompressionMethod(
                name="fp32",
                description="Uncompressed FP32 reference model",
                loader=self._load_fp32,
            ),
            "pruned": CompressionMethod(
                name="pruned",
                description="Unstructured magnitude-pruned model",
                loader=self._load_pruned,
            ),
        }

    def supported_methods(self) -> tuple[str, ...]:
        """Return method names supported by this agent."""

        return tuple(self.methods)

    def get_method(self, method: str) -> CompressionMethod:
        """Return method metadata or raise a clear error."""

        normalized = method.lower()
        if normalized not in self.methods:
            supported = ", ".join(self.supported_methods())
            raise ValueError(f"Unsupported compression method '{method}'. Supported: {supported}")
        return self.methods[normalized]

    def load_model(self, method: str, device: str) -> Any:
        """Load the requested baseline/compressed model variant."""

        selected_method = self.get_method(method)
        try:
            return selected_method.loader(device)
        except Exception as exc:
            self.unavailable_methods[selected_method.name] = str(exc)
            LOGGER.exception("Compression method '%s' failed while loading.", selected_method.name)
            raise

    def is_available(self, method: str) -> bool:
        """Return whether a method has not failed in this process."""

        return method.lower() not in self.unavailable_methods

    @staticmethod
    def _load_fp32(device: str) -> Any:
        """Load the uncompressed FP32 reference model."""

        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        model.to(device)
        model.eval()
        return model

    @staticmethod
    def _load_pruned(device: str) -> Any:
        """Lazy-load pruning code and return a pruned model."""

        from compression.pruning import load_tinyllama_pruned

        return load_tinyllama_pruned(device)
