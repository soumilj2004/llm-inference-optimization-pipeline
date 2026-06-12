"""Benchmark Agent.

This agent receives an already prepared model and measures it with the existing
benchmark infrastructure. It intentionally does not know how compression works.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.comparison import benchmark_model, load_tokenizer, save_comparison_results


class BenchmarkAgent:
    """Runs reusable benchmark logic for one or more prepared models."""

    def __init__(self, prompt: str = "What is machine learning?") -> None:
        self.prompt = prompt
        self.tokenizer = load_tokenizer()

    def benchmark(
        self,
        *,
        method: str,
        description: str,
        model: Any,
        device: str,
        ram_before_model_load_mb: float,
        ram_after_model_load_mb: float,
        quality_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Benchmark one model and return the standard comparison schema."""

        result = benchmark_model(
            method=method,
            description=description,
            model=model,
            tokenizer=self.tokenizer,
            prompt=self.prompt,
            device=device,
            ram_before_model_load_mb=ram_before_model_load_mb,
            ram_after_model_load_mb=ram_after_model_load_mb,
        )
        if quality_result is not None:
            result.update(quality_result)
        return result

    @staticmethod
    def save_results(results: list[dict[str, Any]], output_path: str | Path) -> None:
        """Persist benchmark results as JSON."""

        save_comparison_results(results, Path(output_path))
