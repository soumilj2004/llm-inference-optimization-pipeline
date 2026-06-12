"""Throughput helpers for generated-token benchmarking."""

from __future__ import annotations


def calculate_tokens_per_second(generated_token_count: int, latency_seconds: float) -> float:
    """Calculate generated-token throughput."""

    if generated_token_count <= 0 or latency_seconds <= 0:
        return 0.0
    return round(generated_token_count / latency_seconds, 4)
