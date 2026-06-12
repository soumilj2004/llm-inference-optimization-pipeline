"""Latency helpers for baseline text generation."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def measure_latency(function: Callable[[], T]) -> tuple[T, float]:
    """Run a callable once and return its result with elapsed seconds."""

    start_time = time.perf_counter()
    result = function()
    elapsed_seconds = time.perf_counter() - start_time
    return result, elapsed_seconds
