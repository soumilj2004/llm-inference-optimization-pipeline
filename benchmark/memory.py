"""Memory and model-size helpers for the baseline benchmark."""

from __future__ import annotations

import gc
import time
from typing import Any

try:
    import psutil
except ModuleNotFoundError:
    psutil = None


def get_ram_usage_mb() -> float:
    """Return current process RAM usage in megabytes."""

    if psutil is None:
        raise RuntimeError("RAM measurement requires `psutil`. Install dependencies from requirements.txt.")
    process = psutil.Process()
    return round(process.memory_info().rss / (1024**2), 2)


def stabilize_memory(delay_seconds: float = 0.2) -> None:
    """Give Python a chance to release memory before taking a RAM snapshot."""

    gc.collect()
    time.sleep(delay_seconds)


def get_stable_ram_usage_mb(samples: int = 3, delay_seconds: float = 0.1) -> float:
    """Return a stable process RAM estimate using multiple RSS samples."""

    readings: list[float] = []
    for _ in range(samples):
        stabilize_memory(delay_seconds)
        readings.append(get_ram_usage_mb())
    return round(min(readings), 2)


def calculate_ram_delta_mb(before_mb: float, after_mb: float) -> float:
    """Calculate model-load RAM delta as after minus before, clamped at zero."""

    return round(max(after_mb - before_mb, 0.0), 2)


def count_parameters(model: Any) -> int:
    """Count all parameters in a loaded PyTorch model."""

    return sum(parameter.numel() for parameter in model.parameters())
