"""Experiment Manager Agent.

This is the first orchestration layer for the lab. It coordinates compression,
benchmarking, and analysis without introducing an external agent framework.
"""

from __future__ import annotations

import gc
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.analysis_agent import AnalysisAgent
from agents.benchmark_agent import BenchmarkAgent
from agents.compression_agent import CompressionAgent
from benchmark.comparison import load_torch, select_device
from benchmark.memory import get_stable_ram_usage_mb, stabilize_memory
from evaluation.quality_evaluator import QualityEvaluator


DEFAULT_METHODS = ("fp32", "pruned")
DEFAULT_RESULTS_PATH = Path("results/agent_experiment_results.json")
DEFAULT_REPORT_PATH = Path("results/agent_experiment_report.txt")
LOGGER = logging.getLogger(__name__)


class ExperimentManagerAgent:
    """Runs the full compression experiment from a single entry point."""

    def __init__(
        self,
        methods: tuple[str, ...] = DEFAULT_METHODS,
        prompt: str = "What is machine learning?",
        device: str | None = None,
        results_path: str | Path = DEFAULT_RESULTS_PATH,
        report_path: str | Path = DEFAULT_REPORT_PATH,
    ) -> None:
        self.methods = self._ensure_fp32_reference_method(methods)
        self.device = select_device(device)
        self.results_path = Path(results_path)
        self.report_path = Path(report_path)
        self.compression_agent = CompressionAgent()
        self.benchmark_agent = BenchmarkAgent(prompt=prompt)
        self.analysis_agent = AnalysisAgent()
        self.quality_evaluator = QualityEvaluator()

    def run(self) -> dict[str, Any]:
        """Execute compression, benchmarking, and analysis for all methods."""

        results: list[dict[str, Any]] = []
        quality_references: dict[str, str] | None = None

        for method_name in self.methods:
            method = self.compression_agent.get_method(method_name)
            stabilize_memory()
            ram_before_model_load_mb = get_stable_ram_usage_mb()
            model: Any | None = None
            try:
                model = self.compression_agent.load_model(method.name, self.device)
                stabilize_memory()
                ram_after_model_load_mb = get_stable_ram_usage_mb()
                if method.name == "fp32":
                    quality_references = self.quality_evaluator.build_references(
                        model,
                        self.benchmark_agent.tokenizer,
                        self.device,
                    )
                    quality_result = self.quality_evaluator.evaluate_baseline(quality_references)
                else:
                    if quality_references is None:
                        raise RuntimeError("FP32 references must be generated before compressed methods.")
                    quality_result = self.quality_evaluator.evaluate_model(
                        model,
                        self.benchmark_agent.tokenizer,
                        self.device,
                        quality_references,
                    )

                result = self.benchmark_agent.benchmark(
                    method=method.name,
                    description=method.description,
                    model=model,
                    device=self.device,
                    ram_before_model_load_mb=ram_before_model_load_mb,
                    ram_after_model_load_mb=ram_after_model_load_mb,
                    quality_result=quality_result,
                )
                results.append(result)
            except Exception as exc:
                LOGGER.exception("Method '%s' failed. Continuing with remaining methods.", method.name)
                results.append(
                    self._failed_result(
                        method=method.name,
                        description=method.description,
                        error=exc,
                        ram_before_model_load_mb=ram_before_model_load_mb,
                    )
                )
            finally:
                if model is not None:
                    self._release_model(model)

        self.benchmark_agent.save_results(results, self.results_path)
        analysis = self.analysis_agent.analyze(results)
        report = self.analysis_agent.generate_report(results)
        self.analysis_agent.save_report(report, self.report_path)

        return {
            "results": results,
            "analysis": analysis,
            "report": report,
            "results_path": str(self.results_path),
            "report_path": str(self.report_path),
        }

    @staticmethod
    def _release_model(model: Any) -> None:
        """Release a model between methods to reduce memory pressure."""

        del model
        gc.collect()
        torch = load_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _failed_result(
        self,
        *,
        method: str,
        description: str,
        error: Exception,
        ram_before_model_load_mb: float,
    ) -> dict[str, Any]:
        """Create a standard failed-method result row."""

        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "description": description,
            "status": "failed",
            "error": str(error),
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "prompt": None,
            "response": None,
            "device": self.device,
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

    @staticmethod
    def _ensure_fp32_reference_method(methods: tuple[str, ...]) -> tuple[str, ...]:
        """Ensure FP32 runs first because quality scoring uses it as reference."""

        normalized = tuple(method.lower() for method in methods)
        without_fp32 = tuple(method for method in normalized if method != "fp32")
        return ("fp32", *without_fp32)


def run_experiment() -> dict[str, Any]:
    """Convenience entry point for scripts or future agent callers."""

    return ExperimentManagerAgent().run()


if __name__ == "__main__":
    outcome = run_experiment()
    print(outcome["report"])
