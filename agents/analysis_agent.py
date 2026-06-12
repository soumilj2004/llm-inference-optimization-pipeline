"""Analysis Agent.

This agent turns raw benchmark rows into a compact experiment summary and a
human-readable report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.memory import calculate_ram_delta_mb


class AnalysisAgent:
    """Analyzes benchmark results across compression methods."""

    def analyze(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Find fastest, smallest, highest-throughput, and tradeoff winners."""

        if not results:
            raise ValueError("Cannot analyze an empty benchmark result list.")

        successful_results = self._successful_results(results)
        failed_results = self._failed_results(results)
        if not successful_results:
            return {
                "status": "failed",
                "failed_methods": self._failed_summary(failed_results),
                "message": "No methods completed successfully.",
            }

        fastest = min(successful_results, key=lambda item: item["inference_latency_seconds"])
        smallest = min(successful_results, key=self._model_load_ram_delta)
        best_throughput = max(successful_results, key=lambda item: item["tokens_per_second"])
        best_quality = max(successful_results, key=lambda item: self._quality_score(item))
        best_tradeoff = max(successful_results, key=self._tradeoff_score)

        return {
            "status": "completed",
            "failed_methods": self._failed_summary(failed_results),
            "fastest_method": self._summary_row(fastest),
            "smallest_method": {
                **self._summary_row(smallest),
                "model_load_ram_delta_mb": round(self._model_load_ram_delta(smallest), 2),
            },
            "best_throughput_method": self._summary_row(best_throughput),
            "best_quality_method": self._summary_row(best_quality),
            "best_overall_tradeoff": {
                **self._summary_row(best_tradeoff),
                "tradeoff_score": round(self._tradeoff_score(best_tradeoff), 6),
            },
        }

    def analyze_file(self, results_path: str | Path) -> dict[str, Any]:
        """Read benchmark results from disk and analyze them."""

        results = json.loads(Path(results_path).read_text(encoding="utf-8"))
        return self.analyze(results)

    def generate_report(self, results: list[dict[str, Any]]) -> str:
        """Generate a plain-text report from benchmark results."""

        analysis = self.analyze(results)
        lines = [
            "Agentic LLM Compression Lab - Experiment Report",
            "",
            "Methods evaluated:",
        ]
        for result in results:
            if self._is_failed(result):
                lines.append(
                    "- {method}: FAILED - {error}".format(
                        method=result["method"],
                        error=result.get("error", "Unknown error"),
                    )
                )
            else:
                lines.append(
                    "- {method}: latency={latency}s, throughput={throughput} tok/s, "
                    "quality={quality}, RAM delta={ram_delta} MB".format(
                        method=result["method"],
                        latency=result["inference_latency_seconds"],
                        throughput=result["tokens_per_second"],
                        quality=self._quality_score(result),
                        ram_delta=round(self._model_load_ram_delta(result), 2),
                    )
                )

        if analysis.get("status") == "failed":
            lines.extend(["", "No methods completed successfully."])
            return "\n".join(lines)

        lines.extend(
            [
                "",
                f"Fastest method: {analysis['fastest_method']['method']}",
                f"Smallest method by model-load RAM delta: {analysis['smallest_method']['method']}",
                f"Best throughput: {analysis['best_throughput_method']['method']}",
                f"Best quality: {analysis['best_quality_method']['method']}",
                f"Best overall tradeoff: {analysis['best_overall_tradeoff']['method']}",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def save_report(report: str, output_path: str | Path) -> None:
        """Persist the text analysis report."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")

    @staticmethod
    def _model_load_ram_delta(result: dict[str, Any]) -> float:
        if "model_load_ram_delta_mb" in result:
            if result["model_load_ram_delta_mb"] is None:
                return float("inf")
            return result["model_load_ram_delta_mb"]
        return calculate_ram_delta_mb(
            result["ram_usage_before_model_load_mb"],
            result["ram_usage_after_model_load_mb"],
        )

    @staticmethod
    def _summary_row(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "method": result["method"],
            "latency_seconds": result["inference_latency_seconds"],
            "tokens_per_second": result["tokens_per_second"],
            "quality_score": AnalysisAgent._quality_score(result),
            "parameter_count": result["parameter_count"],
        }

    def _tradeoff_score(self, result: dict[str, Any]) -> float:
        throughput = max(result["tokens_per_second"], 0.0)
        quality = max(self._quality_score(result), 0.0)
        latency = max(result["inference_latency_seconds"], 1e-9)
        ram_delta = max(self._model_load_ram_delta(result), 1e-9)
        return (throughput * quality) / (latency * ram_delta)

    @staticmethod
    def _quality_score(result: dict[str, Any]) -> float:
        score = result.get("quality_score")
        return 0.0 if score is None else float(score)

    @staticmethod
    def _is_failed(result: dict[str, Any]) -> bool:
        return result.get("status") == "failed"

    def _successful_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [result for result in results if not self._is_failed(result)]

    def _failed_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [result for result in results if self._is_failed(result)]

    @staticmethod
    def _failed_summary(results: list[dict[str, Any]]) -> list[dict[str, str]]:
        return [
            {
                "method": str(result.get("method", "unknown")),
                "error": str(result.get("error", "Unknown error")),
            }
            for result in results
        ]
