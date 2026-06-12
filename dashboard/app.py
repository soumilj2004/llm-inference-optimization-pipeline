"""Streamlit dashboard for the Agentic LLM Compression Lab."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
AGENT_RESULTS_PATH = RESULTS_DIR / "agent_experiment_results.json"
AGENT_REPORT_PATH = RESULTS_DIR / "agent_experiment_report.txt"
COMPARISON_RESULTS_PATH = RESULTS_DIR / "comparison_results.json"
BASELINE_RESULTS_PATH = RESULTS_DIR / "benchmark_results.json"
ACTIVE_METHODS = ("fp32", "pruned")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from benchmark.memory import calculate_ram_delta_mb


st.set_page_config(
    page_title="Agentic LLM Compression Lab",
    layout="wide",
)


def main() -> None:
    """Render the dashboard shell and selected page."""

    st.sidebar.title("Compression Lab")
    page = st.sidebar.radio(
        "Navigate",
        ("Overview", "Run Experiment", "Results", "Visualizations", "Agent Workflow"),
    )

    if page == "Overview":
        render_overview()
    elif page == "Run Experiment":
        render_run_experiment()
    elif page == "Results":
        render_results()
    elif page == "Visualizations":
        render_visualizations()
    else:
        render_agent_workflow()


def render_overview() -> None:
    """Show project summary and supported methods."""

    st.title("Agentic LLM Compression Lab")
    st.write(
        "A lightweight lab for benchmarking TinyLlama and a magnitude-pruned variant, "
        "evaluating quality against FP32 references, and summarizing tradeoffs "
        "with a simple agent workflow."
    )

    cols = st.columns(2)
    methods = [
        ("FP32", "Uncompressed reference model"),
        ("Pruned", "Magnitude pruning baseline"),
    ]
    for column, (name, description) in zip(cols, methods):
        with column:
            st.metric(name, "Supported")
            st.caption(description)

    st.subheader("Agent Architecture")
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR;
          node [shape=box, style="rounded,filled", color="#1f2937", fillcolor="#f8fafc"];
          User -> "Experiment Manager";
          "Experiment Manager" -> "Compression Agent";
          "Compression Agent" -> "Benchmark Agent";
          "Benchmark Agent" -> "Analysis Agent";
          "Analysis Agent" -> "Final Report";
        }
        """
    )


def render_run_experiment() -> None:
    """Run the full agent experiment from the dashboard."""

    st.title("Run Experiment")
    st.write(
        "Launches the existing `run_experiment()` entry point. This can take a "
        "long time because each model variant may need to load, prune, and run "
        "the quality prompt set."
    )

    if st.button("Run Compression Experiment", type="primary"):
        progress = st.progress(0)
        status = st.empty()

        try:
            status.info("Importing agent workflow...")
            progress.progress(10)
            from agents.experiment_manager import run_experiment

            status.info("Running experiment. Model loading, pruning, and quality evaluation may take a while...")
            progress.progress(35)
            outcome = run_experiment()
            progress.progress(100)

            status.success("Experiment complete.")
            st.write("Results saved to:", outcome["results_path"])
            st.write("Report saved to:", outcome["report_path"])
            st.subheader("Generated Report")
            st.text(outcome["report"])
        except Exception as exc:  # pragma: no cover - dashboard display path
            status.error("Experiment failed.")
            st.exception(exc)

    st.subheader("Current Output Files")
    st.write(f"Results folder: `{RESULTS_DIR}`")
    for path in sorted(RESULTS_DIR.glob("*")):
        if path.is_file():
            st.write(f"- `{path.name}`")


def render_results() -> None:
    """Display benchmark rows and generated text report."""

    st.title("Results")
    results = load_results()

    if not results:
        st.warning("No result files were found in the results folder yet.")
        return

    st.subheader("Benchmark Results")
    st.dataframe(table_rows(results), use_container_width=True)

    failed_rows = [row for row in results if row.get("status") == "failed"]
    if failed_rows:
        st.subheader("Failed Methods")
        for row in failed_rows:
            st.error(f"{row.get('method', 'unknown')} failed: {row.get('error', 'Unknown error')}")

    st.subheader("Quality Scores")
    quality_rows = [
        {
            "method": row.get("method", "fp32"),
            "status": row.get("status", "completed"),
            "quality_score": row.get("quality_score"),
        }
        for row in results
    ]
    st.dataframe(quality_rows, use_container_width=True)

    st.subheader("Generated Report")
    report = build_active_report(results)
    if report:
        st.text(report)
    else:
        st.info("No generated agent report found yet.")


def render_visualizations() -> None:
    """Render simple metric comparison charts."""

    st.title("Visualizations")
    results = load_results()

    if not results:
        st.warning("No results available for visualization.")
        return

    rows = [row for row in table_rows(results) if row.get("status") != "failed"]
    failed_rows = [row for row in results if row.get("status") == "failed"]
    if failed_rows:
        st.warning(
            "Some methods failed and are excluded from charts: "
            + ", ".join(str(row.get("method", "unknown")) for row in failed_rows)
        )
    chart_specs = [
        ("Latency Comparison", "inference_latency_seconds"),
        ("Throughput Comparison", "tokens_per_second"),
        ("Memory Comparison", "model_load_ram_delta_mb"),
        ("Quality Comparison", "quality_score"),
    ]

    for title, metric in chart_specs:
        st.subheader(title)
        chart_data = {
            row["method"]: row.get(metric)
            for row in rows
            if row.get(metric) is not None
        }
        if chart_data:
            st.bar_chart(chart_data)
        else:
            st.info(f"`{metric}` is not available in the current result file.")


def render_agent_workflow() -> None:
    """Show a visual and textual view of the agent workflow."""

    st.title("Agent Workflow")
    st.graphviz_chart(
        """
        digraph {
          rankdir=TB;
          node [shape=box, style="rounded,filled", color="#334155", fillcolor="#eef2ff"];
          "Experiment Manager" -> "Compression Agent" [label="method list"];
          "Compression Agent" -> "FP32";
          "Compression Agent" -> "Pruning";
          "Compression Agent" -> "Benchmark Agent" [label="prepared model"];
          "Benchmark Agent" -> "Quality Evaluator";
          "Benchmark Agent" -> "Analysis Agent" [label="metrics + quality"];
          "Quality Evaluator" -> "Analysis Agent" [label="quality_score"];
          "Analysis Agent" -> "Report";
        }
        """
    )

    st.markdown(
        """
        **Experiment Manager** coordinates the full run.

        **Compression Agent** loads the requested FP32 or pruned model.

        **Benchmark Agent** measures latency, memory, throughput, and attaches quality results.

        **Quality Evaluator** compares model outputs with FP32 references using deterministic local scoring.

        **Analysis Agent** recommends the fastest, smallest, highest-throughput, highest-quality, and best tradeoff methods.
        """
    )


def load_results() -> list[dict[str, Any]]:
    """Load the newest available result file with graceful fallbacks."""

    for path in (AGENT_RESULTS_PATH, COMPARISON_RESULTS_PATH, BASELINE_RESULTS_PATH):
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return active_results(data)
            if isinstance(data, dict):
                return active_results([normalize_baseline_result(data)])
    return []


def active_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only methods in the simplified active workflow."""

    return [row for row in results if row.get("method", "fp32") in ACTIVE_METHODS]


def build_active_report(results: list[dict[str, Any]]) -> str:
    """Build a report from active results so stale removed methods are hidden."""

    try:
        from agents.analysis_agent import AnalysisAgent

        return AnalysisAgent().generate_report(results)
    except Exception:
        if AGENT_REPORT_PATH.exists():
            return AGENT_REPORT_PATH.read_text(encoding="utf-8")
        return ""


def normalize_baseline_result(result: dict[str, Any]) -> dict[str, Any]:
    """Map the original baseline output into the comparison result shape."""

    normalized = dict(result)
    normalized.setdefault("method", "fp32")
    normalized.setdefault("status", "completed")
    normalized.setdefault("error", None)
    normalized.setdefault("description", "Baseline benchmark result")
    if "model_load_ram_delta_mb" not in normalized:
        before = normalized.get("ram_usage_before_model_load_mb", 0.0)
        after = normalized.get("ram_usage_after_model_load_mb", 0.0)
        normalized["model_load_ram_delta_mb"] = calculate_ram_delta_mb(before, after)
    normalized.setdefault("quality_score", None)
    return normalized


def table_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compact rows for tables and charts."""

    keys = [
        "method",
        "status",
        "error",
        "description",
        "device",
        "parameter_count",
        "model_load_ram_delta_mb",
        "inference_latency_seconds",
        "generated_token_count",
        "tokens_per_second",
        "quality_score",
    ]
    return [{key: row.get(key) for key in keys} for row in results]


if __name__ == "__main__":
    main()
