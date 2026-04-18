#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Graph_comparison.Ising_5qubits_9benchmark.plots.plot_l2_error_topology_slices import (
    configure_style,
    select_sparse_log_ticks,
)


PLOT_DIR = Path(__file__).resolve().parent
VQLS_REPORTS_DIR = ROOT / "run" / "cen_9q_7l_qjit_seedscan" / "reports"
SINGLE_AGENT_REPORTS_DIR = ROOT / "run" / "dist_single_9q_lr1e-3" / "reports"
PLOT_MAX_ITERATION = 20000
Y_LABEL = r"relative aligned $L_2$ error of solution"


@dataclass(frozen=True)
class CaseSpec:
    label: str
    source_dir: Path
    color: str
    source_kind: str
    excluded_seeds: tuple[int, ...] = ()


CASE_SPECS = (
    CaseSpec(
        label="VQLS",
        source_dir=VQLS_REPORTS_DIR,
        color="#1f77b4",
        source_kind="history",
    ),
    CaseSpec(
        label="single-agent",
        source_dir=SINGLE_AGENT_REPORTS_DIR,
        color="#d95f02",
        source_kind="metrics",
        excluded_seeds=(2, 3),
    ),
)


def average_series(series: list[list[float]]) -> list[float]:
    return [sum(values_at_epoch) / len(values_at_epoch) for values_at_epoch in zip(*series)]


def seed_sort_key(path: Path) -> int:
    for part in path.parts:
        if part.startswith("seed="):
            return int(part.split("=", 1)[1])
    raise ValueError(f"Could not find seed token in path: {path}")


def load_history_rows(case: CaseSpec) -> tuple[list[int], list[float]]:
    history_paths = sorted(case.source_dir.glob("seed=*/*/history.json"), key=seed_sort_key)
    history_paths = [path for path in history_paths if seed_sort_key(path) not in case.excluded_seeds]
    if not history_paths:
        raise FileNotFoundError(f"No history files found in {case.source_dir}")

    iterations_ref: list[int] | None = None
    l2_series: list[list[float]] = []
    for history_path in history_paths:
        rows = json.loads(history_path.read_text(encoding="utf-8"))
        if not rows:
            continue

        iterations = [int(row["iteration"]) for row in rows]
        if iterations_ref is None:
            iterations_ref = iterations
        elif iterations != iterations_ref:
            raise ValueError(f"Iteration grid mismatch in {history_path}")

        l2_values = [float(row["l2_rel_aligned"]) for row in rows]
        if any(value <= 0 for value in l2_values):
            raise ValueError(f"Log-scale plot requires positive values in {history_path}")
        l2_series.append(l2_values)

    if iterations_ref is None or not l2_series:
        raise ValueError(f"No usable history rows found in {case.source_dir}")
    return iterations_ref, average_series(l2_series)


def load_metric_rows(case: CaseSpec) -> tuple[list[int], list[float]]:
    metrics_paths = sorted(case.source_dir.glob("seed=*/*/metrics.jsonl"), key=seed_sort_key)
    metrics_paths = [path for path in metrics_paths if seed_sort_key(path) not in case.excluded_seeds]
    if not metrics_paths:
        raise FileNotFoundError(f"No metrics files found in {case.source_dir}")

    iterations_ref: list[int] | None = None
    l2_series: list[list[float]] = []
    for metrics_path in metrics_paths:
        with metrics_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if not rows:
            continue

        iterations = [int(row["iteration"]) for row in rows]
        if iterations_ref is None:
            iterations_ref = iterations
        elif iterations != iterations_ref:
            raise ValueError(f"Iteration grid mismatch in {metrics_path}")

        l2_values = [float(row["l2_rel_aligned"]) for row in rows]
        if any(value <= 0 for value in l2_values):
            raise ValueError(f"Log-scale plot requires positive values in {metrics_path}")
        l2_series.append(l2_values)

    if iterations_ref is None or not l2_series:
        raise ValueError(f"No usable metric rows found in {case.source_dir}")
    return iterations_ref, average_series(l2_series)


def load_case_series(case: CaseSpec) -> tuple[list[int], list[float]]:
    if case.source_kind == "history":
        iterations, values = load_history_rows(case)
    elif case.source_kind == "metrics":
        iterations, values = load_metric_rows(case)
    else:
        raise ValueError(f"Unsupported source kind: {case.source_kind}")

    filtered = [(iteration, value) for iteration, value in zip(iterations, values) if iteration <= PLOT_MAX_ITERATION]
    if not filtered:
        raise ValueError(f"No points within iteration <= {PLOT_MAX_ITERATION} for {case.label}")

    kept_iterations, kept_values = zip(*filtered)
    return list(kept_iterations), list(kept_values)


def draw_axis(ax, series_by_case: list[tuple[CaseSpec, list[int], list[float]]]) -> None:
    min_value = min(min(values) for _, _, values in series_by_case)
    max_value = max(max(values) for _, _, values in series_by_case)
    y_min = 10.0 ** math.floor(math.log10(min_value))
    y_max = max_value * 1.15

    for case, iterations, values in series_by_case:
        ax.plot(
            iterations,
            values,
            label=case.label,
            color=case.color,
            linestyle=":",
            linewidth=2.4,
            dash_capstyle="round",
        )

    ax.set_xlim(0, PLOT_MAX_ITERATION)
    ax.set_ylim(y_min, y_max)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel(Y_LABEL)

    ax.set_xticks([0, 5000, 10000, 15000, 20000])
    ax.yaxis.set_major_locator(FixedLocator(select_sparse_log_ticks(y_min, y_max)))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.minorticks_off()

    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#303030")
        spine.set_linewidth(0.9)
    ax.tick_params(axis="both", which="major", direction="out", length=4.2, width=0.9, color="#303030")


def make_plot() -> tuple[Path, Path]:
    configure_style()

    series_by_case = []
    for case in CASE_SPECS:
        iterations, values = load_case_series(case)
        series_by_case.append((case, iterations, values))

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 6.2), facecolor="white")
    draw_axis(ax, series_by_case)

    ax.legend(
        loc="upper right",
        frameon=False,
        handlelength=2.0,
        handletextpad=0.4,
        fontsize=plt.rcParams["axes.labelsize"],
    )
    fig.subplots_adjust(left=0.15, right=0.965, bottom=0.13, top=0.95)

    png_path = PLOT_DIR / "vqls_single_agent_l2_rel_aligned_compare.png"
    pdf_path = PLOT_DIR / "vqls_single_agent_l2_rel_aligned_compare.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    png_path, pdf_path = make_plot()
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
