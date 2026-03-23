#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, MaxNLocator


PLOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PLOT_DIR.parent
RUN_DIR = PROJECT_DIR / "run/20260321_graph_compare_4x4_e10000_lr5e3_log20_seed0_fixlr"
TARGET_ITERATION = 10000
FINAL_L2_ERROR_MAX = 1e-1
METRIC_KEYS = ("global_cost", "residual_norm", "l2_error", "consensus_error")
BENCHMARK_COLORS = ("#1f77b4", "#d95f02", "#1b9e77", "#b22222", "#6a3d9a", "#8c564b")


@dataclass(frozen=True)
class MetricPlotSpec:
    metric_key: str
    y_label: str
    output_stem: str


@dataclass
class BenchmarkRuns:
    label: str
    benchmark_name: str
    color: str
    epochs: list[int]
    selected_seeds: list[int]
    series_by_metric: dict[str, list[list[float]]]


PLOTS = (
    MetricPlotSpec(
        metric_key="global_cost",
        y_label="global cost",
        output_stem="topology_global_cost_comparison",
    ),
    MetricPlotSpec(
        metric_key="residual_norm",
        y_label=r"residual norm $\|Ax-b\|$",
        output_stem="topology_residual_norm_comparison",
    ),
    MetricPlotSpec(
        metric_key="l2_error",
        y_label=r"relative $L_2$ error of solution",
        output_stem="topology_relative_l2_error_comparison",
    ),
    MetricPlotSpec(
        metric_key="consensus_error",
        y_label="consensus error",
        output_stem="topology_consensus_error_comparison",
    ),
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 0.9,
            "mathtext.fontset": "cm",
            "savefig.dpi": 300,
        }
    )


def benchmark_sort_key(benchmark_dir: Path) -> int:
    benchmark_token = benchmark_dir.name.split("=", 1)[1].split("_", 1)[0]
    return int(benchmark_token[1:])


def seed_sort_key(metrics_path: Path) -> int:
    return int(metrics_path.parent.name.split("=", 1)[1])


def load_completed_benchmark_runs() -> list[BenchmarkRuns]:
    benchmark_dirs = sorted(RUN_DIR.glob("benchmark=*"), key=benchmark_sort_key)
    if len(benchmark_dirs) != len(BENCHMARK_COLORS):
        raise ValueError(f"expected {len(BENCHMARK_COLORS)} benchmarks, found {len(benchmark_dirs)}")

    benchmark_runs: list[BenchmarkRuns] = []
    for color, benchmark_dir in zip(BENCHMARK_COLORS, benchmark_dirs):
        benchmark_name = benchmark_dir.name.split("=", 1)[1]
        label = benchmark_name.split("_", 1)[0]
        epochs_ref: list[int] | None = None
        selected_seeds: list[int] = []
        series_by_metric = {metric_key: [] for metric_key in METRIC_KEYS}

        for metrics_path in sorted(benchmark_dir.glob("seed=*/metrics.jsonl"), key=seed_sort_key):
            with metrics_path.open("r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            if not rows:
                continue

            epochs = [int(row["epoch"]) for row in rows]
            if max(epochs) < TARGET_ITERATION:
                continue
            if float(rows[-1]["l2_error"]) > FINAL_L2_ERROR_MAX:
                continue

            if epochs_ref is None:
                epochs_ref = epochs
            elif epochs != epochs_ref:
                raise ValueError(f"epoch grid mismatch in {metrics_path}")

            selected_seeds.append(seed_sort_key(metrics_path))
            for metric_key in METRIC_KEYS:
                values = [float(row[metric_key]) for row in rows]
                if any(value <= 0 for value in values):
                    raise ValueError(f"log-scale plot requires positive {metric_key} values in {metrics_path}")
                series_by_metric[metric_key].append(values)

        if epochs_ref is None or not selected_seeds:
            raise ValueError(
                f"no runs passed the iteration and final l2_error filter for {benchmark_dir}"
            )

        benchmark_runs.append(
            BenchmarkRuns(
                label=label,
                benchmark_name=benchmark_name,
                color=color,
                epochs=epochs_ref,
                selected_seeds=selected_seeds,
                series_by_metric=series_by_metric,
            )
        )

    return benchmark_runs


def average_series(series: list[list[float]]) -> list[float]:
    if not series:
        raise ValueError("cannot average an empty series collection")
    return [sum(values_at_epoch) / len(values_at_epoch) for values_at_epoch in zip(*series)]


def select_sparse_log_ticks(y_min: float, y_max: float) -> list[float]:
    min_exp = math.floor(math.log10(y_min))
    max_exp = math.ceil(math.log10(y_max))
    all_ticks = [10.0**exp for exp in range(min_exp, max_exp + 1)]

    if len(all_ticks) <= 5:
        return [tick for tick in all_ticks if y_min <= tick <= y_max]

    step = max(1, math.ceil((len(all_ticks) - 1) / 4))
    major_ticks = all_ticks[::step]

    lower_tick = 10.0**min_exp
    upper_tick = 10.0**max_exp
    if lower_tick not in major_ticks:
        major_ticks.insert(0, lower_tick)
    if upper_tick not in major_ticks:
        major_ticks.append(upper_tick)

    return [tick for tick in major_ticks if y_min <= tick <= y_max]


def plot_metric(metric_plot: MetricPlotSpec, benchmark_runs: list[BenchmarkRuns]) -> tuple[Path, Path]:
    configure_style()

    averaged_curves: list[tuple[BenchmarkRuns, list[float]]] = []
    for benchmark in benchmark_runs:
        averaged_curves.append((benchmark, average_series(benchmark.series_by_metric[metric_plot.metric_key])))

    min_value = min(min(values) for _, values in averaged_curves)
    max_value = max(max(values) for _, values in averaged_curves)
    y_min = 10.0 ** math.floor(math.log10(min_value))
    y_max = max_value * 1.15

    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True, facecolor="white")
    for benchmark, values in averaged_curves:
        ax.plot(
            benchmark.epochs,
            values,
            label=benchmark.label,
            color=benchmark.color,
            linestyle=":",
            linewidth=2.4,
            dash_capstyle="round",
        )

    ax.set_xlim(0, TARGET_ITERATION)
    ax.set_ylim(y_min, y_max)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel(metric_plot.y_label)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
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

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.2,
        handletextpad=0.4,
    )

    png_path = PLOT_DIR / f"{metric_plot.output_stem}.png"
    pdf_path = PLOT_DIR / f"{metric_plot.output_stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    benchmark_runs = load_completed_benchmark_runs()
    for benchmark in benchmark_runs:
        seeds_text = ", ".join(str(seed) for seed in benchmark.selected_seeds)
        print(
            f"{benchmark.label}: averaged over {len(benchmark.selected_seeds)} selected runs "
            f"(final l2_error <= {FINAL_L2_ERROR_MAX:g}; seeds: {seeds_text})"
        )

    for metric_plot in PLOTS:
        png_path, pdf_path = plot_metric(metric_plot, benchmark_runs)
        print(f"saved {png_path}")
        print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
