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
RUN_DIR = PROJECT_DIR / "run/20260323_graph_compare_ising5q_3x3_e20000_lr1e-3_log40_s3_fixlr_fig3"
TARGET_ITERATION = 20000
PLOT_MAX_ITERATION = 10000
TOPOLOGY_ORDER = ("path", "ring", "complete")
TOPOLOGY_COLORS = {
    "path": "#1f77b4",
    "ring": "#d95f02",
    "complete": "#1b9e77",
}
Y_LABEL = r"relative $L_2$ error of solution"


@dataclass
class BenchmarkAverage:
    benchmark_id: str
    benchmark_name: str
    row_topology: str
    column_topology: str
    epochs: list[int]
    mean_l2_error: list[float]
    seeds: list[int]


@dataclass(frozen=True)
class SlicePlotSpec:
    fixed_axis: str
    fixed_topology: str
    varying_axis: str
    output_stem: str


PLOTS = (
    SlicePlotSpec(
        fixed_axis="row",
        fixed_topology="path",
        varying_axis="column",
        output_stem="l2_error_row_fixed_path_compare_columns",
    ),
    SlicePlotSpec(
        fixed_axis="row",
        fixed_topology="complete",
        varying_axis="column",
        output_stem="l2_error_row_fixed_complete_compare_columns",
    ),
    SlicePlotSpec(
        fixed_axis="column",
        fixed_topology="path",
        varying_axis="row",
        output_stem="l2_error_column_fixed_path_compare_rows",
    ),
    SlicePlotSpec(
        fixed_axis="column",
        fixed_topology="complete",
        varying_axis="row",
        output_stem="l2_error_column_fixed_complete_compare_rows",
    ),
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 16,
            "axes.titlesize": 13,
            "legend.fontsize": 13,
            "legend.title_fontsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 0.9,
            "mathtext.fontset": "cm",
            "savefig.dpi": 300,
        }
    )


def benchmark_sort_key(benchmark_dir: Path) -> int:
    benchmark_id = benchmark_dir.name.split("=", 1)[1].split("_", 1)[0]
    return int(benchmark_id[1:])


def seed_sort_key(metrics_path: Path) -> int:
    return int(metrics_path.parent.name.split("=", 1)[1])


def normalize_topology(token: str) -> str:
    mapping = {"path": "path", "ring": "ring", "comp": "complete"}
    return mapping[token.lower()]


def parse_benchmark_name(benchmark_name: str) -> tuple[str, str, str]:
    benchmark_id, row_token, column_token = benchmark_name.split("_")
    row_topology = normalize_topology(row_token.split("-", 1)[1])
    column_topology = normalize_topology(column_token.split("-", 1)[1])
    return benchmark_id, row_topology, column_topology


def average_series(series: list[list[float]]) -> list[float]:
    return [sum(values_at_epoch) / len(values_at_epoch) for values_at_epoch in zip(*series)]


def load_benchmark_averages() -> list[BenchmarkAverage]:
    benchmark_averages: list[BenchmarkAverage] = []
    for benchmark_dir in sorted(RUN_DIR.glob("benchmark=*"), key=benchmark_sort_key):
        benchmark_name = benchmark_dir.name.split("=", 1)[1]
        benchmark_id, row_topology, column_topology = parse_benchmark_name(benchmark_name)

        epochs_ref: list[int] | None = None
        seed_series: list[list[float]] = []
        seeds: list[int] = []

        for metrics_path in sorted(benchmark_dir.glob("seed=*/metrics.jsonl"), key=seed_sort_key):
            with metrics_path.open("r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            if not rows:
                continue

            epochs = [int(row["epoch"]) for row in rows]
            if max(epochs) < TARGET_ITERATION:
                continue

            if epochs_ref is None:
                epochs_ref = epochs
            elif epochs != epochs_ref:
                raise ValueError(f"epoch grid mismatch in {metrics_path}")

            l2_values = [float(row["l2_error"]) for row in rows]
            if any(value <= 0 for value in l2_values):
                raise ValueError(f"log-scale plot requires positive l2_error values in {metrics_path}")

            seed_series.append(l2_values)
            seeds.append(seed_sort_key(metrics_path))

        if epochs_ref is None or not seed_series:
            raise ValueError(f"no complete runs found for {benchmark_dir}")

        benchmark_averages.append(
            BenchmarkAverage(
                benchmark_id=benchmark_id,
                benchmark_name=benchmark_name,
                row_topology=row_topology,
                column_topology=column_topology,
                epochs=epochs_ref,
                mean_l2_error=average_series(seed_series),
                seeds=seeds,
            )
        )

    return benchmark_averages


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


def topology_sort_key(benchmark: BenchmarkAverage, axis: str) -> int:
    topology = benchmark.row_topology if axis == "row" else benchmark.column_topology
    return TOPOLOGY_ORDER.index(topology)


def topology_symbol(topology: str) -> str:
    return {
        "path": "P4",
        "ring": "C4",
        "complete": "K4",
    }[topology]


def plot_slice(plot_spec: SlicePlotSpec, benchmark_averages: list[BenchmarkAverage]) -> tuple[Path, Path]:
    configure_style()

    selected = [
        benchmark
        for benchmark in benchmark_averages
        if (
            benchmark.row_topology if plot_spec.fixed_axis == "row" else benchmark.column_topology
        )
        == plot_spec.fixed_topology
    ]
    selected.sort(key=lambda benchmark: topology_sort_key(benchmark, plot_spec.varying_axis))

    if len(selected) != 3:
        raise ValueError(f"expected 3 curves for fixed {plot_spec.fixed_axis}={plot_spec.fixed_topology}, found {len(selected)}")

    truncated_curves: list[tuple[BenchmarkAverage, list[int], list[float]]] = []
    for benchmark in selected:
        filtered = [
            (epoch, value)
            for epoch, value in zip(benchmark.epochs, benchmark.mean_l2_error)
            if epoch <= PLOT_MAX_ITERATION
        ]
        if not filtered:
            raise ValueError(f"no l2_error points within iteration <= {PLOT_MAX_ITERATION} for {benchmark.benchmark_name}")
        epochs, values = zip(*filtered)
        truncated_curves.append((benchmark, list(epochs), list(values)))

    min_value = min(min(values) for _, _, values in truncated_curves)
    max_value = max(max(values) for _, _, values in truncated_curves)
    y_min = 10.0 ** math.floor(math.log10(min_value))
    y_max = max_value * 1.15

    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True, facecolor="white")
    for benchmark, epochs, values in truncated_curves:
        varying_topology = benchmark.column_topology if plot_spec.varying_axis == "column" else benchmark.row_topology
        ax.plot(
            epochs,
            values,
            label=f"row={topology_symbol(benchmark.row_topology)}, col={topology_symbol(benchmark.column_topology)}",
            color=TOPOLOGY_COLORS[varying_topology],
            linestyle=":",
            linewidth=2.4,
            dash_capstyle="round",
        )

    ax.set_xlim(0, PLOT_MAX_ITERATION)
    ax.set_ylim(y_min, y_max)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel(Y_LABEL)

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
        title="",
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        ncol=1,
        frameon=False,
        handlelength=2.0,
        handletextpad=0.4,
    )

    png_path = PLOT_DIR / f"{plot_spec.output_stem}.png"
    pdf_path = PLOT_DIR / f"{plot_spec.output_stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    benchmark_averages = load_benchmark_averages()
    for benchmark in benchmark_averages:
        seeds_text = ", ".join(str(seed) for seed in benchmark.seeds)
        print(f"{benchmark.benchmark_id}: averaged over {len(benchmark.seeds)} seeds ({seeds_text})")

    for plot_spec in PLOTS:
        png_path, pdf_path = plot_slice(plot_spec, benchmark_averages)
        print(f"saved {png_path}")
        print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
