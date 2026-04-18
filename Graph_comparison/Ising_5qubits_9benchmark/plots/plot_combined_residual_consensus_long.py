#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, MaxNLocator

from plot_l2_error_topology_slices import (
    PLOT_DIR,
    PLOTS,
    SlicePlotSpec,
    TOPOLOGY_COLORS,
    configure_style,
    select_sparse_log_ticks,
    topology_sort_key,
    topology_symbol,
)
from plot_residual_consensus_topology_slices import METRICS, load_benchmark_metric_averages


OUTPUT_SPECS = (
    ("row", "path", "residual_consensus_row_fixed_path_compare_columns.pdf", {"B1", "B2", "B3"}),
    ("column", "path", "residual_consensus_column_fixed_path_compare_rows.pdf", {"B1", "B4", "B7"}),
)


def get_plot_spec(fixed_axis: str, fixed_topology: str) -> SlicePlotSpec:
    for plot_spec in PLOTS:
        if plot_spec.fixed_axis == fixed_axis and plot_spec.fixed_topology == fixed_topology:
            return plot_spec
    raise ValueError(f"missing plot spec for {fixed_axis}={fixed_topology}")


def truncate_selected_curves(metric_key: str, plot_spec: SlicePlotSpec, allowed_benchmark_ids: set[str]):
    metric = next(metric for metric in METRICS if metric.metric_key == metric_key)
    benchmark_averages = load_benchmark_metric_averages(metric, allowed_benchmark_ids=allowed_benchmark_ids)
    selected = [
        benchmark
        for benchmark in benchmark_averages
        if (
            benchmark.row_topology if plot_spec.fixed_axis == "row" else benchmark.column_topology
        )
        == plot_spec.fixed_topology
    ]
    selected.sort(key=lambda benchmark: topology_sort_key(benchmark, plot_spec.varying_axis))

    truncated = []
    for benchmark in selected:
        filtered = [
            (epoch, value)
            for epoch, value in zip(benchmark.epochs, benchmark.mean_values)
            if epoch <= metric.plot_max_iteration
        ]
        if not filtered:
            raise ValueError(f"no {metric_key} points for {benchmark.benchmark_name}")
        epochs, values = zip(*filtered)
        truncated.append((benchmark, list(epochs), list(values)))

    return metric, truncated


def draw_metric_axis(ax, metric_key: str, plot_spec: SlicePlotSpec, allowed_benchmark_ids: set[str]) -> None:
    metric, truncated_curves = truncate_selected_curves(metric_key, plot_spec, allowed_benchmark_ids)
    min_value = min(min(values) for _, _, values in truncated_curves)
    max_value = max(max(values) for _, _, values in truncated_curves)
    y_min = 10.0 ** math.floor(math.log10(min_value))
    y_max = max_value * 1.15

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

    ax.set_xlim(0, metric.plot_max_iteration)
    ax.set_ylim(y_min, y_max)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel(metric.y_label)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
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


def make_combined_pdf(plot_spec: SlicePlotSpec, output_filename: str, allowed_benchmark_ids: set[str]) -> Path:
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.4), constrained_layout=True, facecolor="white")
    draw_metric_axis(axes[0], "residual_norm", plot_spec, allowed_benchmark_ids)
    draw_metric_axis(axes[1], "consensus_error", plot_spec, allowed_benchmark_ids)

    pdf_path = Path(PLOT_DIR) / output_filename
    fig.savefig(pdf_path)
    plt.close(fig)
    return pdf_path


def main() -> None:
    for fixed_axis, fixed_topology, output_filename, allowed_benchmark_ids in OUTPUT_SPECS:
        plot_spec = get_plot_spec(fixed_axis, fixed_topology)
        pdf_path = make_combined_pdf(plot_spec, output_filename, allowed_benchmark_ids)
        print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
