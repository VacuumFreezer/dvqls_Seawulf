#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, MaxNLocator


PLOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PLOT_DIR.parent
MAX_ITERATION = 15000


@dataclass(frozen=True)
class CurveSpec:
    label: str
    metrics_paths: tuple[Path, ...]
    color: str
    linestyle: str


CURVES = (
    CurveSpec(
        label=r"$1\times 1$",
        metrics_paths=(Path("1d1/run/brickwall3_e20000_lr1e2_nodecay_log20_s3/seed=3958/metrics.jsonl"),),
        color="#1f77b4",
        linestyle=":",
    ),
    CurveSpec(
        label=r"$2\times 2$",
        metrics_paths=(
            Path("2d2/run/old2b2style_brickwall5_u0pi_e20000_lr1e2_log20_s3/seed=3958/metrics.jsonl"),
            Path("2d2/run/old2b2style_brickwall5_u0pi_e20000_lr1e2_log20_s3/seed=566788/metrics.jsonl"),
        ),
        color="#d95f02",
        linestyle=":",
    ),
    CurveSpec(
        label=r"$4\times 4$",
        metrics_paths=(Path("4d4/run/old2b2style_brickwall5_u0pi_e20000_lr1e2_log20_seed566788_single/seed=566788/metrics.jsonl"),),
        color="#1b9e77",
        linestyle=":",
    ),
    CurveSpec(
        label=r"$8\times 8$",
        metrics_paths=(
            Path("../Old_asymmetry_stabilizer/run/8b8_followup_qjitcat_u0pi_e20000_lr1e2_mem128_seed220296_single/seed=220296/metrics.jsonl"),
            Path("../Old_asymmetry_stabilizer/run/8b8_followup_qjitcat_u0pi_e20000_lr1e2_mem128_s2_seeds66224_365931/seed=66224/metrics.jsonl"),
            Path("../Old_asymmetry_stabilizer/run/8b8_followup_qjitcat_u0pi_e20000_lr1e2_mem128_s2_seeds66224_365931/seed=365931/metrics.jsonl"),
        ),
        color="#b22222",
        linestyle=":",
    ),
)


@dataclass(frozen=True)
class MetricPlotSpec:
    metric_key: str
    y_label: str
    y_floor: float
    output_stem: str
    max_iteration: int


PLOTS = (
    MetricPlotSpec(
        metric_key="global_cost",
        y_label="global cost",
        y_floor=1e-4,
        output_stem="global_cost_vs_iteration_13stabilizer_partition_comparison",
        max_iteration=14000,
    ),
    MetricPlotSpec(
        metric_key="residual_norm",
        y_label=r"residual norm $\|Ax-b\|$",
        y_floor=1e-2,
        output_stem="residual_norm_vs_iteration_13stabilizer_partition_comparison",
        max_iteration=14000,
    ),
    MetricPlotSpec(
        metric_key="l2_error",
        y_label=r"relative $L_2$ error of solution",
        y_floor=1e-2,
        output_stem="relative_l2_error_vs_iteration_13stabilizer_partition_comparison",
        max_iteration=14000,
    ),
    MetricPlotSpec(
        metric_key="consensus_error",
        y_label="consensus error",
        y_floor=1e-6,
        output_stem="consensus_error_vs_iteration_13stabilizer_partition_comparison",
        max_iteration=1000,
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


def load_average_curve(
    spec: CurveSpec,
    metric_key: str,
    y_floor: float,
    max_iteration: int,
) -> tuple[list[int], list[float]]:
    values_by_epoch: dict[int, list[float]] = defaultdict(list)
    for rel_path in spec.metrics_paths:
        metrics_file = PROJECT_DIR / rel_path
        if not metrics_file.exists():
            raise FileNotFoundError(f"missing metrics file: {metrics_file}")

        with metrics_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                epoch = int(record["epoch"])
                if epoch > max_iteration:
                    continue
                values_by_epoch[epoch].append(float(record[metric_key]))

    if not values_by_epoch:
        raise ValueError(f"no {metric_key} data found for {spec.label} within iteration <= {max_iteration}")

    epochs = sorted(values_by_epoch)
    values = [sum(values_by_epoch[epoch]) / len(values_by_epoch[epoch]) for epoch in epochs]

    first_floor_index = next((idx for idx, value in enumerate(values) if value <= y_floor), None)
    if first_floor_index is not None:
        values[first_floor_index:] = [y_floor] * (len(values) - first_floor_index)

    return epochs, values


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#303030")
        spine.set_linewidth(0.9)
    ax.tick_params(axis="both", which="major", direction="out", length=4.2, width=0.9, color="#303030")


def draw_metric(ax: plt.Axes, plot_spec: MetricPlotSpec, *, show_legend: bool, x_nbins: int = 7) -> None:
    plotted_curves: list[tuple[list[int], list[float]]] = []
    for spec in CURVES:
        epochs, values = load_average_curve(
            spec,
            plot_spec.metric_key,
            plot_spec.y_floor,
            plot_spec.max_iteration,
        )
        plotted_curves.append((epochs, values))
        ax.plot(
            epochs,
            values,
            label=spec.label,
            color=spec.color,
            linestyle=spec.linestyle,
            linewidth=2.3,
            dash_capstyle="round",
        )

    max_value = max(max(values) for _, values in plotted_curves)
    ax.set_xlim(0, plot_spec.max_iteration)
    ax.set_ylim(plot_spec.y_floor, max_value * 1.15)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel(plot_spec.y_label)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=x_nbins, integer=True))
    ax.yaxis.set_major_locator(FixedLocator(select_sparse_log_ticks(plot_spec.y_floor, max_value * 1.15)))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.minorticks_off()

    style_axis(ax)
    if show_legend:
        ax.legend(
            title="block partition",
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            ncol=1,
            frameon=False,
            handlelength=2.2,
            handletextpad=0.5,
            fontsize=15,
            title_fontsize=15,
        )


def make_plot(plot_spec: MetricPlotSpec) -> tuple[Path, Path]:
    configure_style()

    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True, facecolor="white")
    draw_metric(ax, plot_spec, show_legend=True)

    png_path = PLOT_DIR / f"{plot_spec.output_stem}.png"
    pdf_path = PLOT_DIR / f"{plot_spec.output_stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def make_grid_plot() -> tuple[Path, Path]:
    configure_style()

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.2), facecolor="white")
    fig.subplots_adjust(right=0.82, wspace=0.28, hspace=0.3)
    layout = {
        (0, 0): next(spec for spec in PLOTS if spec.metric_key == "global_cost"),
        (0, 1): next(spec for spec in PLOTS if spec.metric_key == "residual_norm"),
        (1, 0): next(spec for spec in PLOTS if spec.metric_key == "l2_error"),
        (1, 1): next(spec for spec in PLOTS if spec.metric_key == "consensus_error"),
    }

    for (row, col), plot_spec in layout.items():
        draw_metric(axes[row, col], plot_spec, show_legend=False, x_nbins=4)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="block partition",
        loc="center left",
        bbox_to_anchor=(0.845, 0.5),
        ncol=1,
        frameon=False,
        handlelength=2.2,
        handletextpad=0.5,
        fontsize=15,
        title_fontsize=15,
    )

    png_path = PLOT_DIR / "partition_metrics_13stabilizer_grid.png"
    pdf_path = PLOT_DIR / "partition_metrics_13stabilizer_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    for curve in CURVES:
        print(f"{curve.label}: averaging {len(curve.metrics_paths)} seed paths")
    for plot_spec in PLOTS:
        png_path, pdf_path = make_plot(plot_spec)
        print(f"saved {png_path}")
        print(f"saved {pdf_path}")
    png_path, pdf_path = make_grid_plot()
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
