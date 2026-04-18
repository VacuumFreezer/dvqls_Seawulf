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
MAX_ITERATION = 14000
COST_FLOOR = 1e-4
OUTPUT_STEM = "global_cost_vs_iteration_partition_comparison"


@dataclass(frozen=True)
class CurveSpec:
    label: str
    metrics_paths: tuple[Path, ...]
    color: str
    linestyle: str
    final_l2_threshold: float | None = None


CURVES = (
    CurveSpec(
        label=r"$1 \times 1$",
        metrics_paths=(
            Path("run/four_partition_qjitcat_u0pi_e5000_log20_s5/1b1/seed=156597/metrics.jsonl"),
            Path("run/four_partition_qjitcat_u0pi_e5000_log20_s5/1b1/seed=566788/metrics.jsonl"),
            Path("run/four_partition_qjitcat_u0pi_e5000_log20_s5/1b1/seed=579757/metrics.jsonl"),
            Path("run/four_partition_qjitcat_u0pi_e5000_log20_s5/1b1/seed=797737/metrics.jsonl"),
            Path("run/four_partition_qjitcat_u0pi_e5000_log20_s5/1b1/seed=871892/metrics.jsonl"),
        ),
        color="#1f77b4",
        linestyle="-",
        final_l2_threshold=5e-2,
    ),
    CurveSpec(
        label=r"$2 \times 2$",
        metrics_paths=(Path("run/1b1_2b2_followup_qjitcat_u0pi_e20000_lr1e2_log20_s1/2b2/seed=566788/metrics.jsonl"),),
        color="#d95f02",
        linestyle="--",
        final_l2_threshold=5e-2,
    ),
    CurveSpec(
        label=r"$4 \times 4$",
        metrics_paths=(
            Path("run/4b4_followup_qjitcat_u0pi_e20000_lr1e2_log20_s4/4b4/seed=314271/metrics.jsonl"),
            Path("run/4b4_followup_qjitcat_u0pi_e20000_lr1e2_log20_s4/4b4/seed=482609/metrics.jsonl"),
            Path("run/4b4_followup_qjitcat_u0pi_e20000_lr1e2_log20_s4/4b4/seed=566788/metrics.jsonl"),
            Path("run/4b4_followup_qjitcat_u0pi_e20000_lr1e2_log20_s4/4b4/seed=905143/metrics.jsonl"),
        ),
        color="#1b9e77",
        linestyle="-.",
        final_l2_threshold=1e-2,
    ),
    CurveSpec(
        label=r"$8 \times 8$",
        metrics_paths=(
            Path("run/8b8_followup_qjitcat_u0pi_e20000_lr1e2_mem128_seed220296_single/seed=220296/metrics.jsonl"),
            Path("run/8b8_followup_qjitcat_u0pi_e20000_lr1e2_mem128_s2_seeds66224_365931/seed=66224/metrics.jsonl"),
            Path("run/8b8_followup_qjitcat_u0pi_e20000_lr1e2_mem128_s2_seeds66224_365931/seed=365931/metrics.jsonl"),
        ),
        color="#b22222",
        linestyle=":",
    ),
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 0.9,
            "mathtext.fontset": "cm",
            "savefig.dpi": 300,
        }
    )


def load_l2_stats(metrics_file: Path) -> tuple[float, float]:
    final_l2_error: float | None = None
    min_l2_error: float | None = None
    with metrics_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            current_l2_error = float(record["l2_error"])
            final_l2_error = current_l2_error
            if min_l2_error is None or current_l2_error < min_l2_error:
                min_l2_error = current_l2_error

    if final_l2_error is None or min_l2_error is None:
        raise ValueError(f"no l2_error data found in {metrics_file}")

    return final_l2_error, min_l2_error


def select_metrics_files(spec: CurveSpec) -> tuple[Path, ...]:
    selected: list[Path] = []
    for rel_path in spec.metrics_paths:
        metrics_file = PROJECT_DIR / rel_path
        if not metrics_file.exists():
            raise FileNotFoundError(f"missing metrics file: {metrics_file}")

        if spec.final_l2_threshold is not None:
            final_l2_error, _ = load_l2_stats(metrics_file)
        else:
            final_l2_error = None

        if spec.final_l2_threshold is not None:
            if final_l2_error > spec.final_l2_threshold:
                continue

        selected.append(metrics_file)

    return tuple(selected)


def load_metric_curve(
    spec: CurveSpec,
    metric_key: str,
    y_floor: float,
    max_iteration: int = MAX_ITERATION,
) -> tuple[list[int], list[float]]:
    values_by_epoch: dict[int, list[float]] = defaultdict(list)
    selected_files = select_metrics_files(spec)
    if not selected_files:
        raise ValueError(f"no runs kept for {spec.label}")
    for metrics_file in selected_files:
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
        raise ValueError(f"no data points found within iteration <= {max_iteration} for {spec.label}")

    epochs = sorted(values_by_epoch)
    values = [sum(values_by_epoch[epoch]) / len(values_by_epoch[epoch]) for epoch in epochs]

    first_floor_index = next((idx for idx, value in enumerate(values) if value <= y_floor), None)
    if first_floor_index is not None:
        values[first_floor_index:] = [y_floor] * (len(values) - first_floor_index)

    return epochs, values


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


def make_metric_plot(
    *,
    curves: tuple[CurveSpec, ...],
    metric_key: str,
    y_label: str,
    y_floor: float,
    output_stem: str,
    max_iteration: int = MAX_ITERATION,
) -> tuple[Path, Path]:
    configure_style()

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True, facecolor="white")
    plotted_curves: list[tuple[list[int], list[float]]] = []
    plotted_specs: list[CurveSpec] = []

    for spec in curves:
        try:
            epochs, values = load_metric_curve(
                spec,
                metric_key=metric_key,
                y_floor=y_floor,
                max_iteration=max_iteration,
            )
        except ValueError as exc:
            print(f"skipping {spec.label}: {exc}")
            continue
        plotted_curves.append((epochs, values))
        plotted_specs.append(spec)
        ax.plot(
            epochs,
            values,
            label=spec.label,
            color=spec.color,
            linestyle=spec.linestyle,
            linewidth=2.3,
            solid_capstyle="round",
        )

    if not plotted_curves:
        raise ValueError(f"no curves available for {metric_key}")

    max_value = max(max(values) for _, values in plotted_curves)
    ax.set_xlim(0, max_iteration)
    ax.set_ylim(y_floor, max_value * 1.15)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel(y_label)

    y_major_ticks = select_sparse_log_ticks(y_floor, max_value * 1.15)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.yaxis.set_major_locator(FixedLocator(y_major_ticks))
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
        ncol=len(plotted_specs),
        frameon=False,
        handlelength=2.2,
        columnspacing=1.0,
        handletextpad=0.5,
    )

    png_path = PLOT_DIR / f"{output_stem}.png"
    pdf_path = PLOT_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def make_plot() -> tuple[Path, Path]:
    return make_metric_plot(
        curves=CURVES,
        metric_key="global_cost",
        y_label="global cost",
        y_floor=COST_FLOOR,
        output_stem=OUTPUT_STEM,
    )


def main() -> None:
    for curve in CURVES:
        kept = select_metrics_files(curve)
        if kept:
            print(f"{curve.label}: averaging {len(kept)} runs")
        else:
            print(f"{curve.label}: 0 runs kept; curve omitted")
    png_path, pdf_path = make_plot()
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
