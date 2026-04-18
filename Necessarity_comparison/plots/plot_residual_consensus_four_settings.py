#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
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
TRACKING_SUITE_DIR = ROOT / "Necessarity_comparison" / "run" / (
    "20260325_necessarity_ising5q_4x4_path_cond200_e20000_lr1e-3_log40_s3_fixlr_fig3"
)
CONSENSUS_SUITE_DIR = ROOT / "Necessarity_comparison" / "run" / (
    "20260328_consensus_adamX_adamZ_ising5q_4x4_path_cond200_e30000_lr1e-3_log40_s3_fixlr_fig3"
)
RESIDUAL_FLOOR = 1.5e-2
RESIDUAL_XMAX = 20000


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    label: str
    suite_dir: Path
    color: str


CASE_SPECS = (
    CaseSpec(
        case_id="track_adamX_adamZ",
        label="track + AdamX + AdamZ",
        suite_dir=TRACKING_SUITE_DIR,
        color="#d62728",
    ),
    CaseSpec(
        case_id="track_adamX_only",
        label="track + AdamX",
        suite_dir=TRACKING_SUITE_DIR,
        color="#1f77b4",
    ),
    CaseSpec(
        case_id="track_adamZ_only",
        label="track + AdamZ",
        suite_dir=TRACKING_SUITE_DIR,
        color="#1b9e77",
    ),
    CaseSpec(
        case_id="consensus_adamX_adamZ",
        label="consensus + AdamX + AdamZ",
        suite_dir=CONSENSUS_SUITE_DIR,
        color="#7570b3",
    ),
)


def summarize_series(series: list[list[float]]) -> tuple[list[float], list[float]]:
    means: list[float] = []
    stds: list[float] = []
    for values_at_epoch in zip(*series):
        mean_value = statistics.fmean(values_at_epoch)
        means.append(mean_value)
        stds.append(statistics.stdev(values_at_epoch) if len(values_at_epoch) > 1 else 0.0)
    return means, stds


def load_metric_rows(case: CaseSpec) -> tuple[list[int], list[float], list[float], int]:
    case_dir = case.suite_dir / f"case={case.case_id}"
    metrics_paths = sorted(case_dir.glob("seed=*/metrics.jsonl"))
    if not metrics_paths:
        raise FileNotFoundError(f"No metrics files found in {case_dir}")

    epochs_ref: list[int] | None = None
    residual_series: list[list[float]] = []
    for metrics_path in metrics_paths:
        with metrics_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if not rows:
            continue
        epochs = [int(row["epoch"]) for row in rows]
        if epochs_ref is None:
            epochs_ref = epochs
        elif epochs != epochs_ref:
            raise ValueError(f"Epoch grid mismatch in {metrics_path}")
        residual_series.append([float(row["residual_norm"]) for row in rows])

    if epochs_ref is None or not residual_series:
        raise ValueError(f"No usable metric rows found in {case_dir}")
    mean_series, std_series = summarize_series(residual_series)
    return epochs_ref, mean_series, std_series, len(residual_series)


def truncate_residual_series(
    epochs_in: list[int], values_in: list[float], stds_in: list[float]
) -> tuple[list[int], list[float], list[float]]:
    epochs: list[int] = []
    values: list[float] = []
    stds: list[float] = []
    for epoch, value, std in zip(epochs_in, values_in, stds_in):
        if epoch > RESIDUAL_XMAX:
            break
        epochs.append(epoch)
        values.append(max(value, RESIDUAL_FLOOR))
        stds.append(std)
        if value <= RESIDUAL_FLOOR:
            break
    if not epochs:
        raise ValueError("No residual points survived truncation.")
    return epochs, values, stds


def build_log_ticks(y_min: float, y_max: float, *, include_floor: float | None = None) -> list[float]:
    ticks = select_sparse_log_ticks(y_min, y_max)
    if include_floor is not None and y_min <= include_floor <= y_max:
        ticks = sorted(set([*ticks, include_floor]))
    return ticks


def draw_axis(ax, series_by_case, *, y_label: str, x_max: int) -> None:
    y_min = RESIDUAL_FLOOR
    y_max = max(max(upper_band) for _, _, _, _, upper_band in series_by_case) * 1.15

    for case, epochs, values, lower_band, upper_band in series_by_case:
        ax.fill_between(
            epochs,
            lower_band,
            upper_band,
            color=case.color,
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            epochs,
            values,
            label=case.label,
            color=case.color,
            linestyle=":",
            linewidth=2.4,
            dash_capstyle="round",
            zorder=2,
        )

    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel(y_label)

    ax.set_xticks([0, 5000, 10000, 15000, 20000])
    ax.yaxis.set_major_locator(
        FixedLocator(
            build_log_ticks(
                y_min,
                y_max,
                include_floor=RESIDUAL_FLOOR,
            )
        )
    )
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.minorticks_off()

    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#303030")
        spine.set_linewidth(0.9)
    ax.tick_params(axis="both", which="major", direction="out", length=4.2, width=0.9, color="#303030")


def make_plot() -> tuple[Path, Path, dict[str, int]]:
    configure_style()

    residual_series = []
    run_counts: dict[str, int] = {}
    for case in CASE_SPECS:
        epochs, values, stds, run_count = load_metric_rows(case)
        residual_epochs, residual_values, residual_stds = truncate_residual_series(epochs, values, stds)
        residual_lower = [max(value - std, RESIDUAL_FLOOR) for value, std in zip(residual_values, residual_stds)]
        residual_upper = [max(value + std, RESIDUAL_FLOOR) for value, std in zip(residual_values, residual_stds)]
        residual_series.append((case, residual_epochs, residual_values, residual_lower, residual_upper))
        run_counts[case.case_id] = run_count

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 6.2), facecolor="white")
    draw_axis(
        ax,
        residual_series,
        y_label=r"residual norm $\|Ax-b\|$",
        x_max=RESIDUAL_XMAX,
    )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        handletextpad=0.4,
        columnspacing=1.2,
        fontsize=11,
    )
    fig.subplots_adjust(left=0.15, right=0.965, bottom=0.13, top=0.82)

    png_path = PLOT_DIR / "residual_four_settings_compare.png"
    pdf_path = PLOT_DIR / "residual_four_settings_compare.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path, run_counts


def main() -> None:
    png_path, pdf_path, run_counts = make_plot()
    for case in CASE_SPECS:
        print(f"{case.case_id}: runs={run_counts[case.case_id]}")
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
