#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, MaxNLocator

PLOT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PLOT_DIR.parents[2]
DEFAULT_DATASET_FILE = PLOT_DIR / "dataset.md"


@dataclass(frozen=True)
class CurveSpec:
    key: str
    label: str
    metrics_paths: tuple[Path, ...]
    color: str
    linestyle: str


STYLE_BY_KEY = {
    "1d1": (r"$1 \times 1$", "#1f77b4", ":"),
    "2d2": (r"$2 \times 2$", "#d95f02", ":"),
    "4d4": (r"$4 \times 4$", "#1b9e77", ":"),
    "8d8": (r"$8 \times 8$", "#b22222", ":"),
}

DEFAULT_ORDER = ("1d1", "2d2", "4d4", "8d8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read dataset.md, average the listed runs, and plot residual norm and consensus error."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_FILE,
        help="Markdown file listing datasets grouped by section.",
    )
    parser.add_argument(
        "--output-stem",
        default="residual_consensus_from_dataset_partition_comparison",
        help="Output file stem written into the plots directory.",
    )
    parser.add_argument(
        "--residual-max-iteration",
        type=int,
        default=16000,
        help="Maximum iteration shown on the residual panel.",
    )
    parser.add_argument(
        "--consensus-max-iteration",
        type=int,
        default=1000,
        help="Maximum iteration shown on the consensus panel.",
    )
    parser.add_argument(
        "--residual-floor",
        type=float,
        default=1e-2,
        help="Lower floor for residual norm on the log-scale y-axis.",
    )
    parser.add_argument(
        "--consensus-floor",
        type=float,
        default=1e-3,
        help="Lower floor for consensus error on the log-scale y-axis.",
    )
    return parser.parse_args()


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


def parse_dataset_file(dataset_file: Path) -> tuple[CurveSpec, ...]:
    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    section_order: list[str] = []
    current_key: str | None = None
    inside_html_comment = False

    for raw_line in dataset_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if inside_html_comment:
            if "-->" in line:
                inside_html_comment = False
            continue

        if not line:
            continue
        if line.startswith("<!--"):
            if "-->" not in line:
                inside_html_comment = True
            continue
        if line.startswith("#"):
            continue

        # Allow inline comments after a path or section label.
        if " #" in line:
            line = line.split(" #", 1)[0].rstrip()
        if "<!--" in line:
            line = line.split("<!--", 1)[0].rstrip()
        if not line:
            continue

        if line.endswith(":"):
            current_key = line[:-1]
            if current_key not in grouped_paths:
                section_order.append(current_key)
            continue
        if current_key is None:
            raise ValueError(f"path listed before a section header in {dataset_file}")
        grouped_paths[current_key].append(Path(line) / "metrics.jsonl")

    ordered_keys = [key for key in DEFAULT_ORDER if key in grouped_paths]
    ordered_keys.extend(key for key in section_order if key not in ordered_keys)

    curves: list[CurveSpec] = []
    fallback_styles = [
        ("#1f77b4", ":"),
        ("#d95f02", ":"),
        ("#1b9e77", ":"),
        ("#b22222", ":"),
        ("#6a3d9a", ":"),
        ("#4d4d4d", ":"),
    ]

    for idx, key in enumerate(ordered_keys):
        if not grouped_paths[key]:
            continue
        if key in STYLE_BY_KEY:
            label, color, linestyle = STYLE_BY_KEY[key]
        else:
            color, linestyle = fallback_styles[idx % len(fallback_styles)]
            label = key
        curves.append(
            CurveSpec(
                key=key,
                label=label,
                metrics_paths=tuple(grouped_paths[key]),
                color=color,
                linestyle=linestyle,
            )
        )

    if not curves:
        raise ValueError(f"no dataset entries found in {dataset_file}")

    return tuple(curves)


def load_average_curve(
    spec: CurveSpec,
    metric_key: str,
    y_floor: float,
    max_iteration: int,
) -> tuple[list[int], list[float], list[float]]:
    values_by_epoch: dict[int, list[float]] = defaultdict(list)

    for rel_path in spec.metrics_paths:
        metrics_file = WORKSPACE_DIR / rel_path
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
                value = float(record[metric_key])
                if metric_key == "consensus_error":
                    value = math.sqrt(max(value, 0.0))
                values_by_epoch[epoch].append(value)

    if not values_by_epoch:
        raise ValueError(f"no {metric_key} data found for {spec.label}")

    epochs = sorted(values_by_epoch)
    values: list[float] = []
    std_values: list[float] = []
    for epoch in epochs:
        epoch_values = values_by_epoch[epoch]
        mean_value = sum(epoch_values) / len(epoch_values)
        variance = sum((value - mean_value) ** 2 for value in epoch_values) / len(epoch_values)
        values.append(mean_value)
        std_values.append(math.sqrt(variance))

    first_floor_index = next((idx for idx, value in enumerate(values) if value <= y_floor), None)
    if first_floor_index is not None:
        epochs = epochs[: first_floor_index + 1]
        values = values[: first_floor_index + 1]
        std_values = std_values[: first_floor_index + 1]
        values[-1] = y_floor
        std_values[-1] = 0.0

    return epochs, values, std_values


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#303030")
        spine.set_linewidth(0.9)
    ax.tick_params(axis="both", which="major", direction="out", length=4.2, width=0.9, color="#303030")


def draw_metric_panel(
    ax: plt.Axes,
    curves: tuple[CurveSpec, ...],
    metric_key: str,
    y_label: str,
    y_floor: float,
    max_iteration: int,
) -> None:
    plotted_curves: list[tuple[list[int], list[float]]] = []

    for spec in curves:
        epochs, values, std_values = load_average_curve(
            spec,
            metric_key=metric_key,
            y_floor=y_floor,
            max_iteration=max_iteration,
        )
        plotted_curves.append((epochs, values))
        lower = [max(y_floor, mean_value - std_value) for mean_value, std_value in zip(values, std_values)]
        upper = [max(y_floor, mean_value + std_value) for mean_value, std_value in zip(values, std_values)]
        ax.fill_between(
            epochs,
            lower,
            upper,
            color=spec.color,
            alpha=0.16,
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(
            epochs,
            values,
            label=spec.label,
            color=spec.color,
            linestyle=spec.linestyle,
            linewidth=2.4,
            dash_capstyle="round",
            zorder=2,
        )

    max_value = max(max(values) for _, values in plotted_curves)
    ax.set_xlim(0, max_iteration)
    ax.set_ylim(y_floor, max_value * 1.15)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(FixedLocator(select_sparse_log_ticks(y_floor, max_value * 1.15)))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.minorticks_off()
    style_axis(ax)


def add_top_legend(fig: plt.Figure, handles, labels, *, y_anchor: float = 0.955) -> None:
    fig.legend(
        handles,
        labels,
        title="block partition",
        loc="upper center",
        bbox_to_anchor=(0.5, y_anchor),
        ncol=4,
        frameon=False,
        handlelength=2.0,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.0,
    )


def legend_entries_for_curves(
    curves: tuple[CurveSpec, ...],
    handles,
    *,
    exclude_keys: set[str] | None = None,
):
    exclude_keys = exclude_keys or set()
    filtered_handles = []
    filtered_labels = []
    for spec, handle in zip(curves, handles):
        if spec.key in exclude_keys:
            continue
        filtered_handles.append(handle)
        filtered_labels.append(spec.label)
    return filtered_handles, filtered_labels


def make_single_metric_plot(
    *,
    curves: tuple[CurveSpec, ...],
    metric_key: str,
    y_label: str,
    y_floor: float,
    max_iteration: int,
    output_stem: str,
) -> tuple[Path, Path]:
    configure_style()

    fig, ax = plt.subplots(figsize=(6.2, 5.1), facecolor="white")
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.15, top=0.79)
    draw_metric_panel(
        ax,
        curves=curves,
        metric_key=metric_key,
        y_label=y_label,
        y_floor=y_floor,
        max_iteration=max_iteration,
    )
    handles, labels = ax.get_legend_handles_labels()
    if metric_key == "consensus_error":
        handles, labels = legend_entries_for_curves(curves, handles, exclude_keys={"1d1"})
    add_top_legend(fig, handles, labels)

    png_path = PLOT_DIR / f"{output_stem}.png"
    pdf_path = PLOT_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def make_plot(args: argparse.Namespace) -> tuple[Path, Path]:
    configure_style()
    curves = parse_dataset_file(args.dataset)

    for curve in curves:
        print(f"{curve.label}: averaging {len(curve.metrics_paths)} runs from {args.dataset.name}")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4), facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.79, wspace=0.24)

    draw_metric_panel(
        axes[0],
        curves=curves,
        metric_key="residual_norm",
        y_label=r"residual norm $\|Ax-b\|$",
        y_floor=args.residual_floor,
        max_iteration=args.residual_max_iteration,
    )
    draw_metric_panel(
        axes[1],
        curves=curves,
        metric_key="consensus_error",
        y_label="consensus error",
        y_floor=args.consensus_floor,
        max_iteration=args.consensus_max_iteration,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    add_top_legend(fig, handles, labels)

    png_path = PLOT_DIR / f"{args.output_stem}.png"
    pdf_path = PLOT_DIR / f"{args.output_stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    png_path, pdf_path = make_plot(args)
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")

    residual_png_path, residual_pdf_path = make_single_metric_plot(
        curves=parse_dataset_file(args.dataset),
        metric_key="residual_norm",
        y_label=r"residual norm $\|Ax-b\|$",
        y_floor=args.residual_floor,
        max_iteration=args.residual_max_iteration,
        output_stem=f"{args.output_stem}_residual",
    )
    print(f"saved {residual_png_path}")
    print(f"saved {residual_pdf_path}")

    consensus_png_path, consensus_pdf_path = make_single_metric_plot(
        curves=parse_dataset_file(args.dataset),
        metric_key="consensus_error",
        y_label="consensus error",
        y_floor=args.consensus_floor,
        max_iteration=args.consensus_max_iteration,
        output_stem=f"{args.output_stem}_consensus",
    )
    print(f"saved {consensus_png_path}")
    print(f"saved {consensus_pdf_path}")


if __name__ == "__main__":
    main()
