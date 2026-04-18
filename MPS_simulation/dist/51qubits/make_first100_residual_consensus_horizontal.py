#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from importlib.machinery import SourceFileLoader

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_JSON = (
    THIS_DIR
    / "run"
    / "j0p1_two_layer_ry_cz_true_scales_51q_pi6_lr0p02_iter300_seed1234.json"
)
DEFAULT_OUTDIR = THIS_DIR / "run" / "plots"


plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.1,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.labelsize": 11,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a horizontal residual/consensus figure averaged across one or more 51-qubit first-100 benchmarks."
    )
    parser.add_argument("--json", type=Path, nargs="+", default=[DEFAULT_JSON])
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--max-iteration", type=int, default=100)
    return parser.parse_args()


def load_logged_history(path: Path, max_iteration: int) -> tuple[list[int], np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    logged = [
        item
        for item in data["history"]
        if int(item["iteration"]) <= max_iteration and "global_residual" in item
    ]
    if not logged:
        raise ValueError("No logged residual / consensus entries found in the selected history window.")

    iterations = [int(item["iteration"]) for item in logged]
    residuals = np.asarray([float(item["global_residual"]) for item in logged], dtype=np.float64)
    consensus_variance = np.asarray(
        [float(item["parameter_consensus_variance"]) for item in logged],
        dtype=np.float64,
    )
    consensus = np.sqrt(np.maximum(consensus_variance, 0.0))
    cutoff_idx = None
    for idx, value in enumerate(residuals):
        if value <= 1.0e-2:
            cutoff_idx = idx
            break
    if cutoff_idx is not None:
        keep = cutoff_idx + 1
        iterations = iterations[:keep]
        residuals = residuals[:keep]
        consensus = consensus[:keep]
    return iterations, residuals, consensus


def load_benchmark_module():
    benchmark_path = THIS_DIR / "quimb_dist_eq26_j0p1_two_layer_ry_cz_true_scales_51q.py"
    return SourceFileLoader("benchmark51q", str(benchmark_path)).load_module()


def load_initial_metrics(path: Path, mod) -> tuple[float, float]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    cfg = mod.Config(**data["config"])
    problem_np = mod.build_direct_problem(cfg)
    alpha_init, beta_init = mod.make_initial_parameters(cfg)
    metrics = mod.parameter_consensus_metrics(alpha_init)
    diag = mod.build_final_diagnostics(alpha_init, beta_init, cfg, problem_np)
    return float(diag["global_residual"]), math.sqrt(max(float(metrics["parameter_consensus_variance"]), 0.0))


def average_logged_histories(
    paths: list[Path],
    max_iteration: int,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    histories: list[dict[int, tuple[float, float]]] = []
    common_iterations: set[int] | None = None

    for path in paths:
        iterations, residuals, consensus = load_logged_history(path, max_iteration)
        mapping = {
            int(iteration): (float(residual), float(consensus))
            for iteration, residual, consensus in zip(iterations, residuals, consensus, strict=True)
        }
        histories.append(mapping)
        current_iters = set(mapping.keys())
        common_iterations = current_iters if common_iterations is None else (common_iterations & current_iters)

    if not common_iterations:
        raise ValueError("No common logged iterations were found across the selected run JSON files.")

    ordered_iterations = sorted(common_iterations)
    residual_avg = np.asarray(
        [np.mean([history[it][0] for history in histories]) for it in ordered_iterations],
        dtype=np.float64,
    )
    residual_std = np.asarray(
        [np.std([history[it][0] for history in histories]) for it in ordered_iterations],
        dtype=np.float64,
    )
    consensus_avg = np.asarray(
        [np.mean([history[it][1] for history in histories]) for it in ordered_iterations],
        dtype=np.float64,
    )
    consensus_std = np.asarray(
        [np.std([history[it][1] for history in histories]) for it in ordered_iterations],
        dtype=np.float64,
    )
    return ordered_iterations, residual_avg, residual_std, consensus_avg, consensus_std


def style_axes(ax: plt.Axes, ylabel: str, ymin: float | None = None) -> None:
    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.1)
        ax.spines[side].set_color("#444444")
    ax.tick_params(axis="both", which="major", width=1.1, length=4.5, color="#444444")
    if ymin is not None:
        ymax = ax.get_ylim()[1]
        ax.set_ylim(bottom=ymin, top=max(ymax, ymin * 10.0))


def add_series(ax: plt.Axes, x: list[int], y: np.ndarray, y_std: np.ndarray) -> None:
    lower = np.maximum(y - y_std, np.finfo(np.float64).tiny)
    upper = np.maximum(y + y_std, np.finfo(np.float64).tiny)
    ax.fill_between(
        x,
        lower,
        upper,
        color="red",
        alpha=0.18,
        linewidth=0.0,
    )
    ax.plot(
        x,
        y,
        linestyle=":",
        linewidth=2.0,
        color="red",
    )


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    json_paths = [path.resolve() for path in args.json]
    mod = load_benchmark_module()

    iterations, residuals, residual_stds, consensus, consensus_stds = average_logged_histories(
        json_paths,
        args.max_iteration,
    )
    initial_pairs = [load_initial_metrics(path, mod) for path in json_paths]
    initial_residual = float(np.mean([pair[0] for pair in initial_pairs]))
    initial_residual_std = float(np.std([pair[0] for pair in initial_pairs]))
    initial_consensus = float(np.mean([pair[1] for pair in initial_pairs]))
    initial_consensus_std = float(np.std([pair[1] for pair in initial_pairs]))

    iterations_with_init = [0] + iterations
    residuals_with_init = np.concatenate(
        [np.asarray([max(initial_residual, np.finfo(np.float64).tiny)]), np.maximum(residuals, np.finfo(np.float64).tiny)]
    )
    residual_stds_with_init = np.concatenate(
        [np.asarray([initial_residual_std]), residual_stds]
    )
    consensus_with_init = np.concatenate(
        [np.asarray([max(initial_consensus, np.finfo(np.float64).tiny)]), np.maximum(consensus, np.finfo(np.float64).tiny)]
    )
    consensus_stds_with_init = np.concatenate(
        [np.asarray([initial_consensus_std]), consensus_stds]
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), dpi=220)
    fig.patch.set_facecolor("white")

    add_series(axes[0], iterations_with_init, residuals_with_init, residual_stds_with_init)
    axes[0].set_xlim(0, max(iterations))
    style_axes(axes[0], ylabel=r"residual norm $\|Ax-b\|$", ymin=1.0e-2)

    add_series(axes[1], iterations_with_init, consensus_with_init, consensus_stds_with_init)
    axes[1].set_xlim(0, max(iterations))
    style_axes(axes[1], ylabel="parameter consensus error")

    fig.tight_layout(w_pad=1.4)

    png_path = args.outdir / "residual_consensus_first100_horizontal.png"
    pdf_path = args.outdir / "residual_consensus_first100_horizontal.pdf"
    fig.savefig(png_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    single_specs = [
        (
            residuals_with_init,
            r"residual norm $\|Ax-b\|$",
            1.0e-2,
            "residual_first100_n51",
        ),
        (
            consensus_with_init,
            "parameter consensus error",
            None,
            "consensus_first100_n51",
        ),
    ]
    for values, ylabel, ymin, stem in single_specs:
        fig_single, ax_single = plt.subplots(1, 1, figsize=(4.6, 3.8), dpi=220)
        fig_single.patch.set_facecolor("white")
        std_values = residual_stds_with_init if stem.startswith("residual") else consensus_stds_with_init
        add_series(ax_single, iterations_with_init, values, std_values)
        ax_single.set_xlim(0, max(iterations))
        style_axes(ax_single, ylabel=ylabel, ymin=ymin)
        fig_single.tight_layout()
        single_png = args.outdir / f"{stem}.png"
        single_pdf = args.outdir / f"{stem}.pdf"
        fig_single.savefig(single_png, bbox_inches="tight", facecolor=fig_single.get_facecolor())
        fig_single.savefig(single_pdf, bbox_inches="tight", facecolor=fig_single.get_facecolor())
        plt.close(fig_single)

    print(f"Averaged {len(json_paths)} runs.")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
