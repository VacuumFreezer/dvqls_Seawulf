#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
        description="Create framed first-100-iteration plots for the 51-qubit distributed benchmark."
    )
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--max-iteration", type=int, default=100)
    return parser.parse_args()


def load_history(
    path: Path,
    max_iteration: int,
) -> tuple[list[int], np.ndarray, list[int], np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    history = [item for item in data["history"] if int(item["iteration"]) <= max_iteration]
    if not history:
        raise ValueError(f"No history entries found with iteration <= {max_iteration}.")

    logged = [item for item in history if "global_residual" in item]
    if not logged:
        raise ValueError("No logged residual / consensus entries found in the selected history window.")

    iterations = [int(item["iteration"]) for item in history]
    logged_iterations = [int(item["iteration"]) for item in logged]
    costs = np.asarray([float(item["global_cost"]) for item in history], dtype=np.float64)
    residuals = np.asarray([float(item["global_residual"]) for item in logged], dtype=np.float64)
    consensus = np.asarray(
        [float(item["parameter_consensus_variance"]) for item in logged],
        dtype=np.float64,
    )
    return iterations, costs, logged_iterations, residuals, consensus


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


def save_plot(
    outpath: Path,
    iterations: list[int],
    values: np.ndarray,
    ylabel: str,
    color: str,
    ymin: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=220)
    fig.patch.set_facecolor("white")
    clipped = np.maximum(values, np.finfo(np.float64).tiny)
    ax.plot(iterations, clipped, color=color, linewidth=1.8)
    ax.set_xlim(min(iterations), max(iterations))
    style_axes(ax, ylabel=ylabel, ymin=ymin)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    iterations, costs, logged_iterations, residuals, consensus = load_history(args.json, args.max_iteration)

    save_plot(
        args.outdir / "global_cost_first100_framed.png",
        iterations,
        costs,
        ylabel="global cost",
        color="#1f77b4",
        ymin=1.0e-4,
    )
    save_plot(
        args.outdir / "residual_norm_first100_framed.png",
        logged_iterations,
        residuals,
        ylabel=r"residual norm $\|Ax-b\|$",
        color="#d95f02",
        ymin=1.0e-2,
    )
    save_plot(
        args.outdir / "consensus_error_first100_framed.png",
        logged_iterations,
        consensus,
        ylabel="parameter consensus error",
        color="#2ca25f",
    )

    print(f"Wrote plots to {args.outdir}")


if __name__ == "__main__":
    main()
