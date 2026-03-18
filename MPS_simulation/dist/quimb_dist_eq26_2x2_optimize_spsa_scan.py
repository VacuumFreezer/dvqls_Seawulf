#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quimb_dist_eq26_2x2_optimize import resolve_output_paths, write_report
from quimb_dist_eq26_2x2_optimize_spsa import Config, optimize, plot_history


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = THIS_DIR / "5qubits" / "spsa_scan"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan SPSA learning rate and perturbation c for the 5-qubit local "
            "distributed Eq. (26) benchmark."
        )
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[0.002, 0.005, 0.01],
    )
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.05],
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--spsa-seed", type=int, default=1234)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
    )
    return parser.parse_args()


def build_base_config(args: argparse.Namespace) -> Config:
    return Config(
        global_qubits=6,
        local_qubits=5,
        j_coupling=0.1,
        kappa=20.0,
        row_self_loop_weight=1.0,
        layers=4,
        gate_max_bond=32,
        gate_cutoff=1.0e-10,
        apply_max_bond=64,
        apply_cutoff=1.0e-10,
        learning_rate=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1.0e-8,
        iterations=int(args.iterations),
        report_every=int(args.report_every),
        init_mode="structured_linspace",
        init_seed=1234,
        init_start=0.01,
        init_stop=0.2,
        x_scale_init=0.75,
        z_scale_init=0.10,
        spsa_seed=int(args.spsa_seed),
        spsa_c=0.05,
        spsa_directions=1,
        spsa_full_params=False,
        out_json=None,
        out_figure=None,
        out_report=None,
    )


def combo_slug(lr: float, c: float, iterations: int) -> str:
    lr_str = format(lr, ".6g").replace(".", "p")
    c_str = format(c, ".6g").replace(".", "p")
    return f"spsa_lr{lr_str}_c{c_str}_iter{iterations}"


def write_scan_report(
    report_path: Path,
    learning_rates: list[float],
    c_values: list[float],
    scan_results: list[dict[str, object]],
    best: dict[str, object],
    heatmap_path: Path,
) -> None:
    lines = [
        "# SPSA Parameter Scan",
        "",
        "## Setup",
        f"- Learning rates: `{learning_rates}`",
        f"- SPSA c values: `{c_values}`",
        f"- Iterations per run: `{best['optimization']['iterations']}`",
        f"- Shared SPSA seed: `{best['optimization']['spsa_seed']}`",
        "",
        "## Best By Final Residual",
        f"- Learning rate: `{best['config']['learning_rate']}`",
        f"- SPSA c: `{best['optimization']['spsa_c']}`",
        f"- Final residual: `{best['final_metrics']['residual_norm_l2']:.12g}`",
        f"- Final global cost: `{best['final_metrics']['global_cost']:.12g}`",
        f"- Final relative solution error: `{best['final_metrics']['relative_solution_error_l2']:.12g}`",
        f"- Run report: `{best['artifacts']['report']}`",
        "",
        "## Ranking",
    ]

    for idx, item in enumerate(scan_results, start=1):
        final = item["final_metrics"]
        lines.append(
            f"{idx}. `lr={item['config']['learning_rate']}`, `c={item['optimization']['spsa_c']}`: "
            f"residual=`{final['residual_norm_l2']:.12g}`, "
            f"cost=`{final['global_cost']:.12g}`, "
            f"rel_err=`{final['relative_solution_error_l2']:.12g}`"
        )

    lines.extend(
        [
            "",
            "## Heatmap",
            f"- Residual heatmap: `{heatmap_path}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_heatmap(
    heatmap_path: Path,
    learning_rates: list[float],
    c_values: list[float],
    scan_results: list[dict[str, object]],
) -> None:
    residual_map = np.full((len(learning_rates), len(c_values)), np.nan, dtype=np.float64)
    for item in scan_results:
        i = learning_rates.index(float(item["config"]["learning_rate"]))
        j = c_values.index(float(item["optimization"]["spsa_c"]))
        residual_map[i, j] = float(item["final_metrics"]["residual_norm_l2"])

    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=160)
    im = ax.imshow(np.log10(np.maximum(residual_map, 1.0e-16)), cmap="viridis")
    ax.set_xticks(range(len(c_values)), [str(c) for c in c_values])
    ax.set_yticks(range(len(learning_rates)), [str(lr) for lr in learning_rates])
    ax.set_xlabel("SPSA c")
    ax.set_ylabel("Learning Rate")
    ax.set_title("log10 Final Residual")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10 residual")

    for i in range(len(learning_rates)):
        for j in range(len(c_values)):
            ax.text(j, i, f"{residual_map[i, j]:.3g}", ha="center", va="center", color="white")

    fig.tight_layout()
    fig.savefig(heatmap_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    learning_rates = [float(x) for x in args.learning_rates]
    c_values = [float(x) for x in args.c_values]
    base_cfg = build_base_config(args)

    scan_results: list[dict[str, object]] = []
    for lr in learning_rates:
        for c in c_values:
            slug = combo_slug(lr, c, int(args.iterations))
            run_cfg = replace(
                base_cfg,
                learning_rate=float(lr),
                spsa_c=float(c),
                out_json=str(out_dir / f"{slug}.json"),
                out_figure=str(out_dir / f"{slug}_history.png"),
                out_report=str(out_dir / f"{slug}_report.md"),
            )

            print(f"\n=== Running SPSA scan combo: lr={lr}, c={c} ===")
            artifact_paths = resolve_output_paths(run_cfg)
            result = optimize(run_cfg)
            result["artifacts"] = {key: str(path.resolve()) for key, path in artifact_paths.items()}

            artifact_paths["json"].write_text(json.dumps(result, indent=2), encoding="utf-8")
            plot_history(result["history"], artifact_paths["figure"])
            write_report(artifact_paths["report"], result)
            scan_results.append(
                {
                    "config": {
                        "learning_rate": float(lr),
                    },
                    "optimization": result["optimization"],
                    "final_metrics": result["history"][-1],
                    "artifacts": result["artifacts"],
                }
            )

    scan_results.sort(key=lambda item: item["final_metrics"]["residual_norm_l2"])
    best = scan_results[0]

    summary_json_path = out_dir / "spsa_scan_summary.json"
    summary_report_path = out_dir / "spsa_scan_summary.md"
    heatmap_path = out_dir / "spsa_scan_residual_heatmap.png"

    write_heatmap(heatmap_path, learning_rates, c_values, scan_results)
    write_scan_report(summary_report_path, learning_rates, c_values, scan_results, best, heatmap_path)
    summary_json_path.write_text(
        json.dumps(
            {
                "learning_rates": learning_rates,
                "c_values": c_values,
                "ranking": scan_results,
                "best": best,
                "artifacts": {
                    "report": str(summary_report_path.resolve()),
                    "heatmap": str(heatmap_path.resolve()),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== Scan complete ===")
    print(json.dumps({"best": best, "summary_report": str(summary_report_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
