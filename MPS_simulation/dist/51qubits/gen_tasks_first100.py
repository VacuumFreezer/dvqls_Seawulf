#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
BENCHMARK_SCRIPT = THIS_DIR / "quimb_dist_eq26_j0p1_two_layer_ry_cz_true_scales_51q.py"
EXISTING_JSON = THIS_DIR / "run" / "j0p1_two_layer_ry_cz_true_scales_51q_pi6_lr0p02_iter300_seed1234.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the 9 additional 51-qubit first-100 benchmark tasks and averaging-plot inputs."
    )
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--existing-json", type=Path, default=EXISTING_JSON)
    parser.add_argument("--seeds", type=str, default="1235,1236,1237,1238,1239,1240,1241,1242,1243")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-iteration", type=int, default=100)
    return parser.parse_args()


def parse_seed_text(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("At least one seed must be supplied.")
    return seeds


def main() -> None:
    args = parse_args()
    existing_json = args.existing_json.resolve()
    if not existing_json.exists():
        raise FileNotFoundError(f"Existing reference run not found: {existing_json}")

    config = json.loads(existing_json.read_text(encoding="utf-8"))["config"]
    seeds = parse_seed_text(args.seeds)
    tag = args.tag.strip() or datetime.now().strftime("%Y%m%d_51q_first100_%H%M%S")

    control_dir = THIS_DIR / "run" / tag
    runs_dir = control_dir / "benchmarks"
    plots_dir = control_dir / "plots"
    slurm_dir = THIS_DIR / "slurm"
    control_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)

    tasks_path = control_dir / "tasks.txt"
    manifest_path = control_dir / "task_manifest.json"
    plot_inputs_path = control_dir / "plot_jsons.txt"

    task_lines: list[str] = []
    manifest: list[dict[str, object]] = []
    plot_jsons: list[str] = [str(existing_json)]

    sigma_init = float(config["sigma_init"])
    angle_init_radius = float(config["angle_init_radius"])
    learning_rate = float(config["learning_rate"])
    j_coupling = float(config["j_coupling"])
    global_qubits = int(config["global_qubits"])
    local_qubits = int(config["local_qubits"])
    kappa = float(config["kappa"])
    row_self_loop_weight = float(config["row_self_loop_weight"])
    report_every = int(config["report_every"])

    for seed in seeds:
        base_name = f"j0p1_two_layer_ry_cz_true_scales_51q_pi6_lr0p02_iter{args.iterations}_seed{seed}"
        out_json = (runs_dir / f"{base_name}.json").resolve()
        out_figure = (runs_dir / f"{base_name}_cost.png").resolve()
        out_report = (runs_dir / f"{base_name}_report.md").resolve()
        command = " ".join(
            [
                "python",
                str(BENCHMARK_SCRIPT.resolve()),
                "--global-qubits",
                str(global_qubits),
                "--local-qubits",
                str(local_qubits),
                "--j-coupling",
                str(j_coupling),
                "--kappa",
                str(kappa),
                "--row-self-loop-weight",
                str(row_self_loop_weight),
                "--learning-rate",
                str(learning_rate),
                "--iterations",
                str(args.iterations),
                "--report-every",
                str(report_every),
                "--init-seed",
                str(seed),
                "--sigma-init",
                str(sigma_init),
                "--angle-init-radius",
                str(angle_init_radius),
                "--out-json",
                str(out_json),
                "--out-figure",
                str(out_figure),
                "--out-report",
                str(out_report),
            ]
        )
        task_lines.append(command)
        manifest.append(
            {
                "seed": seed,
                "iterations": args.iterations,
                "out_json": str(out_json),
                "out_figure": str(out_figure),
                "out_report": str(out_report),
                "command": command,
            }
        )
        plot_jsons.append(str(out_json))

    tasks_path.write_text("\n".join(task_lines) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    plot_inputs_path.write_text("\n".join(plot_jsons) + "\n", encoding="utf-8")

    print(f"[OK] tasks written: {tasks_path}")
    print(f"[OK] manifest written: {manifest_path}")
    print(f"[OK] plot-input list written: {plot_inputs_path}")
    print(f"[OK] total tasks: {len(task_lines)}")
    print(f"[OK] control root: {control_dir}")
    print(f"[OK] plot outdir: {plots_dir}")
    print("[OK] array submit command:")
    print(
        "  sbatch --array=0-"
        f"{len(task_lines) - 1} --export=TAG={tag} "
        "MPS_simulation/dist/51qubits/submit_array_first100.slurm"
    )
    print("[OK] dependent plot submit command:")
    print(
        "  sbatch --dependency=afterok:<array_jobid> --export=TAG="
        f"{tag} MPS_simulation/dist/51qubits/submit_average_plot.slurm"
    )


if __name__ == "__main__":
    main()
