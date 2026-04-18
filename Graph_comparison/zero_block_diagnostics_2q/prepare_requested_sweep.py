#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_SEEDS = [1342, 2836, 5539]
PROBLEM_KEYS = ("ising_zero", "ising_fill050", "ising_fill100", "eq1_full_k196")
ROW_GRAPHS = ("P4", "C4", "K4")
COLUMN_GRAPH = "K4"
MEMORY = "16G"
TIME_LIMIT = "24:00:00"


def _lr_tag(value: float) -> str:
    text = f"{value:.0e}".replace("+0", "").replace("+", "")
    return text.replace("-0", "-")


def build_task_command(
    *,
    python_bin: str,
    out_dir: Path,
    seed: int,
    system_key: str,
    row_graph: str,
    column_graph: str,
    epochs: int,
    lr: float,
    decay: float,
    log_every: int,
    layers: int,
) -> str:
    cmd = [
        python_bin,
        "Graph_comparison/zero_block_diagnostics_2q/seawulf_row_sparsity_diag_qjit.py",
        "--static_ops",
        "Graph_comparison/zero_block_diagnostics_2q/static_ops_block_sparsity_4x4_2q.py",
        "--system_key",
        system_key,
        "--row_graph",
        row_graph,
        "--column_graph",
        column_graph,
        "--epochs",
        str(int(epochs)),
        "--seed",
        str(int(seed)),
        "--lr",
        str(float(lr)),
        "--decay",
        str(float(decay)),
        "--log_every",
        str(int(log_every)),
        "--layers",
        str(int(layers)),
        "--out",
        out_dir.as_posix(),
    ]
    return " ".join(shlex.quote(part) for part in cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--python-bin", default="/gpfs/home/tonshen/.conda/envs/pennylane/bin/python")
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--decay", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=40)
    ap.add_argument("--layers", type=int, default=2)
    args = ap.parse_args()

    seeds = [int(seed) for seed in args.seeds]

    run_root = ROOT / "Graph_comparison" / "zero_block_diagnostics_2q" / "run"
    timestamp = datetime.now().strftime("%Y%m%d")
    seed_suffix = f"s{len(seeds)}"
    decay_suffix = "fixlr" if abs(float(args.decay) - 1.0) < 1.0e-15 else f"decay{args.decay:g}"
    suite_tag = args.tag.strip() or (
        f"{timestamp}_row_zero_block_diag_2q_e{int(args.epochs)}"
        f"_lr{_lr_tag(float(args.lr))}_log{int(args.log_every)}_{seed_suffix}_{decay_suffix}_col{COLUMN_GRAPH}"
    )
    suite_dir = run_root / suite_tag
    slurm_dir = suite_dir / "slurm"
    suite_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "suite_tag": suite_tag,
        "created_at": datetime.now().isoformat(),
        "python_bin": args.python_bin,
        "seeds": seeds,
        "fixed_column_graph": COLUMN_GRAPH,
        "hyperparameters": {
            "epochs": int(args.epochs),
            "log_every": int(args.log_every),
            "lr": float(args.lr),
            "decay": float(args.decay),
            "layers": int(args.layers),
            "ansatz": "paper_fig3_ry_cz: RY -> CZ(even bonds) -> RY -> CZ(odd bonds)",
            "workflow": "2-qubit row-sparsity diagnostics with fixed column graph and row-disagreement logging",
        },
        "experiments": [],
    }

    submit_lines = ["#!/bin/bash", "set -euo pipefail", ""]
    task_index = 0
    for system_key in PROBLEM_KEYS:
        for row_graph in ROW_GRAPHS:
            case_name = f"problem={system_key}__row={row_graph}__col={COLUMN_GRAPH}"
            case_dir = suite_dir / case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            tasks_path = case_dir / "tasks.txt"
            lines: list[str] = []
            output_dirs = []
            for seed in seeds:
                out_dir = case_dir / f"seed={int(seed)}"
                out_dir.mkdir(parents=True, exist_ok=True)
                lines.append(
                    build_task_command(
                        python_bin=args.python_bin,
                        out_dir=out_dir,
                        seed=int(seed),
                        system_key=system_key,
                        row_graph=row_graph,
                        column_graph=COLUMN_GRAPH,
                        epochs=int(args.epochs),
                        lr=float(args.lr),
                        decay=float(args.decay),
                        log_every=int(args.log_every),
                        layers=int(args.layers),
                    )
                )
                output_dirs.append(out_dir.as_posix())

            tasks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            task_index += 1
            exp_id = f"E{task_index:02d}"
            manifest["experiments"].append(
                {
                    "experiment_id": exp_id,
                    "system_key": system_key,
                    "row_graph": row_graph,
                    "column_graph": COLUMN_GRAPH,
                    "mem": MEMORY,
                    "time": TIME_LIMIT,
                    "tasks_file": tasks_path.as_posix(),
                    "output_dirs": output_dirs,
                }
            )
            sbatch_parts = [
                "sbatch",
                "--job-name",
                f"rowsparse_{exp_id}",
                "--array",
                f"0-{len(seeds) - 1}",
                "--mem",
                MEMORY,
                "--time",
                TIME_LIMIT,
                "--output",
                f"Graph_comparison/zero_block_diagnostics_2q/run/{suite_tag}/slurm/%x_%A_%a.out",
                "--error",
                f"Graph_comparison/zero_block_diagnostics_2q/run/{suite_tag}/slurm/%x_%A_%a.err",
                "--export",
                f"ALL,TAG={suite_tag}/{case_name}",
                "Graph_comparison/zero_block_diagnostics_2q/submit_array.slurm",
            ]
            submit_lines.append(" ".join(shlex.quote(part) for part in sbatch_parts))

    manifest_path = suite_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    submit_script_path = suite_dir / "submit_commands.sh"
    submit_script_path.write_text("\n".join(submit_lines) + "\n", encoding="utf-8")
    submit_script_path.chmod(0o755)

    print(f"[OK] suite_dir={suite_dir}")
    print(f"[OK] manifest={manifest_path}")
    print(f"[OK] submit_commands={submit_script_path}")
    print(f"[OK] seeds={seeds}")
    for exp in manifest["experiments"]:
        print(f"[OK] tasks={exp['tasks_file']}")


if __name__ == "__main__":
    main()
