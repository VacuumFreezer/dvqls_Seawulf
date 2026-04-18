#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shlex
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


JOB_MEM = "64G"
JOB_TIME = "24:00:00"
ROW_GRAPH = "P4"
COLUMN_GRAPH = "P4"
OPTIMIZATION_CASES = (
    ("track_adamX_adamZ", "tracking+AdamOnX+AdamOnZ"),
    ("track_adamX_only", "tracking+AdamOnX"),
    ("track_adamZ_only", "tracking+AdamOnZ"),
    ("gradshare_adamX_adamZ", "gradient-sharing+AdamOnX+AdamOnZ"),
    ("consensus_adamX_adamZ", "consensus+AdamOnX+AdamOnZ"),
)
OPTIMIZATION_CASE_LABELS = dict(OPTIMIZATION_CASES)


def _lr_tag(value: float) -> str:
    text = f"{value:.0e}".replace("+0", "").replace("+", "")
    return text.replace("-0", "-")


def _generate_random_seeds(count: int, *, max_exclusive: int = 10000) -> list[int]:
    if int(count) < 1:
        raise ValueError("seed count must be at least 1")
    if int(max_exclusive) < int(count):
        raise ValueError("max_exclusive must be >= seed count")
    rng = random.SystemRandom()
    return sorted(rng.sample(range(int(max_exclusive)), int(count)))


def build_task_command(
    *,
    python_bin: str,
    out_dir: Path,
    seed: int,
    optimization_case: str,
    epochs: int,
    lr: float,
    decay: float,
    log_every: int,
    layers: int,
) -> str:
    cmd = [
        python_bin,
        "Necessarity_comparison/seawulf_necessarity_ising_qjit.py",
        "--static_ops",
        "Necessarity_comparison/static_ops_ising_4x4_5q_cond200.py",
        "--topology",
        ROW_GRAPH,
        "--optimization_case",
        optimization_case,
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
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--seed-count", type=int, default=3)
    ap.add_argument("--python-bin", default="/gpfs/home/tonshen/.conda/envs/pennylane/bin/python")
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--decay", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=40)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--cases", nargs="*", default=None)
    args = ap.parse_args()

    seeds = [int(seed) for seed in args.seeds] if args.seeds else _generate_random_seeds(int(args.seed_count))
    if args.cases:
        requested_case_ids = [str(case_id).strip() for case_id in args.cases if str(case_id).strip()]
        unknown_case_ids = [case_id for case_id in requested_case_ids if case_id not in OPTIMIZATION_CASE_LABELS]
        if unknown_case_ids:
            valid = ", ".join(case_id for case_id, _ in OPTIMIZATION_CASES)
            raise ValueError(f"Unknown case ids: {unknown_case_ids}. Expected subset of: {valid}")
        selected_cases = tuple((case_id, OPTIMIZATION_CASE_LABELS[case_id]) for case_id in requested_case_ids)
    else:
        selected_cases = OPTIMIZATION_CASES

    run_root = ROOT / "Necessarity_comparison" / "run"
    timestamp = datetime.now().strftime("%Y%m%d")
    seed_suffix = f"s{len(seeds)}"
    decay_suffix = "fixlr" if abs(float(args.decay) - 1.0) < 1.0e-15 else f"decay{args.decay:g}"
    suite_tag = args.tag.strip() or (
        f"{timestamp}_necessarity_ising5q_4x4_path_cond200_e{int(args.epochs)}"
        f"_lr{_lr_tag(float(args.lr))}_log{int(args.log_every)}_{seed_suffix}_{decay_suffix}_fig3"
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
        "topology": {
            "row_graph": ROW_GRAPH,
            "column_graph": COLUMN_GRAPH,
        },
        "problem": {
            "static_ops": "Necessarity_comparison/static_ops_ising_4x4_5q_cond200.py",
            "spectrum_target": {
                "lambda_min": 1.0 / 200.0,
                "lambda_max": 1.0,
                "condition_number": 200.0,
            },
        },
        "hyperparameters": {
            "epochs": int(args.epochs),
            "log_every": int(args.log_every),
            "lr": float(args.lr),
            "decay": float(args.decay),
            "layers": int(args.layers),
            "ansatz": "paper_fig3_ry_cz: RY -> CZ(even bonds) -> RY -> CZ(odd bonds)",
            "workflow": "4x4 Ising necessity comparison on the 7-qubit benchmark with 5 local qubits per agent",
        },
        "optimization_cases": [
            {
                "case_id": case_id,
                "label": label,
            }
            for case_id, label in selected_cases
        ],
        "jobs": [],
    }

    submit_lines = ["#!/bin/bash", "set -euo pipefail", ""]
    tasks_path = suite_dir / "tasks.txt"
    lines: list[str] = []

    for case_id, label in selected_cases:
        case_dir = suite_dir / f"case={case_id}"
        case_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            out_dir = case_dir / f"seed={int(seed)}"
            out_dir.mkdir(parents=True, exist_ok=True)
            lines.append(
                build_task_command(
                    python_bin=args.python_bin,
                    out_dir=out_dir,
                    seed=int(seed),
                    optimization_case=case_id,
                    epochs=int(args.epochs),
                    lr=float(args.lr),
                    decay=float(args.decay),
                    log_every=int(args.log_every),
                    layers=int(args.layers),
                )
            )

    tasks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for task_index, (case_id, label) in enumerate(
        [(case_id, label) for case_id, label in selected_cases for _ in seeds]
    ):
        seed = seeds[task_index % len(seeds)]
        out_dir = suite_dir / f"case={case_id}" / f"seed={int(seed)}"
        manifest["jobs"].append(
            {
                "task_index": task_index,
                "case_id": case_id,
                "case_label": label,
                "seed": int(seed),
                "mem": JOB_MEM,
                "time": JOB_TIME,
                "tasks_file": tasks_path.as_posix(),
                "out_dir": out_dir.as_posix(),
            }
        )
        sbatch_parts = [
            "sbatch",
            "--job-name",
            f"ising5qnec_{case_id}_s{int(seed)}",
            "--mem",
            JOB_MEM,
            "--time",
            JOB_TIME,
            "--output",
            f"Necessarity_comparison/run/{suite_tag}/slurm/%x_%j.out",
            "--error",
            f"Necessarity_comparison/run/{suite_tag}/slurm/%x_%j.err",
            "--export",
            f"ALL,TASKS_FILE={tasks_path.as_posix()},TASK_INDEX={task_index}",
            "Necessarity_comparison/submit_array.slurm",
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
    print(f"[OK] tasks={tasks_path}")
    print(f"[OK] total_jobs={len(manifest['jobs'])}")


if __name__ == "__main__":
    main()
