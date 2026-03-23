#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_task_command(
    *,
    python_bin: str,
    out_dir: Path,
    extra_args: list[str],
) -> str:
    cmd = [
        python_bin,
        "Graph_comparison/seawulf_graph_comparison_qjit.py",
        "--static_ops",
        "Graph_comparison/static_ops_13q_xzx_fresh_4x4.py",
        "--benchmark",
        "B3",
        "--epochs",
        "10000",
        "--seed",
        "218",
        "--lr",
        "0.005",
        "--decay",
        "0.9999",
        "--log_every",
        "20",
        "--ansatz",
        "brickwall_ry_cz",
        "--layers",
        "4",
        "--init_mode",
        "uniform_pm_pi",
        "--init_sigma_value",
        "1.0",
        "--init_sigma_noise_std",
        "0.0",
        "--init_lambda_value",
        "1.0",
        "--init_lambda_noise_std",
        "0.0",
        *extra_args,
        "--out",
        out_dir.as_posix(),
    ]
    return " ".join(shlex.quote(part) for part in cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="20260322_B3_flat_region_diagnostics_seed218")
    ap.add_argument("--python-bin", default="/gpfs/home/tonshen/.conda/envs/pennylane/bin/python")
    ap.add_argument("--mem", default="32G")
    ap.add_argument("--time", default="12:00:00")
    ap.add_argument("--freeze-from-epoch", type=int, default=4500)
    ap.add_argument("--freeze-until-epoch", type=int, default=6800)
    args = ap.parse_args()

    suite_dir = ROOT / "Graph_comparison" / "run" / args.tag
    slurm_dir = suite_dir / "slurm"
    suite_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = [
        {
            "slug": "baseline_full_diagnostics",
            "job_name": "graphcmp_B3diag_base",
            "extra_args": [],
            "description": "Replay the original B3 seed-218 run with enhanced gradient/tracker/update/parameter-change logging.",
        },
        {
            "slug": "freeze_alpha_sigma_window",
            "job_name": "graphcmp_B3diag_as",
            "extra_args": [
                "--freeze_groups",
                "alpha,sigma",
                "--freeze_from_epoch",
                str(int(args.freeze_from_epoch)),
                "--freeze_until_epoch",
                str(int(args.freeze_until_epoch)),
            ],
            "description": "Freeze alpha/sigma during the observed flat window to test whether beta/lambda alone can reduce the loss.",
        },
        {
            "slug": "freeze_beta_lambda_window",
            "job_name": "graphcmp_B3diag_bl",
            "extra_args": [
                "--freeze_groups",
                "beta,lambda",
                "--freeze_from_epoch",
                str(int(args.freeze_from_epoch)),
                "--freeze_until_epoch",
                str(int(args.freeze_until_epoch)),
            ],
            "description": "Freeze beta/lambda during the observed flat window to test whether alpha/sigma alone can reduce the loss.",
        },
    ]

    manifest = {
        "suite_tag": args.tag,
        "created_at": datetime.now().astimezone().isoformat(),
        "python_bin": args.python_bin,
        "benchmark": "B3",
        "seed": 218,
        "reference_run": "/gpfs/home/tonshen/Seawulf_simulation/Graph_comparison/run/20260321_graph_compare_4x4_e10000_lr5e3_log20_seed0_fixlr/benchmark=B3_Row-Neck/seed=218",
        "frozen_window": {
            "from_epoch": int(args.freeze_from_epoch),
            "until_epoch": int(args.freeze_until_epoch),
        },
        "jobs": [],
    }

    submit_lines = ["#!/bin/bash", "set -euo pipefail", ""]
    for diag in diagnostics:
        diag_dir = suite_dir / diag["slug"]
        diag_dir.mkdir(parents=True, exist_ok=True)
        out_dir = diag_dir / "seed=218"
        out_dir.mkdir(parents=True, exist_ok=True)
        tasks_path = diag_dir / "tasks.txt"
        cmd = build_task_command(
            python_bin=args.python_bin,
            out_dir=out_dir,
            extra_args=diag["extra_args"],
        )
        tasks_path.write_text(cmd + "\n", encoding="utf-8")
        manifest["jobs"].append(
            {
                "slug": diag["slug"],
                "job_name": diag["job_name"],
                "description": diag["description"],
                "tasks_file": tasks_path.as_posix(),
                "out_dir": out_dir.as_posix(),
                "extra_args": diag["extra_args"],
                "mem": args.mem,
                "time": args.time,
            }
        )

        sbatch_parts = [
            "sbatch",
            "--job-name",
            diag["job_name"],
            "--array",
            "0-0",
            "--mem",
            args.mem,
            "--time",
            args.time,
            "--output",
            f"Graph_comparison/run/{args.tag}/slurm/%x_%A_%a.out",
            "--error",
            f"Graph_comparison/run/{args.tag}/slurm/%x_%A_%a.err",
            "--export",
            f"ALL,TAG={args.tag}/{diag['slug']}",
            "Graph_comparison/submit_array.slurm",
        ]
        submit_lines.append(" ".join(shlex.quote(part) for part in sbatch_parts))

    (suite_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    submit_path = suite_dir / "submit_commands.sh"
    submit_path.write_text("\n".join(submit_lines) + "\n", encoding="utf-8")
    submit_path.chmod(0o755)

    print(f"[OK] suite_dir={suite_dir}")
    print(f"[OK] submit_commands={submit_path}")
    print(f"[OK] manifest={suite_dir / 'manifest.json'}")
    for diag in diagnostics:
        print(f"[OK] tasks={suite_dir / diag['slug'] / 'tasks.txt'}")


if __name__ == "__main__":
    main()
