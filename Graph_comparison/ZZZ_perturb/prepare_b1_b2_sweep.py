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

from Graph_comparison.topology_registry import get_benchmark_spec


BENCHMARK_IDS = ("B1", "B2")


def build_task_command(
    *,
    python_bin: str,
    out_dir: Path,
    benchmark_id: str,
    seed: int,
    epochs: int,
) -> str:
    cmd = [
        python_bin,
        "Graph_comparison/seawulf_graph_comparison_qjit.py",
        "--static_ops",
        "Graph_comparison/ZZZ_perturb/static_ops_13q_zzz_perturb_4x4.py",
        "--benchmark",
        benchmark_id,
        "--epochs",
        str(int(epochs)),
        "--seed",
        str(int(seed)),
        "--lr",
        "0.01",
        "--decay",
        "0.9999",
        "--log_every",
        "20",
        "--ansatz",
        "hadamard_brickwall_ry_cz",
        "--layers",
        "2",
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
        "--out",
        out_dir.as_posix(),
    ]
    return " ".join(shlex.quote(part) for part in cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="20260322_zzz13_graph_compare_B1_B2_seed0")
    ap.add_argument("--python-bin", default="/gpfs/home/tonshen/.conda/envs/pennylane/bin/python")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--mem", default="32G")
    ap.add_argument("--time", default="12:00:00")
    args = ap.parse_args()

    suite_dir = ROOT / "Graph_comparison" / "ZZZ_perturb" / "run" / args.tag
    slurm_dir = suite_dir / "slurm"
    suite_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "suite_tag": args.tag,
        "created_at": datetime.now().astimezone().isoformat(),
        "python_bin": args.python_bin,
        "static_ops": "Graph_comparison/ZZZ_perturb/static_ops_13q_zzz_perturb_4x4.py",
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "ansatz": "hadamard_brickwall_ry_cz",
        "layers": 2,
        "lr": 0.01,
        "benchmarks": [],
    }

    submit_lines = ["#!/bin/bash", "set -euo pipefail", ""]
    for benchmark_id in BENCHMARK_IDS:
        spec = get_benchmark_spec(benchmark_id)
        bench_dir = suite_dir / f"benchmark={spec.benchmark_id}_{spec.name}"
        bench_dir.mkdir(parents=True, exist_ok=True)
        out_dir = bench_dir / f"seed={int(args.seed)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        tasks_path = bench_dir / "tasks.txt"
        tasks_path.write_text(
            build_task_command(
                python_bin=args.python_bin,
                out_dir=out_dir,
                benchmark_id=benchmark_id,
                seed=int(args.seed),
                epochs=int(args.epochs),
            )
            + "\n",
            encoding="utf-8",
        )
        manifest["benchmarks"].append(
            {
                "benchmark_id": spec.benchmark_id,
                "name": spec.name,
                "row_graph": spec.row_graph,
                "column_graph": spec.column_graph,
                "tasks_file": tasks_path.as_posix(),
                "out_dir": out_dir.as_posix(),
                "mem": args.mem,
                "time": args.time,
            }
        )

        sbatch_parts = [
            "sbatch",
            "--job-name",
            f"zzzcmp_{spec.benchmark_id}",
            "--array",
            "0-0",
            "--mem",
            args.mem,
            "--time",
            args.time,
            "--output",
            f"Graph_comparison/ZZZ_perturb/run/{args.tag}/slurm/%x_%A_%a.out",
            "--error",
            f"Graph_comparison/ZZZ_perturb/run/{args.tag}/slurm/%x_%A_%a.err",
            "--export",
            f"ALL,TAG=../ZZZ_perturb/run/{args.tag}/benchmark={spec.benchmark_id}_{spec.name}",
            "Graph_comparison/submit_array.slurm",
        ]
        submit_lines.append(" ".join(shlex.quote(part) for part in sbatch_parts))

    (suite_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    submit_path = suite_dir / "submit_commands.sh"
    submit_path.write_text("\n".join(submit_lines) + "\n", encoding="utf-8")
    submit_path.chmod(0o755)

    print(f"[OK] suite_dir={suite_dir}")
    print(f"[OK] manifest={suite_dir / 'manifest.json'}")
    print(f"[OK] submit_commands={submit_path}")


if __name__ == "__main__":
    main()
