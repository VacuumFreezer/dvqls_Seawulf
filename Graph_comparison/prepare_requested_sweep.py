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

from Graph_comparison.topology_registry import iter_benchmarks


BENCHMARK_MEM = {
    "B1": "32G",
    "B2": "32G",
    "B3": "32G",
    "B4": "32G",
    "B5": "32G",
    "B6": "32G",
}


def build_task_command(
    *,
    python_bin: str,
    out_dir: Path,
    seed: int,
    benchmark_id: str,
    epochs: int,
    log_every: int,
) -> str:
    cmd = [
        python_bin,
        "Graph_comparison/seawulf_graph_comparison_qjit.py",
        "--static_ops",
        "Graph_comparison/static_ops_13q_xzx_fresh_4x4.py",
        "--benchmark",
        benchmark_id,
        "--epochs",
        str(int(epochs)),
        "--seed",
        str(int(seed)),
        "--lr",
        "0.005",
        "--decay",
        "0.9999",
        "--log_every",
        str(int(log_every)),
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
        "--out",
        out_dir.as_posix(),
    ]
    return " ".join(shlex.quote(part) for part in cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--python-bin", default="/gpfs/home/tonshen/.conda/envs/pennylane/bin/python")
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    root = ROOT
    run_root = root / "Graph_comparison" / "run"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_tag = args.tag.strip() or (
        f"xzx13_graph_compare_4x4_lr1e2_decay9999_e{int(args.epochs)}_log{int(args.log_every)}"
        f"_s{len(args.seeds)}_{timestamp}"
    )
    suite_dir = run_root / suite_tag
    slurm_dir = suite_dir / "slurm"
    suite_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "suite_tag": suite_tag,
        "created_at": datetime.now().isoformat(),
        "python_bin": args.python_bin,
        "seeds": [int(seed) for seed in args.seeds],
        "hyperparameters": {
            "epochs": int(args.epochs),
            "log_every": int(args.log_every),
            "lr": 0.005,
            "decay": 0.9999,
            "ansatz": "brickwall_ry_cz",
            "layers": 4,
            "init_mode": "uniform_pm_pi",
            "init_sigma_value": 1.0,
            "init_sigma_noise_std": 0.0,
            "init_lambda_value": 1.0,
            "init_lambda_noise_std": 0.0,
            "workflow": "4x4 graph comparison on the 13-qubit XZX stabilizer benchmark",
        },
        "benchmarks": [],
    }

    submit_lines = ["#!/bin/bash", "set -euo pipefail", ""]
    for benchmark in iter_benchmarks():
        benchmark_dir = suite_dir / f"benchmark={benchmark.benchmark_id}_{benchmark.name}"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        tasks_path = benchmark_dir / "tasks.txt"
        lines: list[str] = []
        output_dirs = []
        for seed in args.seeds:
            out_dir = benchmark_dir / f"seed={int(seed)}"
            out_dir.mkdir(parents=True, exist_ok=True)
            lines.append(
                build_task_command(
                    python_bin=args.python_bin,
                    out_dir=out_dir,
                    seed=int(seed),
                    benchmark_id=benchmark.benchmark_id,
                    epochs=int(args.epochs),
                    log_every=int(args.log_every),
                )
            )
            output_dirs.append(out_dir.as_posix())

        tasks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest["benchmarks"].append(
            {
                "benchmark_id": benchmark.benchmark_id,
                "name": benchmark.name,
                "row_graph": benchmark.row_graph,
                "column_graph": benchmark.column_graph,
                "mem": BENCHMARK_MEM[benchmark.benchmark_id],
                "time": "12:00:00",
                "tasks_file": tasks_path.as_posix(),
                "output_dirs": output_dirs,
            }
        )
        sbatch_parts = [
            "sbatch",
            "--job-name",
            f"graphcmp_{benchmark.benchmark_id}",
            "--array",
            f"0-{len(args.seeds) - 1}",
            "--mem",
            BENCHMARK_MEM[benchmark.benchmark_id],
            "--time",
            "12:00:00",
            "--output",
            f"Graph_comparison/run/{suite_tag}/slurm/%x_%A_%a.out",
            "--error",
            f"Graph_comparison/run/{suite_tag}/slurm/%x_%A_%a.err",
            "--export",
            f"ALL,TAG={suite_tag}/benchmark={benchmark.benchmark_id}_{benchmark.name}",
            "Graph_comparison/submit_array.slurm",
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
    for benchmark in iter_benchmarks():
        print(f"[OK] tasks={suite_dir / f'benchmark={benchmark.benchmark_id}_{benchmark.name}' / 'tasks.txt'}")


if __name__ == "__main__":
    main()
