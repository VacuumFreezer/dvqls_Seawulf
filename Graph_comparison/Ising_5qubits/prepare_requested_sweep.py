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

from Graph_comparison.topology_registry import iter_benchmarks


BENCHMARK_MEM = {
    "B1": "64G",
    "B2": "64G",
    "B3": "64G",
    "B4": "64G",
    "B5": "64G",
    "B6": "64G",
}


def _lr_tag(value: float) -> str:
    text = f"{value:.0e}".replace("+0", "").replace("+", "")
    return text.replace("-0", "-")


def build_task_command(
    *,
    python_bin: str,
    out_dir: Path,
    seed: int,
    benchmark_id: str,
    epochs: int,
    lr: float,
    decay: float,
    log_every: int,
    layers: int,
) -> str:
    cmd = [
        python_bin,
        "Graph_comparison/Ising_5qubits/seawulf_graph_comparison_ising_qjit.py",
        "--static_ops",
        "Graph_comparison/Ising_5qubits/static_ops_ising_4x4_5q.py",
        "--benchmark",
        benchmark_id,
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
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--python-bin", default="/gpfs/home/tonshen/.conda/envs/pennylane/bin/python")
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--decay", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--layers", type=int, default=3)
    args = ap.parse_args()

    root = ROOT
    run_root = root / "Graph_comparison" / "Ising_5qubits" / "run"
    timestamp = datetime.now().strftime("%Y%m%d")
    seed_suffix = f"seed{args.seeds[0]}" if len(args.seeds) == 1 else f"s{len(args.seeds)}"
    decay_suffix = "fixlr" if abs(float(args.decay) - 1.0) < 1.0e-15 else f"decay{args.decay:g}"
    suite_tag = args.tag.strip() or (
        f"{timestamp}_graph_compare_ising5q_4x4_e{int(args.epochs)}"
        f"_lr{_lr_tag(float(args.lr))}_log{int(args.log_every)}_{seed_suffix}_{decay_suffix}"
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
            "lr": float(args.lr),
            "decay": float(args.decay),
            "layers": int(args.layers),
            "ansatz": "qml.BasicEntanglerLayers(weights, wires=range(n_qubits), rotation=qml.RY)",
            "workflow": "4x4 graph comparison on the 7-qubit Ising benchmark with 5 local qubits per agent",
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
                    lr=float(args.lr),
                    decay=float(args.decay),
                    log_every=int(args.log_every),
                    layers=int(args.layers),
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
                "time": "24:00:00",
                "tasks_file": tasks_path.as_posix(),
                "output_dirs": output_dirs,
            }
        )
        sbatch_parts = [
            "sbatch",
            "--job-name",
            f"ising5q_{benchmark.benchmark_id}",
            "--array",
            f"0-{len(args.seeds) - 1}",
            "--mem",
            BENCHMARK_MEM[benchmark.benchmark_id],
            "--time",
            "24:00:00",
            "--output",
            f"Graph_comparison/Ising_5qubits/run/{suite_tag}/slurm/%x_%A_%a.out",
            "--error",
            f"Graph_comparison/Ising_5qubits/run/{suite_tag}/slurm/%x_%A_%a.err",
            "--export",
            f"ALL,TAG={suite_tag}/benchmark={benchmark.benchmark_id}_{benchmark.name}",
            "Graph_comparison/Ising_5qubits/submit_array.slurm",
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
