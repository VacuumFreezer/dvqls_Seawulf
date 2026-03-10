#!/usr/bin/env python3
"""Generate CPU/GPU task files for Qiskit Hadamard-test benchmarks."""

from __future__ import annotations

import argparse
import itertools
import json
import shlex
from datetime import datetime
from pathlib import Path


EVAL_MODES = ("estimator", "sampler_v2")
GRAD_METHODS = ("reverse", "spsa")
TARGETS = ("cpu", "gpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Qiskit benchmark task lists.")
    parser.add_argument("--output-root", default="Qiskit_Test/run")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--num-qubits", type=int, default=20)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--sampler-shots", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260306)
    parser.add_argument("--eval-warmup", type=int, default=2)
    parser.add_argument("--eval-repeats", type=int, default=6)
    parser.add_argument("--grad-warmup", type=int, default=1)
    parser.add_argument("--grad-repeats", type=int, default=3)
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_command(
    python_bin: str,
    script_path: Path,
    json_out: Path,
    *,
    target: str,
    eval_mode: str,
    grad_method: str,
    num_qubits: int,
    layers: int,
    sampler_shots: int,
    seed: int,
    eval_warmup: int,
    eval_repeats: int,
    grad_warmup: int,
    grad_repeats: int,
    verbose: bool,
) -> str:
    cmd = [
        python_bin,
        str(script_path),
        "--compute-target",
        target,
        "--eval-mode",
        eval_mode,
        "--gradient-method",
        grad_method,
        "--num-qubits",
        str(num_qubits),
        "--layers",
        str(layers),
        "--sampler-shots",
        str(sampler_shots),
        "--seed",
        str(seed),
        "--eval-warmup",
        str(eval_warmup),
        "--eval-repeats",
        str(eval_repeats),
        "--grad-warmup",
        str(grad_warmup),
        "--grad-repeats",
        str(grad_repeats),
        "--json-out",
        str(json_out),
    ]
    if verbose:
        cmd.append("--verbose")
    return shlex.join(cmd)


def main() -> int:
    args = parse_args()
    out_root = Path(args.output_root)
    tag = args.tag or f"qiskit_hadamard20_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suite_dir = out_root / tag
    if suite_dir.exists():
        raise FileExistsError(f"Suite already exists: {suite_dir}")

    script_path = (Path(__file__).resolve().parent / "benchmark_hadamard_qiskit.py").resolve()
    combos = list(itertools.product(EVAL_MODES, GRAD_METHODS))

    (suite_dir / "results" / "cpu").mkdir(parents=True, exist_ok=False)
    (suite_dir / "results" / "gpu").mkdir(parents=True, exist_ok=False)
    (suite_dir / "task_logs" / "cpu").mkdir(parents=True, exist_ok=True)
    (suite_dir / "task_logs" / "gpu").mkdir(parents=True, exist_ok=True)
    (out_root / "slurm").mkdir(parents=True, exist_ok=True)

    tasks: dict[str, list[str]] = {"cpu": [], "gpu": []}
    rows = []
    for target in TARGETS:
        target_offset = 0 if target == "cpu" else 50_000
        for i, (eval_mode, grad_method) in enumerate(combos):
            name = f"{eval_mode}__{grad_method}.json"
            json_out = (suite_dir / "results" / target / name).resolve()
            seed = args.seed + target_offset + i
            cmd = build_command(
                python_bin=args.python_bin,
                script_path=script_path,
                json_out=json_out,
                target=target,
                eval_mode=eval_mode,
                grad_method=grad_method,
                num_qubits=args.num_qubits,
                layers=args.layers,
                sampler_shots=args.sampler_shots,
                seed=seed,
                eval_warmup=args.eval_warmup,
                eval_repeats=args.eval_repeats,
                grad_warmup=args.grad_warmup,
                grad_repeats=args.grad_repeats,
                verbose=args.verbose,
            )
            tasks[target].append(cmd)
            rows.append(
                {
                    "target": target,
                    "eval_mode": eval_mode,
                    "gradient_method": grad_method,
                    "seed": seed,
                    "json_out": str(json_out),
                }
            )

    for target in TARGETS:
        path = suite_dir / f"tasks_{target}.txt"
        path.write_text("\n".join(tasks[target]) + "\n", encoding="utf-8")

    suite_cfg = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "suite_dir": str(suite_dir.resolve()),
        "tag": tag,
        "num_qubits_total": args.num_qubits,
        "num_system_qubits": args.num_qubits - 1,
        "layers": args.layers,
        "sampler_shots": args.sampler_shots,
        "eval_modes": list(EVAL_MODES),
        "gradient_methods": list(GRAD_METHODS),
        "targets": list(TARGETS),
        "counts": {
            "cpu_tasks": len(tasks["cpu"]),
            "gpu_tasks": len(tasks["gpu"]),
            "total_tasks": len(tasks["cpu"]) + len(tasks["gpu"]),
        },
        "eval_warmup": args.eval_warmup,
        "eval_repeats": args.eval_repeats,
        "grad_warmup": args.grad_warmup,
        "grad_repeats": args.grad_repeats,
        "python_bin": args.python_bin,
        "rows": rows,
    }
    with (suite_dir / "suite_config.json").open("w", encoding="utf-8") as f:
        json.dump(suite_cfg, f, indent=2, sort_keys=True)

    print(f"TAG={tag}")
    print(f"SUITE_DIR={suite_dir.resolve()}")
    print(f"CPU_TASKS={len(tasks['cpu'])}")
    print(f"GPU_TASKS={len(tasks['gpu'])}")
    print("CPU_ARRAY_RANGE=0-3")
    print("GPU_ARRAY_RANGE=0-3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
