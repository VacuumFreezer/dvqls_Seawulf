#!/usr/bin/env python3
"""Generate Slurm task files for the 20-qubit PennyLane benchmark sweep."""

from __future__ import annotations

import argparse
import itertools
import json
import shlex
from datetime import datetime
from pathlib import Path


DEVICES = ("default.qubit", "lightning.qubit")
INTERFACES = ("numpy", "torch", "jax")
DIFF_METHODS = ("backprop", "adjoint", "finite_diff")
TARGETS = ("cpu", "gpu")


def slug(text: str) -> str:
    return text.replace(".", "_").replace("-", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full benchmark sweep task lists.")
    parser.add_argument("--output-root", default="GPU_Test/run")
    parser.add_argument("--tag", default=None, help="If omitted, a timestamped tag is used.")
    parser.add_argument("--num-qubits", type=int, default=20)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval-warmup", type=int, default=1)
    parser.add_argument("--eval-repeats", type=int, default=3)
    parser.add_argument("--grad-warmup", type=int, default=1)
    parser.add_argument("--grad-repeats", type=int, default=1)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_command(
    python_bin: str,
    benchmark_script: Path,
    output_json: Path,
    *,
    target: str,
    device: str,
    interface: str,
    diff_method: str,
    num_qubits: int,
    layers: int,
    seed: int,
    eval_warmup: int,
    eval_repeats: int,
    grad_warmup: int,
    grad_repeats: int,
    shots: int | None,
    verbose: bool,
) -> str:
    cmd = [
        python_bin,
        str(benchmark_script),
        "--compute-target",
        target,
        "--device",
        device,
        "--interface",
        interface,
        "--diff-method",
        diff_method,
        "--num-qubits",
        str(num_qubits),
        "--layers",
        str(layers),
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
        str(output_json),
    ]
    if shots is not None:
        cmd.extend(["--shots", str(shots)])
    if verbose:
        cmd.append("--verbose")
    return shlex.join(cmd)


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    tag = args.tag or f"hadamard20_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suite_dir = output_root / tag

    if suite_dir.exists():
        raise FileExistsError(
            f"Suite directory already exists: {suite_dir}. "
            "Choose a different --tag or remove the existing directory."
        )

    benchmark_script = (Path(__file__).resolve().parent / "benchmark_hadamard_test.py").resolve()
    combos = list(itertools.product(DEVICES, INTERFACES, DIFF_METHODS))

    (suite_dir / "results" / "cpu").mkdir(parents=True, exist_ok=False)
    (suite_dir / "results" / "gpu").mkdir(parents=True, exist_ok=False)
    (suite_dir / "task_logs" / "cpu").mkdir(parents=True, exist_ok=True)
    (suite_dir / "task_logs" / "gpu").mkdir(parents=True, exist_ok=True)
    (output_root / "slurm").mkdir(parents=True, exist_ok=True)

    task_lines: dict[str, list[str]] = {"cpu": [], "gpu": []}
    matrix_rows = []

    for target in TARGETS:
        target_seed_offset = 0 if target == "cpu" else 10_000
        for idx, (device, interface, diff_method) in enumerate(combos):
            cfg_slug = f"{slug(device)}__{slug(interface)}__{slug(diff_method)}"
            output_json = (suite_dir / "results" / target / f"{cfg_slug}.json").resolve()
            command = build_command(
                python_bin=args.python_bin,
                benchmark_script=benchmark_script,
                output_json=output_json,
                target=target,
                device=device,
                interface=interface,
                diff_method=diff_method,
                num_qubits=args.num_qubits,
                layers=args.layers,
                seed=args.seed + target_seed_offset + idx,
                eval_warmup=args.eval_warmup,
                eval_repeats=args.eval_repeats,
                grad_warmup=args.grad_warmup,
                grad_repeats=args.grad_repeats,
                shots=args.shots,
                verbose=args.verbose,
            )
            task_lines[target].append(command)
            matrix_rows.append(
                {
                    "target": target,
                    "device": device,
                    "interface": interface,
                    "diff_method": diff_method,
                    "json_out": str(output_json),
                    "seed": args.seed + target_seed_offset + idx,
                }
            )

    for target in TARGETS:
        tasks_file = suite_dir / f"tasks_{target}.txt"
        tasks_file.write_text("\n".join(task_lines[target]) + "\n", encoding="utf-8")

    config = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "suite_dir": str(suite_dir.resolve()),
        "tag": tag,
        "num_qubits_total": args.num_qubits,
        "num_system_qubits": args.num_qubits - 1,
        "layers": args.layers,
        "devices": list(DEVICES),
        "interfaces": list(INTERFACES),
        "diff_methods": list(DIFF_METHODS),
        "targets": list(TARGETS),
        "counts": {
            "cpu_tasks": len(task_lines["cpu"]),
            "gpu_tasks": len(task_lines["gpu"]),
            "total_tasks": len(task_lines["cpu"]) + len(task_lines["gpu"]),
        },
        "eval_warmup": args.eval_warmup,
        "eval_repeats": args.eval_repeats,
        "grad_warmup": args.grad_warmup,
        "grad_repeats": args.grad_repeats,
        "shots": args.shots,
        "python_bin": args.python_bin,
        "rows": matrix_rows,
    }
    with (suite_dir / "suite_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    print(f"TAG={tag}")
    print(f"SUITE_DIR={suite_dir.resolve()}")
    print(f"CPU_TASKS={len(task_lines['cpu'])}")
    print(f"GPU_TASKS={len(task_lines['gpu'])}")
    print("CPU_ARRAY_RANGE=0-17")
    print("GPU_ARRAY_RANGE=0-17")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
