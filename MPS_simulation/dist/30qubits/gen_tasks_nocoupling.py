#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
DIST_DIR = THIS_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the 30-qubit no-coupling J=0 task grid for structured init and random seeds 2, 42."
    )
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--report-every", type=int, default=5)
    return parser.parse_args()


def task_name(init_mode: str, seed: int | None) -> str:
    if init_mode == "structured_linspace":
        return "30qubits_nocoupling_structured"
    return f"30qubits_nocoupling_random_seed={seed}"


def main() -> None:
    args = parse_args()
    tag = args.tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")

    control_dir = THIS_DIR / "runs" / tag
    slurm_dir = THIS_DIR / "slurm"
    control_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = control_dir / "tasks.txt"
    manifest_path = control_dir / "task_manifest.json"

    task_specs = [
        {"init_mode": "structured_linspace", "seed": None},
        {"init_mode": "random_uniform", "seed": 2},
        {"init_mode": "random_uniform", "seed": 42},
    ]

    lines = []
    manifest = []
    for spec in task_specs:
        init_mode = str(spec["init_mode"])
        seed = spec["seed"]
        out_dir = DIST_DIR / "30qubits" / "runs" / tag / task_name(init_mode, seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = (
            "python MPS_simulation/dist/run_nocompress_case.py "
            "--label-dir 30qubits_nocoupling "
            f"--init-mode {init_mode} "
            f"--iterations {args.iterations} "
            f"--report-every {args.report_every} "
        )
        if seed is not None:
            cmd += f"--init-seed {int(seed)} "
        cmd += f"--out-dir {out_dir.as_posix()}"
        lines.append(cmd)
        manifest.append(
            {
                "label_dir": "30qubits_nocoupling",
                "init_mode": init_mode,
                "init_seed": seed,
                "out_dir": str(out_dir.resolve()),
                "command": cmd,
            }
        )

    tasks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] tasks written: {tasks_path}")
    print(f"[OK] manifest written: {manifest_path}")
    print(f"[OK] total tasks: {len(lines)}")
    print(f"[OK] control root: {control_dir}")
    print(f"[OK] slurm log dir: {slurm_dir}")


if __name__ == "__main__":
    main()
