#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
DIST_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[2]

LABEL_ORDER = ("5qubits", "10qubits", "30qubits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Seawulf tasks for low-memory nocompress runs across 5q, 10q, and 30q cases."
    )
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--random-seeds", type=str, default="1234,1235,1236")
    return parser.parse_args()


def parse_seed_text(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def task_name(label_dir: str, init_mode: str, seed: int | None) -> str:
    if init_mode == "structured_linspace":
        return f"{label_dir}_structured"
    return f"{label_dir}_random_seed={seed}"


def main() -> None:
    args = parse_args()
    tag = args.tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    seeds = parse_seed_text(args.random_seeds)

    control_dir = THIS_DIR / "runs" / tag
    slurm_dir = THIS_DIR / "slurm"
    control_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = control_dir / "tasks.txt"
    manifest_path = control_dir / "task_manifest.json"

    task_specs: list[dict[str, object]] = []
    task_specs.extend(
        {"label_dir": label_dir, "init_mode": "structured_linspace", "seed": None}
        for label_dir in LABEL_ORDER
    )
    for seed in seeds:
        task_specs.extend(
            {"label_dir": label_dir, "init_mode": "random_uniform", "seed": seed}
            for label_dir in LABEL_ORDER
        )

    lines = []
    manifest = []
    for spec in task_specs:
        label_dir = str(spec["label_dir"])
        init_mode = str(spec["init_mode"])
        seed = spec["seed"]
        out_dir = DIST_DIR / label_dir / "runs" / tag / task_name(label_dir, init_mode, seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = (
            "python MPS_simulation/dist/run_nocompress_case.py "
            f"--label-dir {label_dir} "
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
                "label_dir": label_dir,
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
    print("[OK] output roots:")
    for label_dir in LABEL_ORDER:
        print(f"  - {(DIST_DIR / label_dir / 'runs' / tag).resolve()}")


if __name__ == "__main__":
    main()
