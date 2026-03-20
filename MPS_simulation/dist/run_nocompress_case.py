#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]

SCRIPT_BY_LABEL = {
    "5qubits": THIS_DIR / "5qubits" / "quimb_dist_eq26_2x2_optimize_5q_nocompress.py",
    "10qubits": THIS_DIR / "10qubits" / "quimb_dist_eq26_2x2_optimize_10q_nocompress.py",
    "30qubits": THIS_DIR / "30qubits" / "quimb_dist_eq26_2x2_optimize_30q_nocompress.py",
    "30qubits_nocoupling": THIS_DIR / "30qubits" / "quimb_dist_eq26_2x2_optimize_30q_nocoupling.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one low-memory nocompress optimization case and collect all artifacts in one directory."
    )
    parser.add_argument("--label-dir", type=str, choices=tuple(SCRIPT_BY_LABEL), required=True)
    parser.add_argument("--init-mode", type=str, choices=("structured_linspace", "random_uniform"), required=True)
    parser.add_argument("--init-seed", type=int, default=1234)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    script_path = SCRIPT_BY_LABEL[args.label_dir]
    cmd = [
        sys.executable,
        str(script_path),
        "--init-mode",
        args.init_mode,
        "--iterations",
        str(args.iterations),
        "--report-every",
        str(args.report_every),
        "--out-dir",
        str(out_dir),
    ]
    if args.init_mode == "random_uniform":
        cmd.extend(["--init-seed", str(args.init_seed)])

    run_command(cmd)

    summary = {
        "label_dir": args.label_dir,
        "script": str(script_path.resolve()),
        "init_mode": args.init_mode,
        "init_seed": args.init_seed if args.init_mode == "random_uniform" else None,
        "iterations": args.iterations,
        "report_every": args.report_every,
        "out_dir": str(out_dir),
        "artifacts": {
            "json": str((out_dir / "optimize.json").resolve()),
            "report": str((out_dir / "optimize_report.md").resolve()),
            "history": str((out_dir / "optimize_metrics.jsonl").resolve()),
            "checkpoint": str((out_dir / "optimize_checkpoint.json").resolve()),
            "config": str((out_dir / "optimize_config_used.yaml").resolve()),
        },
    }
    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote run summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
