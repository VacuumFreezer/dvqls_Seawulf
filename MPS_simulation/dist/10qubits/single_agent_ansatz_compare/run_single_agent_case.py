#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
OPTIMIZE_SCRIPT = THIS_DIR / "single_agent_ansatz_optimize.py"
DEFAULT_CONFIG = THIS_DIR / "param.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one prepared single-agent ansatz benchmark and collect artifacts in one output directory."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--report-every", type=int, default=None)
    parser.add_argument("--init-seed", type=int, default=None)
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(OPTIMIZE_SCRIPT),
        "--config",
        str(Path(args.config).resolve()),
        "--case",
        args.case,
        "--out-dir",
        str(out_dir),
    ]
    if args.iterations is not None:
        cmd.extend(["--iterations", str(args.iterations)])
    if args.report_every is not None:
        cmd.extend(["--report-every", str(args.report_every)])
    if args.init_seed is not None:
        cmd.extend(["--init-seed", str(args.init_seed)])

    run_command(cmd)

    summary = {
        "config": str(Path(args.config).resolve()),
        "case": args.case,
        "iterations_override": args.iterations,
        "report_every_override": args.report_every,
        "init_seed_override": args.init_seed,
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
