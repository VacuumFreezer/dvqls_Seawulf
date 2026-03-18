#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MPS benchmark and optimization workflow for a named case."
    )
    parser.add_argument("--config", type=str, default=str(THIS_DIR / "param.yaml"))
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--optimize-only", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def main() -> None:
    args = parse_args()
    if args.benchmark_only and args.optimize_only:
        raise ValueError("Use at most one of --benchmark-only or --optimize-only.")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_benchmark = not args.optimize_only
    run_optimize = not args.benchmark_only

    summary = {
        "case": args.case,
        "config": str(Path(args.config).resolve()),
        "out_dir": str(out_dir),
        "benchmark_enabled": bool(run_benchmark),
        "optimize_enabled": bool(run_optimize),
        "artifacts": {},
    }

    python_exe = sys.executable
    if run_benchmark:
        bench_json = out_dir / "benchmark.json"
        bench_report = out_dir / "benchmark_report.md"
        bench_config = out_dir / "benchmark_config_used.yaml"
        run_command(
            [
                python_exe,
                str(THIS_DIR / "quimb_dist_eq26_2x2_benchmark.py"),
                "--config",
                args.config,
                "--case",
                args.case,
                "--out-json",
                str(bench_json),
                "--out-report",
                str(bench_report),
                "--out-config",
                str(bench_config),
            ]
        )
        summary["artifacts"]["benchmark"] = {
            "json": str(bench_json),
            "report": str(bench_report),
            "config": str(bench_config),
        }

    if run_optimize:
        opt_json = out_dir / "optimize.json"
        opt_fig = out_dir / "optimize_history.png"
        opt_report = out_dir / "optimize_report.md"
        opt_history = out_dir / "optimize_metrics.jsonl"
        opt_checkpoint = out_dir / "optimize_checkpoint.json"
        opt_config = out_dir / "optimize_config_used.yaml"
        run_command(
            [
                python_exe,
                str(THIS_DIR / "quimb_dist_eq26_2x2_optimize.py"),
                "--config",
                args.config,
                "--case",
                args.case,
                "--out-json",
                str(opt_json),
                "--out-figure",
                str(opt_fig),
                "--out-report",
                str(opt_report),
                "--out-history",
                str(opt_history),
                "--out-checkpoint",
                str(opt_checkpoint),
                "--out-config",
                str(opt_config),
            ]
        )
        summary["artifacts"]["optimize"] = {
            "json": str(opt_json),
            "figure": str(opt_fig),
            "report": str(opt_report),
            "history": str(opt_history),
            "checkpoint": str(opt_checkpoint),
            "config": str(opt_config),
        }

    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote run summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
