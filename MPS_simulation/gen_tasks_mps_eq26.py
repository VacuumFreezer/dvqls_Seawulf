#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from quimb_dist_eq26_common import load_yaml_config


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Seawulf task commands for the MPS Eq. (26) workflow.")
    parser.add_argument("--config", type=str, default="MPS_simulation/param.yaml")
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = (REPO_ROOT / args.config).resolve()
    cfg = load_yaml_config(cfg_path)

    jobs = cfg.get("jobs", {})
    cases = list(jobs.get("cases", list((cfg.get("cases") or {}).keys())))
    if not cases:
        raise ValueError("No cases were found in the jobs.cases or cases sections of the config.")

    run_root = REPO_ROOT / jobs.get("run_root", "MPS_simulation/runs")
    tag = args.tag.strip() or str(jobs.get("tag", "")).strip() or datetime.now().strftime("%Y%m%d_%H%M%S")

    group_dir = run_root / tag
    group_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = group_dir / "tasks.txt"

    lines = []
    for case in cases:
        out_dir = group_dir / f"case={case}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = (
            f"python MPS_simulation/run_mps_case.py "
            f"--config {cfg_path.as_posix()} "
            f"--case {case} "
            f"--out-dir {out_dir.as_posix()}"
        )
        lines.append(cmd)

    tasks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] config: {cfg_path}")
    print(f"[OK] tasks written: {tasks_path}")
    print(f"[OK] total tasks: {len(lines)}")
    print(f"[OK] outputs root: {group_dir}")


if __name__ == "__main__":
    main()
