#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("No PyYAML: pip install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="e.g. params/sweep.yaml")
    ap.add_argument("--tag", default="", help="run tag (optional)")
    ap.add_argument(
        "--script",
        default="optimization/seawulf_centralized_vqls.py",
        help="centralized optimizer script",
    )
    ap.add_argument(
        "--case_name",
        default="centralized_vqls",
        help="name shown in output directory tags",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    cfg = load_yaml(root / args.config)

    run_root = root / cfg.get("run_root", "run")
    yaml_tag = str(cfg.get("tag", "")).strip()
    tag = args.tag.strip() or (f"{yaml_tag}_centralized" if yaml_tag else "")
    if not tag:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    group_dir = run_root / tag
    group_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = group_dir / "tasks.txt"

    problems = cfg["problems"]
    lr_sets = cfg.get("lr_sets", [])
    decay_sets = cfg.get("decay_sets", [])
    seeds = cfg.get("seeds", [0])
    common = cfg.get("common", {})
    epochs = int(common.get("epochs", 10))
    log_every = int(common.get("log_every", 1))

    script_path = root / args.script
    if not script_path.exists():
        raise SystemExit(f"Script not found: {args.script}")

    lines = []
    for p in problems:
        for seed in seeds:
            for decay in decay_sets:
                for lr in lr_sets:
                    out_dir = group_dir / (
                        f"prob={p['name']}__case={args.case_name}__decay={decay['tag']}__lr={lr['tag']}__seed={seed}"
                    )
                    out_dir.mkdir(parents=True, exist_ok=True)

                    cmd = (
                        f"python {args.script} "
                        f"--static_ops {p['module']} "
                        f"--epochs {epochs} "
                        f"--seed {int(seed)} "
                        f"--lr {lr['lr']} "
                        f"--decay {decay['decay']} "
                        f"--log_every {log_every} "
                        f"--out {out_dir.as_posix()}"
                    )
                    lines.append(cmd)

    with open(tasks_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"[OK] tasks written: {tasks_path}")
    print(f"[OK] total tasks: {len(lines)}")
    print(f"[OK] outputs root: {group_dir}")


if __name__ == "__main__":
    main()
