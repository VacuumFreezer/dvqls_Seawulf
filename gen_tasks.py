#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime

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
    ap.add_argument("--tag", default="", help="cover tag in YAML (optional)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    cfg = load_yaml(root / args.config)

    run_root = root / cfg.get("run_root", "run")

    tag = args.tag.strip() or str(cfg.get("tag", "")).strip()
    if not tag:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    group_dir = run_root / tag
    group_dir.mkdir(parents=True, exist_ok=True)

    tasks_path = group_dir / "tasks.txt"

    problems = cfg["problems"]
    cases = cfg["cases"]
    topologies = cfg.get("topologies", [{"name": "line"}])
    decays_sets = cfg.get("decay_sets")

    lr_sets = cfg.get("lr_sets")
    seeds = cfg.get("seeds", [0])
    # system_ids = cfg.get("system_ids", [0])

    common = cfg.get("common", {})
    epochs = int(common.get("epochs", 10))
    # decay = float(common.get("decay", 1.0))
    log_every = int(common.get("log_every", 1))
    # lr = float(common.get("lr", 0.01))

    lines = []
    skipped = 0

    for p in problems:
        for c in cases:
            script_path = root / c["script"]
            if not script_path.exists():
                print(f"[WARN] script not found, skip: {c['script']}")
                skipped += 1
                continue

            # for sys_id in system_ids:
            for topo in topologies:
                for seed in seeds:
                    for decay in decays_sets:
                        for lr in lr_sets:
                            out_dir = group_dir / (
                                f"prob={p['name']}__case={c['name']}__graph={str(topo['name'])}__decay={decay['tag']}__lr={lr['tag']}__seed={seed}"
                            )
                            out_dir.mkdir(parents=True, exist_ok=True)

                            cmd = (
                                f"python {c['script']} "
                                f"--static_ops {p['module']} "
                                f"--topology {topo['name']} "
                                # f"--system_id {int(sys_id)} "
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
    if skipped:
        print(f"[INFO] skipped scripts: {skipped}")
    print(f"[OK] outputs root: {group_dir}")

if __name__ == "__main__":
    main()
