#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required: pip install pyyaml") from exc

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="Partition_comparison_qjit/New_stabilizer/config.yaml")
    ap.add_argument("--tag", default="", help="Override the run tag from YAML.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    cfg_path = (root / args.config).resolve()
    cfg = load_yaml(cfg_path)

    run_root = root / str(cfg.get("run_root", "Partition_comparison_qjit/New_stabilizer/run"))
    tag = args.tag.strip() or str(cfg.get("tag", "")).strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    group_dir = run_root / tag
    slurm_dir = run_root / "slurm"
    group_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)

    tasks_path = group_dir / "tasks.txt"
    common = cfg.get("common", {})
    python_bin = str(cfg.get("python_bin", "python"))
    problems = cfg.get("problems", [])
    cases = cfg.get("cases", [])
    lr_sets = cfg.get("lr_sets", [])
    decay_sets = cfg.get("decay_sets", [])
    seeds = cfg.get("seeds", [0])

    topology = str(common.get("topology", "line"))
    ansatz = str(common.get("ansatz", "brickwall_ry_cz"))
    layers = int(common.get("layers", 1))
    epochs = int(common.get("epochs", 2000))
    log_every = int(common.get("log_every", 10))
    repeat_cz_each_layer = bool(common.get("repeat_cz_each_layer", False))
    local_ry_support = str(common.get("local_ry_support", "")).strip()

    lines = []
    for problem in problems:
        static_ops = str(problem["static_ops"])
        for case in cases:
            script = str(case["script"])
            script_path = (root / script).resolve()
            if not script_path.exists():
                raise SystemExit(f"Optimizer script not found: {script_path}")

            for decay_cfg in decay_sets:
                for lr_cfg in lr_sets:
                    for seed in seeds:
                        out_dir = group_dir / (
                            f"partition={problem['name']}__case={case['name']}__topo={topology}"
                            f"__decay={decay_cfg['tag']}__lr={lr_cfg['tag']}__seed={int(seed)}"
                        )
                        out_dir.mkdir(parents=True, exist_ok=True)

                        cmd = [
                            python_bin,
                            script,
                            "--static_ops",
                            static_ops,
                            "--topology",
                            topology,
                            "--epochs",
                            str(epochs),
                            "--seed",
                            str(int(seed)),
                            "--lr",
                            str(lr_cfg["lr"]),
                            "--decay",
                            str(decay_cfg["decay"]),
                            "--log_every",
                            str(log_every),
                            "--ansatz",
                            ansatz,
                            "--layers",
                            str(layers),
                            "--out",
                            out_dir.as_posix(),
                        ]
                        if repeat_cz_each_layer:
                            cmd.append("--repeat_cz_each_layer")
                        if local_ry_support:
                            cmd.extend(["--local_ry_support", local_ry_support])
                        lines.append(" ".join(cmd))

    with tasks_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")

    print(f"[OK] tasks written: {tasks_path}")
    print(f"[OK] total tasks: {len(lines)}")
    print(f"[OK] outputs root: {group_dir}")


if __name__ == "__main__":
    main()
