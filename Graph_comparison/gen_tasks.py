#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Graph_comparison.topology_registry import get_benchmark_spec


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required: pip install pyyaml") from exc

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _append_optional_scalar(cmd: list[str], flag: str, value) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    cmd.extend([flag, str(value)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="Graph_comparison/config.yaml")
    ap.add_argument("--tag", default="", help="Override the run tag from YAML.")
    args = ap.parse_args()

    root = ROOT
    cfg_path = (root / args.config).resolve()
    cfg = load_yaml(cfg_path)

    run_root = root / str(cfg.get("run_root", "Graph_comparison/run"))
    tag = args.tag.strip() or str(cfg.get("tag", "")).strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    group_dir = run_root / tag
    slurm_dir = run_root / "slurm"
    group_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)

    tasks_path = group_dir / "tasks.txt"
    common = cfg.get("common", {})
    problem = cfg.get("problem", {})
    python_bin = str(cfg.get("python_bin", "python"))
    static_ops = str(problem.get("static_ops", "Graph_comparison/static_ops_13q_xzx_fresh_4x4.py"))
    benchmarks = cfg.get("benchmarks", [])
    cases = cfg.get("cases", [])
    lr_sets = cfg.get("lr_sets", [])
    decay_sets = cfg.get("decay_sets", [])
    seeds = cfg.get("seeds", [0])

    ansatz = str(common.get("ansatz", "brickwall_ry_cz"))
    layers = int(common.get("layers", 1))
    epochs = int(common.get("epochs", 2000))
    log_every = int(common.get("log_every", 10))
    repeat_cz_each_layer = bool(common.get("repeat_cz_each_layer", False))
    local_ry_support = str(common.get("local_ry_support", "")).strip()

    optional_init_keys = (
        ("--init_mode", common.get("init_mode")),
        ("--init_angle_center", common.get("init_angle_center")),
        ("--init_angle_noise_std", common.get("init_angle_noise_std")),
        ("--init_sigma_value", common.get("init_sigma_value")),
        ("--init_sigma_noise_std", common.get("init_sigma_noise_std")),
        ("--init_lambda_value", common.get("init_lambda_value")),
        ("--init_lambda_noise_std", common.get("init_lambda_noise_std")),
    )

    lines: list[str] = []
    for benchmark_cfg in benchmarks:
        benchmark_id = str(benchmark_cfg["id"]).strip().upper()
        spec = get_benchmark_spec(benchmark_id)
        for case in cases:
            script = str(case["script"])
            script_path = (root / script).resolve()
            if not script_path.exists():
                raise SystemExit(f"Optimizer script not found: {script_path}")

            for decay_cfg in decay_sets:
                for lr_cfg in lr_sets:
                    for seed in seeds:
                        out_dir = group_dir / (
                            f"benchmark={spec.benchmark_id}_{spec.name}"
                            f"__row={spec.row_graph}__col={spec.column_graph}"
                            f"__case={case['name']}"
                            f"__decay={decay_cfg['tag']}__lr={lr_cfg['tag']}__seed={int(seed)}"
                        )
                        out_dir.mkdir(parents=True, exist_ok=True)

                        cmd = [
                            python_bin,
                            script,
                            "--static_ops",
                            static_ops,
                            "--benchmark",
                            spec.benchmark_id,
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
                        for flag, value in optional_init_keys:
                            _append_optional_scalar(cmd, flag, value)
                        lines.append(" ".join(cmd))

    with tasks_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")

    print(f"[OK] tasks written: {tasks_path}")
    print(f"[OK] total tasks: {len(lines)}")
    print(f"[OK] outputs root: {group_dir}")


if __name__ == "__main__":
    main()
