#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime
from pathlib import Path


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("No PyYAML: pip install pyyaml") from e
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a YAML mapping: {path}")
    return data


def deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_seed_list(seed_text: str) -> list[int]:
    vals = []
    for x in seed_text.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    if not vals:
        raise SystemExit("Parsed empty seed list from --seeds.")
    return vals


def parse_seed_values(values) -> list[int]:
    if values is None:
        return []
    if isinstance(values, str):
        text = values.strip()
        if not text:
            return []
        return parse_seed_list(text)
    if isinstance(values, list):
        return [int(x) for x in values]
    raise SystemExit("`taskgen.seed_scan.seeds` must be a string or a list of integers.")


def draw_random_unique_seeds(num: int, low: int, high: int, rng_seed: int | None) -> list[int]:
    if num <= 0:
        raise SystemExit("--num-seeds must be > 0.")
    if high <= low:
        raise SystemExit("--seed-max must be greater than --seed-min.")
    span = high - low
    if num > span:
        raise SystemExit(
            f"Cannot draw {num} unique seeds from range [{low}, {high}) with only {span} values."
        )
    import numpy as np

    rng = np.random.default_rng(rng_seed)
    arr = rng.choice(np.arange(low, high, dtype=np.int64), size=num, replace=False)
    return [int(x) for x in arr.tolist()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate Slurm task list for random-seed scan of centralized VQLS scripts "
            "(script selected from combined config.yaml)."
        )
    )
    ap.add_argument(
        "--base-config",
        default="cen_vqls/config.yaml",
        help="Combined YAML config (contains taskgen.optimization_script and script_map).",
    )
    ap.add_argument(
        "--script",
        default="",
        help="Optional explicit script path override. If empty, use taskgen.optimization_script + taskgen.script_map.",
    )
    ap.add_argument(
        "--optimization-script",
        default="",
        help="Optional key override for taskgen.optimization_script (e.g., qjit_vqls or residual).",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="Run tag under run/<tag>. If empty, use taskgen.tag from YAML, else timestamp.",
    )
    ap.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated explicit seeds, e.g. 2,3,42. If set, random drawing is skipped.",
    )
    ap.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="Number of random seeds (override taskgen.seed_scan.num_seeds).",
    )
    ap.add_argument("--seed-min", type=int, default=None, help="Random-seed lower bound (inclusive).")
    ap.add_argument("--seed-max", type=int, default=None, help="Random-seed upper bound (exclusive).")
    ap.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        help="RNG seed for generating random seeds (override taskgen.seed_scan.rng_seed).",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    base_cfg_path = (root / args.base_config).resolve()
    script_path = (root / args.script).resolve()

    if not base_cfg_path.exists():
        raise SystemExit(f"Base config not found: {base_cfg_path}")

    base_cfg = load_yaml(base_cfg_path)
    taskgen_cfg = base_cfg.get("taskgen", {})
    if not isinstance(taskgen_cfg, dict):
        raise SystemExit("taskgen section must be a mapping in base config.")

    script_map = taskgen_cfg.get("script_map", {})
    if not isinstance(script_map, dict) or not script_map:
        raise SystemExit("taskgen.script_map must be a non-empty mapping in base config.")

    selected_key = args.optimization_script.strip() or str(taskgen_cfg.get("optimization_script", "")).strip()
    if args.script.strip():
        script_path = (root / args.script).resolve()
        selected_key = selected_key or "explicit_script_override"
    else:
        if not selected_key:
            raise SystemExit(
                "No optimization script selected. Set taskgen.optimization_script in config "
                "or pass --optimization-script / --script."
            )
        if selected_key not in script_map:
            raise SystemExit(
                f"Unknown optimization script key `{selected_key}`. "
                f"Available keys: {sorted(script_map.keys())}"
            )
        script_path = (root / str(script_map[selected_key])).resolve()

    if not script_path.exists():
        raise SystemExit(f"Script not found: {script_path}")

    seed_scan_cfg = taskgen_cfg.get("seed_scan", {})
    if seed_scan_cfg and not isinstance(seed_scan_cfg, dict):
        raise SystemExit("taskgen.seed_scan must be a mapping if provided.")

    seed_mode = str(seed_scan_cfg.get("mode", "random")).strip().lower()
    cfg_num_seeds = int(seed_scan_cfg.get("num_seeds", 10))
    cfg_seed_min = int(seed_scan_cfg.get("seed_min", 0))
    cfg_seed_max = int(seed_scan_cfg.get("seed_max", 1_000_000))
    cfg_rng_seed = seed_scan_cfg.get("rng_seed", None)
    cfg_seeds = parse_seed_values(seed_scan_cfg.get("seeds", []))

    cli_seeds = (args.seeds or "").strip()
    if cli_seeds:
        seeds = parse_seed_list(cli_seeds)
        seed_source = "cli_explicit"
    else:
        # CLI numeric overrides take precedence over YAML defaults for random mode.
        use_num_seeds = int(args.num_seeds) if args.num_seeds is not None else cfg_num_seeds
        use_seed_min = int(args.seed_min) if args.seed_min is not None else cfg_seed_min
        use_seed_max = int(args.seed_max) if args.seed_max is not None else cfg_seed_max
        use_rng_seed = args.rng_seed if args.rng_seed is not None else cfg_rng_seed

        if seed_mode == "explicit":
            if not cfg_seeds:
                raise SystemExit(
                    "taskgen.seed_scan.mode=explicit but no seeds provided in taskgen.seed_scan.seeds."
                )
            seeds = cfg_seeds
            seed_source = "yaml_explicit"
        elif seed_mode == "random":
            seeds = draw_random_unique_seeds(
                num=use_num_seeds,
                low=use_seed_min,
                high=use_seed_max,
                rng_seed=use_rng_seed,
            )
            seed_source = "yaml_random"
        else:
            raise SystemExit("taskgen.seed_scan.mode must be one of: random, explicit.")

    cfg_tag = str(taskgen_cfg.get("tag", "")).strip()
    tag = args.tag.strip() or cfg_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / "run" / tag
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = run_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    tasks_path = run_dir / "tasks.txt"

    # Merge script-specific overrides from combined config.
    profile_overrides_all = taskgen_cfg.get("profile_overrides", {})
    if profile_overrides_all and not isinstance(profile_overrides_all, dict):
        raise SystemExit("taskgen.profile_overrides must be a mapping if provided.")

    effective_cfg = json.loads(json.dumps(base_cfg))
    if selected_key in profile_overrides_all:
        overrides = profile_overrides_all[selected_key]
        if not isinstance(overrides, dict):
            raise SystemExit(f"taskgen.profile_overrides.{selected_key} must be a mapping.")
        deep_update(effective_cfg, overrides)

    # Keep taskgen metadata out of per-task runtime configs.
    effective_cfg.pop("taskgen", None)
    base_report_tag = str(effective_cfg.get("report", {}).get("tag", f"centralized_vqls_{selected_key}"))
    cmds: list[str] = []

    for seed in seeds:
        per_seed_report_dir = report_dir / f"seed={seed}"
        per_seed_report_dir.mkdir(parents=True, exist_ok=True)

        report_tag = f"{base_report_tag}__seed={seed}"
        seed_cfg = json.loads(json.dumps(effective_cfg))
        seed_cfg.setdefault("runtime", {})
        seed_cfg["runtime"]["seed"] = int(seed)
        seed_cfg.setdefault("report", {})
        seed_cfg["report"]["tag"] = report_tag
        seed_cfg["report"]["out_dir"] = per_seed_report_dir.as_posix()

        seed_cfg_path = configs_dir / f"config_seed={seed}.yaml"
        try:
            import yaml
        except ImportError as e:
            raise SystemExit("No PyYAML: pip install pyyaml") from e
        with seed_cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(seed_cfg, f, sort_keys=False)

        cmd = (
            f"python {shlex.quote(script_path.as_posix())} "
            f"--config {shlex.quote(seed_cfg_path.as_posix())}"
        )
        cmds.append(cmd)

    with tasks_path.open("w", encoding="utf-8") as f:
        for cmd in cmds:
            f.write(cmd + "\n")

    meta = {
        "tag": tag,
        "base_config": base_cfg_path.as_posix(),
        "selected_optimization_script": selected_key,
        "script": script_path.as_posix(),
        "seed_source": seed_source,
        "seed_scan_effective": {
            "mode": seed_mode,
            "num_seeds": (int(args.num_seeds) if args.num_seeds is not None else cfg_num_seeds),
            "seed_min": (int(args.seed_min) if args.seed_min is not None else cfg_seed_min),
            "seed_max": (int(args.seed_max) if args.seed_max is not None else cfg_seed_max),
            "rng_seed": (args.rng_seed if args.rng_seed is not None else cfg_rng_seed),
            "seeds_if_explicit": cfg_seeds,
        },
        "effective_base_config": effective_cfg,
        "seeds": seeds,
        "n_tasks": len(cmds),
        "created_at": datetime.now().isoformat(),
    }
    with (run_dir / "seed_scan_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] tasks written: {tasks_path}")
    print(f"[OK] total tasks: {len(cmds)}")
    print(f"[OK] seeds: {seeds}")
    print(f"[OK] outputs root: {run_dir}")


if __name__ == "__main__":
    main()
