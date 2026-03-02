#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


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


def dump_yaml(path: Path, data: dict) -> None:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("No PyYAML: pip install pyyaml") from e
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def as_seed_list(obj: Any) -> List[int]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return [int(x) for x in obj]
    if isinstance(obj, str):
        out = []
        for x in obj.split(","):
            x = x.strip()
            if x:
                out.append(int(x))
        return out
    raise SystemExit(f"Invalid seeds format: {type(obj)}")


def require(mapping: dict, key: str, where: str):
    if key not in mapping:
        raise SystemExit(f"Missing required key `{key}` in {where}")
    return mapping[key]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a single tasks.txt that scans distributed 4x4, distributed 2x2, and centralized VQLS."
    )
    ap.add_argument("--config", required=True, help="Path to unified sweep YAML, e.g. params/sweep_3algorithms.yaml")
    ap.add_argument("--tag", default="", help="Optional override for run tag.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    cfg_path = (root / args.config).resolve()
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    cfg = load_yaml(cfg_path)

    run_root = root / str(cfg.get("run_root", "run"))
    tag = args.tag.strip() or str(cfg.get("tag", "")).strip() or datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = run_root / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = run_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = run_dir / "tasks.txt"

    global_common = cfg.get("common", {}) if isinstance(cfg.get("common", {}), dict) else {}
    global_seeds = as_seed_list(cfg.get("seeds", [0]))
    if not global_seeds:
        raise SystemExit("No seeds provided. Set top-level `seeds` or per-algorithm `seeds`.")

    algos = cfg.get("algorithms", [])
    if not isinstance(algos, list) or not algos:
        raise SystemExit("`algorithms` must be a non-empty list in unified config.")

    cmds: List[str] = []
    meta_algos: List[Dict[str, Any]] = []

    for algo in algos:
        if not isinstance(algo, dict):
            raise SystemExit("Each algorithms entry must be a mapping.")

        name = str(require(algo, "name", "algorithm entry"))
        kind = str(require(algo, "kind", f"algorithm `{name}`")).strip().lower()
        seeds = as_seed_list(algo.get("seeds", global_seeds))
        if not seeds:
            raise SystemExit(f"No seeds resolved for algorithm `{name}`")

        if kind == "distributed":
            script = str(require(algo, "script", f"algorithm `{name}`"))
            static_ops = str(require(algo, "static_ops", f"algorithm `{name}`"))
            system_variant = algo.get("system_variant", None)
            if system_variant is not None:
                system_variant = str(system_variant)
            problem_name = str(algo.get("problem_name", system_variant or "default"))

            topologies = algo.get("topologies", ["line"])
            if not isinstance(topologies, list) or not topologies:
                raise SystemExit(f"`topologies` must be a non-empty list for distributed algorithm `{name}`")
            pass_topology = bool(algo.get("pass_topology", True))

            lr_sets = algo.get("lr_sets", [])
            decay_sets = algo.get("decay_sets", [])
            if not isinstance(lr_sets, list) or not lr_sets:
                raise SystemExit(f"`lr_sets` must be a non-empty list for distributed algorithm `{name}`")
            if not isinstance(decay_sets, list) or not decay_sets:
                raise SystemExit(f"`decay_sets` must be a non-empty list for distributed algorithm `{name}`")

            epochs = int(algo.get("epochs", global_common.get("epochs", 30000)))
            log_every = int(algo.get("log_every", global_common.get("log_every", 20)))

            for topo in topologies:
                topo = str(topo)
                for seed in seeds:
                    for lr in lr_sets:
                        if not isinstance(lr, dict):
                            raise SystemExit(f"Each lr_sets entry must be a mapping in algorithm `{name}`")
                        lr_tag = str(require(lr, "tag", f"algorithm `{name}` lr entry"))
                        lr_value = float(require(lr, "lr", f"algorithm `{name}` lr entry"))

                        for decay in decay_sets:
                            if not isinstance(decay, dict):
                                raise SystemExit(f"Each decay_sets entry must be a mapping in algorithm `{name}`")
                            decay_tag = str(require(decay, "tag", f"algorithm `{name}` decay entry"))
                            decay_value = float(require(decay, "decay", f"algorithm `{name}` decay entry"))

                            out_dir = run_dir / (
                                f"algo={name}__prob={problem_name}__graph={topo}"
                                f"__decay={decay_tag}__lr={lr_tag}__seed={int(seed)}"
                            )
                            out_dir.mkdir(parents=True, exist_ok=True)

                            cmd_parts = [
                                f"python {shlex.quote(script)}",
                                f"--static_ops {shlex.quote(static_ops)}",
                            ]
                            if system_variant:
                                cmd_parts.append(f"--system_variant {shlex.quote(system_variant)}")
                            if pass_topology:
                                cmd_parts.append(f"--topology {shlex.quote(topo)}")
                            cmd_parts.extend(
                                [
                                    f"--epochs {epochs}",
                                    f"--seed {int(seed)}",
                                    f"--lr {lr_value}",
                                    f"--decay {decay_value}",
                                    f"--log_every {log_every}",
                                    f"--out {shlex.quote(out_dir.as_posix())}",
                                ]
                            )
                            cmd = " ".join(cmd_parts)
                            cmds.append(cmd)

        elif kind in ("centralized", "centralized_qjit"):
            script = str(require(algo, "script", f"algorithm `{name}`"))
            base_config_rel = str(algo.get("base_config", "cen_vqls/config.yaml"))
            base_config_path = (root / base_config_rel).resolve()
            if not base_config_path.exists():
                raise SystemExit(f"Base config not found for `{name}`: {base_config_path}")

            base_cfg = load_yaml(base_config_path)
            overrides = algo.get("config_overrides", {})
            if overrides and not isinstance(overrides, dict):
                raise SystemExit(f"config_overrides must be a mapping for algorithm `{name}`")

            merged_cfg = copy.deepcopy(base_cfg)
            deep_update(merged_cfg, copy.deepcopy(overrides))

            report_tag_base = str(
                algo.get(
                    "report_tag",
                    merged_cfg.get("report", {}).get("tag", f"centralized_{name}"),
                )
            )

            for seed in seeds:
                per_seed_cfg = copy.deepcopy(merged_cfg)
                per_seed_cfg.setdefault("runtime", {})
                per_seed_cfg["runtime"]["seed"] = int(seed)

                report_out = run_dir / "reports" / name / f"seed={int(seed)}"
                report_out.mkdir(parents=True, exist_ok=True)

                per_seed_cfg.setdefault("report", {})
                per_seed_cfg["report"]["out_dir"] = report_out.as_posix()
                per_seed_cfg["report"]["tag"] = f"{report_tag_base}__seed={int(seed)}"

                per_seed_cfg_path = configs_dir / f"config__algo={name}__seed={int(seed)}.yaml"
                dump_yaml(per_seed_cfg_path, per_seed_cfg)

                cmd = (
                    f"python {shlex.quote(script)} "
                    f"--config {shlex.quote(per_seed_cfg_path.as_posix())}"
                )
                cmds.append(cmd)

        else:
            raise SystemExit(
                f"Unknown algorithm kind `{kind}` for `{name}`. "
                f"Use `distributed` or `centralized_qjit`."
            )

        meta_algos.append({"name": name, "kind": kind, "num_seeds": len(seeds)})

    with tasks_path.open("w", encoding="utf-8") as f:
        for cmd in cmds:
            f.write(cmd + "\n")

    meta = {
        "config": cfg_path.as_posix(),
        "tag": tag,
        "run_dir": run_dir.as_posix(),
        "num_tasks": len(cmds),
        "algorithms": meta_algos,
    }
    with (run_dir / "taskgen_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] tasks written: {tasks_path}")
    print(f"[OK] total tasks: {len(cmds)}")
    print(f"[OK] outputs root: {run_dir}")


if __name__ == "__main__":
    main()
