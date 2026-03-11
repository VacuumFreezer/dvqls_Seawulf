#!/usr/bin/env python3
"""Generate Qiskit_simulation/run/<TAG>/tasks.txt for submit_array_qiskit_cpu.slurm."""

from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required: pip install pyyaml") from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a YAML mapping: {path}")
    return data


def _parse_csv_ints(value: str) -> List[int]:
    parts = [item.strip() for item in str(value).split(",")]
    return [int(item) for item in parts if item]


def _parse_csv_floats(value: str) -> List[float]:
    parts = [item.strip() for item in str(value).split(",")]
    return [float(item) for item in parts if item]


def _as_int_list(obj: Any) -> List[int]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return [int(item) for item in obj]
    return _parse_csv_ints(str(obj))


def _format_float(value: float) -> str:
    return f"{value:g}"


def _task_label(row: Dict[str, Any], counts: Dict[str, int]) -> str:
    parts: List[str] = []
    if counts["problems"] > 1:
        parts.append(f"prob={row['problem_name']}")
    if counts["cases"] > 1:
        parts.append(f"case={row['case_name']}")
    if counts["lrs"] > 1:
        parts.append(f"lr={row['lr_tag']}")
    if counts["spsa"] > 1:
        parts.append(f"spsa={row['spsa_tag']}")
    if counts["bond_dims"] > 1:
        parts.append(f"mbd={row['bond_tag']}")
    if counts["seeds"] > 1:
        parts.append(f"seed={row['seed']}")
    return "__".join(parts) if parts else "main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tasks.txt for Qiskit_simulation CPU Slurm arrays."
    )
    parser.add_argument(
        "--config",
        default="",
        help="YAML config path, e.g. Qiskit_simulation/params/qiskit_12q_stablizer.yaml",
    )
    parser.add_argument("--tag", default=None, help="Optional override for the run tag.")
    parser.add_argument("--output-root", default="Qiskit_simulation/run")
    parser.add_argument(
        "--entry-script",
        default="Qiskit_simulation/seawulf_cat_line_tracking_nodispatch_2x2_cluster12_stabilizer_qiskit.py",
    )
    parser.add_argument("--static-ops", default="")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--lrs", default="0.01")
    parser.add_argument("--spsa-c-values", default="0.05")
    parser.add_argument("--max-bond-dims", default="8")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _build_rows_from_cli(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = _parse_csv_ints(args.seeds)
    lrs = _parse_csv_floats(args.lrs)
    spsa_c_values = _parse_csv_floats(args.spsa_c_values)
    max_bond_dims = _parse_csv_ints(args.max_bond_dims)

    if not seeds:
        raise SystemExit("No seeds resolved from --seeds.")
    if not lrs:
        raise SystemExit("No learning rates resolved from --lrs.")
    if not spsa_c_values:
        raise SystemExit("No SPSA c values resolved from --spsa-c-values.")
    if not max_bond_dims:
        raise SystemExit("No bond dimensions resolved from --max-bond-dims.")

    rows = []
    for seed, lr, spsa_c, max_bond_dim in product(seeds, lrs, spsa_c_values, max_bond_dims):
        rows.append(
            {
                "problem_name": "default",
                "problem_module": args.static_ops.strip() or None,
                "case_name": "default",
                "script": args.entry_script,
                "lr_tag": f"lr{_format_float(lr)}",
                "lr": lr,
                "spsa_tag": f"spsa{_format_float(spsa_c)}",
                "spsa_c": spsa_c,
                "bond_tag": f"mbd{max_bond_dim}",
                "max_bond_dim": max_bond_dim,
                "seed": seed,
                "epochs": int(args.epochs),
                "log_every": int(args.log_every),
                "num_threads": int(args.num_threads),
                "extra_args": shlex.split(args.extra_args),
            }
        )

    return {
        "tag": args.tag,
        "output_root": args.output_root,
        "python_bin": args.python_bin,
        "rows": rows,
    }


def _build_rows_from_config(cfg: dict, tag_override: str | None) -> Dict[str, Any]:
    run_root = str(cfg.get("run_root", "Qiskit_simulation/run"))
    tag = tag_override or str(cfg.get("tag", "")).strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    python_bin = str(cfg.get("python_bin", "python"))

    problems = cfg.get("problems", [])
    cases = cfg.get("cases", [])
    lr_sets = cfg.get("lr_sets", [])
    spsa_c_sets = cfg.get("spsa_c_sets", [])
    bond_dim_sets = cfg.get("max_bond_dim_sets", [])
    seeds = _as_int_list(cfg.get("seeds", []))
    common = cfg.get("common", {}) if isinstance(cfg.get("common", {}), dict) else {}

    if not isinstance(problems, list) or not problems:
        raise SystemExit("`problems` must be a non-empty list in the Qiskit YAML config.")
    if not isinstance(cases, list) or not cases:
        raise SystemExit("`cases` must be a non-empty list in the Qiskit YAML config.")
    if not isinstance(lr_sets, list) or not lr_sets:
        raise SystemExit("`lr_sets` must be a non-empty list in the Qiskit YAML config.")
    if not isinstance(spsa_c_sets, list) or not spsa_c_sets:
        raise SystemExit("`spsa_c_sets` must be a non-empty list in the Qiskit YAML config.")
    if not isinstance(bond_dim_sets, list) or not bond_dim_sets:
        raise SystemExit("`max_bond_dim_sets` must be a non-empty list in the Qiskit YAML config.")
    if not seeds:
        raise SystemExit("`seeds` must resolve to at least one value in the Qiskit YAML config.")

    epochs = int(common.get("epochs", 5000))
    log_every = int(common.get("log_every", 50))
    num_threads = int(common.get("num_threads", 1))
    extra_args = shlex.split(str(common.get("extra_args", "")))

    rows = []
    for problem, case, lr_item, spsa_item, bond_item, seed in product(
        problems, cases, lr_sets, spsa_c_sets, bond_dim_sets, seeds
    ):
        if not isinstance(problem, dict) or "module" not in problem:
            raise SystemExit("Each `problems` entry must be a mapping with at least `module`.")
        if not isinstance(case, dict) or "script" not in case:
            raise SystemExit("Each `cases` entry must be a mapping with at least `script`.")
        if not isinstance(lr_item, dict) or "lr" not in lr_item or "tag" not in lr_item:
            raise SystemExit("Each `lr_sets` entry must contain `tag` and `lr`.")
        if not isinstance(spsa_item, dict) or "spsa_c" not in spsa_item or "tag" not in spsa_item:
            raise SystemExit("Each `spsa_c_sets` entry must contain `tag` and `spsa_c`.")
        if not isinstance(bond_item, dict) or "max_bond_dim" not in bond_item or "tag" not in bond_item:
            raise SystemExit("Each `max_bond_dim_sets` entry must contain `tag` and `max_bond_dim`.")

        row_extra = list(extra_args)
        if "extra_args" in case:
            row_extra.extend(shlex.split(str(case["extra_args"])))
        if "extra_args" in problem:
            row_extra.extend(shlex.split(str(problem["extra_args"])))

        rows.append(
            {
                "problem_name": str(problem.get("name", problem["module"])),
                "problem_module": str(problem["module"]),
                "case_name": str(case.get("name", case["script"])),
                "script": str(case["script"]),
                "lr_tag": str(lr_item["tag"]),
                "lr": float(lr_item["lr"]),
                "spsa_tag": str(spsa_item["tag"]),
                "spsa_c": float(spsa_item["spsa_c"]),
                "bond_tag": str(bond_item["tag"]),
                "max_bond_dim": int(bond_item["max_bond_dim"]),
                "seed": int(seed),
                "epochs": epochs,
                "log_every": log_every,
                "num_threads": num_threads,
                "extra_args": row_extra,
            }
        )

    return {
        "tag": tag,
        "output_root": run_root,
        "python_bin": python_bin,
        "rows": rows,
    }


def main() -> int:
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    plan = (
        _build_rows_from_config(load_yaml((root / args.config).resolve()), args.tag)
        if args.config
        else _build_rows_from_cli(args)
    )

    tag = str(plan["tag"] or datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_root = (root / str(plan["output_root"])).resolve()
    python_bin = str(plan["python_bin"])
    rows: List[Dict[str, Any]] = list(plan["rows"])

    run_dir = output_root / tag
    tasks_path = run_dir / "tasks.txt"
    config_path = run_dir / "suite_config.json"

    if run_dir.exists() and not args.overwrite:
        raise SystemExit(f"Run directory already exists: {run_dir}. Use --overwrite to reuse it.")

    run_dir.mkdir(parents=True, exist_ok=True)

    counts = {
        "problems": len({row["problem_name"] for row in rows}),
        "cases": len({row["case_name"] for row in rows}),
        "lrs": len({row["lr_tag"] for row in rows}),
        "spsa": len({row["spsa_tag"] for row in rows}),
        "bond_dims": len({row["bond_tag"] for row in rows}),
        "seeds": len({row["seed"] for row in rows}),
    }

    commands: List[str] = []
    materialized_rows: List[Dict[str, Any]] = []

    for row in rows:
        label = _task_label(row, counts)
        out_dir = run_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            python_bin,
            row["script"],
            "--static_ops",
            row["problem_module"],
            "--out",
            out_dir.as_posix(),
            "--epochs",
            str(row["epochs"]),
            "--log_every",
            str(row["log_every"]),
            "--lr",
            _format_float(float(row["lr"])),
            "--spsa_c",
            _format_float(float(row["spsa_c"])),
            "--max_bond_dim",
            str(int(row["max_bond_dim"])),
            "--seed",
            str(int(row["seed"])),
            "--num_threads",
            str(int(row["num_threads"])),
        ]
        cmd.extend(row.get("extra_args", []))

        commands.append(shlex.join(cmd))
        materialized_rows.append(
            {
                "label": label,
                "problem_name": row["problem_name"],
                "problem_module": row["problem_module"],
                "case_name": row["case_name"],
                "script": row["script"],
                "seed": int(row["seed"]),
                "lr_tag": row["lr_tag"],
                "lr": float(row["lr"]),
                "spsa_tag": row["spsa_tag"],
                "spsa_c": float(row["spsa_c"]),
                "bond_tag": row["bond_tag"],
                "max_bond_dim": int(row["max_bond_dim"]),
                "epochs": int(row["epochs"]),
                "log_every": int(row["log_every"]),
                "num_threads": int(row["num_threads"]),
                "out_dir": out_dir.as_posix(),
            }
        )

    tasks_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "tag": tag,
                "run_dir": run_dir.as_posix(),
                "tasks_file": tasks_path.as_posix(),
                "python_bin": python_bin,
                "task_count": len(materialized_rows),
                "rows": materialized_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"TAG={tag}")
    print(f"RUN_DIR={run_dir}")
    print(f"TASKS_FILE={tasks_path}")
    print(f"TASK_COUNT={len(materialized_rows)}")
    if materialized_rows:
        print(f"ARRAY_RANGE=0-{len(materialized_rows) - 1}")
    else:
        print("ARRAY_RANGE=empty")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
