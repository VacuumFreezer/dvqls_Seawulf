#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = THIS_DIR / "param.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate prepared Seawulf task files for the formal-vs-MPS 2x2 comparison."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--cases", type=str, default="")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--report-every", type=int, default=None)
    parser.add_argument("--init-seed", type=int, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Top-level YAML config must be a mapping.")
    return loaded


def resolve_case_order(config_data: dict[str, Any], requested: str) -> list[str]:
    available_cases = config_data.get("cases", {})
    if not isinstance(available_cases, dict) or not available_cases:
        raise ValueError("Config must define a non-empty `cases` mapping.")

    if requested.strip():
        case_order = [part.strip() for part in requested.split(",") if part.strip()]
    else:
        default_cases = config_data.get("job_prep", {}).get("default_cases", [])
        if default_cases:
            case_order = [str(case_name) for case_name in default_cases]
        else:
            case_order = list(available_cases.keys())

    missing = [case_name for case_name in case_order if case_name not in available_cases]
    if missing:
        raise KeyError(f"Unknown case names requested: {missing}")
    return case_order


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_data = load_config(config_path)
    case_order = resolve_case_order(config_data, args.cases)

    default_tag = str(config_data.get("job_prep", {}).get("default_tag", "")).strip()
    tag = args.tag.strip() or default_tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    control_dir = THIS_DIR / "runs" / tag
    slurm_dir = THIS_DIR / "slurm"
    control_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)

    tasks_path = control_dir / "tasks.txt"
    manifest_path = control_dir / "task_manifest.json"

    task_lines: list[str] = []
    manifest: list[dict[str, Any]] = []

    for case_name in case_order:
        out_dir = control_dir / case_name
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd_parts = [
            "python",
            "MPS_simulation/dist/10qubits/formal_compare_mps/run_case.py",
            "--config",
            config_path.as_posix(),
            "--case",
            case_name,
            "--out-dir",
            out_dir.resolve().as_posix(),
        ]
        if args.iterations is not None:
            cmd_parts.extend(["--iterations", str(args.iterations)])
        if args.report_every is not None:
            cmd_parts.extend(["--report-every", str(args.report_every)])
        if args.init_seed is not None:
            cmd_parts.extend(["--init-seed", str(args.init_seed)])

        command = " ".join(cmd_parts)
        task_lines.append(command)
        manifest.append(
            {
                "case": case_name,
                "init_seed": args.init_seed,
                "out_dir": str(out_dir.resolve()),
                "command": command,
            }
        )

    tasks_path.write_text("\n".join(task_lines) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] tasks written: {tasks_path}")
    print(f"[OK] manifest written: {manifest_path}")
    print(f"[OK] total tasks: {len(task_lines)}")
    print(f"[OK] control root: {control_dir}")
    print(f"[OK] slurm log dir: {slurm_dir}")
    print("[OK] prepared output roots:")
    for item in manifest:
        print(f"  - {item['out_dir']}")
    print("[OK] submit command template:")
    print(
        "  sbatch --array=0-"
        f"{len(task_lines) - 1} --export=TAG={tag} "
        "MPS_simulation/dist/10qubits/formal_compare_mps/submit_array.slurm"
    )


if __name__ == "__main__":
    main()
