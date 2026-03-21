#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime
from pathlib import Path


PARTITIONS = [
    {
        "name": "8b8",
        "static_ops": "Partition_comparison_qjit/8b8/static_ops_cluster13_real_8x8.py",
        "mem": "128G",
        "nice": None,
    },
    {
        "name": "1b1",
        "static_ops": "Partition_comparison_qjit/1b1/static_ops_cluster13_real_1x1.py",
        "mem": "16G",
        "nice": 0,
    },
    {
        "name": "2b2",
        "static_ops": "Partition_comparison_qjit/2b2/static_ops_cluster13_real_2x2.py",
        "mem": "16G",
        "nice": 1000,
    },
    {
        "name": "4b4",
        "static_ops": "Partition_comparison_qjit/4b4/static_ops_cluster13_real_4x4.py",
        "mem": "128G",
        "nice": 2000,
    },
]


def build_task_command(
    *,
    python_bin: str,
    static_ops: str,
    out_dir: Path,
    seed: int,
    epochs: int,
    log_every: int,
    init_mode: str,
) -> str:
    cmd = [
        python_bin,
        "Partition_comparison_qjit/seawulf_partition_comparison_qjit.py",
        "--static_ops",
        static_ops,
        "--topology",
        "line",
        "--epochs",
        str(int(epochs)),
        "--seed",
        str(int(seed)),
        "--lr",
        "0.005",
        "--decay",
        "1.0",
        "--log_every",
        str(int(log_every)),
        "--ansatz",
        "brickwall_ry_cz",
        "--layers",
        "5",
        "--init_mode",
        str(init_mode),
        "--init_sigma_value",
        "1.0",
        "--init_sigma_noise_std",
        "0.0",
        "--init_lambda_value",
        "1.0",
        "--init_lambda_noise_std",
        "0.0",
        "--out",
        out_dir.as_posix(),
    ]
    return " ".join(shlex.quote(part) for part in cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="Five random seeds to use for each partition.",
    )
    ap.add_argument(
        "--python-bin",
        default="/gpfs/home/tonshen/.conda/envs/pennylane/bin/python",
    )
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--init-mode", default="uniform_0_pi")
    ap.add_argument("--stagger-seconds", type=int, default=20)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    run_root = root / "Partition_comparison_qjit" / "run"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_tag = args.tag.strip() or (
        f"jit5_{args.init_mode}_siglam1_lr5e3_nodecay_e{int(args.epochs)}_log{int(args.log_every)}_s5_{timestamp}"
    )
    suite_dir = run_root / suite_tag
    slurm_dir = suite_dir / "slurm"
    suite_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "suite_tag": suite_tag,
        "created_at": datetime.now().isoformat(),
        "python_bin": args.python_bin,
        "seeds": [int(seed) for seed in args.seeds],
        "hyperparameters": {
            "topology": "line",
            "epochs": int(args.epochs),
            "log_every": int(args.log_every),
            "lr": 0.005,
            "decay": 1.0,
            "ansatz": "brickwall_ry_cz",
            "layers": 5,
            "init_mode": str(args.init_mode),
            "init_sigma_value": 1.0,
            "init_sigma_noise_std": 0.0,
            "init_lambda_value": 1.0,
            "init_lambda_noise_std": 0.0,
            "workflow": "qjit(loss) + qjit(catalyst.grad)",
        },
        "partitions": [],
    }

    for partition in PARTITIONS:
        part_dir = suite_dir / partition["name"]
        part_dir.mkdir(parents=True, exist_ok=True)
        tasks_path = part_dir / "tasks.txt"
        lines = []
        outputs = []
        for seed in args.seeds:
            out_dir = part_dir / f"seed={int(seed)}"
            out_dir.mkdir(parents=True, exist_ok=True)
            lines.append(
                build_task_command(
                    python_bin=args.python_bin,
                    static_ops=partition["static_ops"],
                    out_dir=out_dir,
                    seed=int(seed),
                    epochs=int(args.epochs),
                    log_every=int(args.log_every),
                    init_mode=str(args.init_mode),
                )
            )
            outputs.append(out_dir.as_posix())
        tasks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest["partitions"].append(
            {
                "name": partition["name"],
                "static_ops": partition["static_ops"],
                "mem": partition["mem"],
                "nice": partition["nice"],
                "tasks_file": tasks_path.as_posix(),
                "output_dirs": outputs,
            }
        )

    manifest_path = suite_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    submit_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"SUITE_TAG={shlex.quote(suite_tag)}",
        "",
    ]

    for partition in PARTITIONS:
        sbatch_parts = [
            "sbatch",
            "--job-name",
            f"pc13_{partition['name']}",
            "--array",
            f"0-{len(args.seeds) - 1}",
            "--mem",
            partition["mem"],
            "--time",
            "24:00:00",
            "--output",
            f"Partition_comparison_qjit/run/{suite_tag}/slurm/%x_%A_%a.out",
            "--error",
            f"Partition_comparison_qjit/run/{suite_tag}/slurm/%x_%A_%a.err",
        ]
        if partition["nice"] is not None:
            sbatch_parts.append(f"--nice={int(partition['nice'])}")
        sbatch_parts.extend(
            [
                "--export",
                f"ALL,TAG={suite_tag}/{partition['name']}",
                "Partition_comparison_qjit/submit_array.slurm",
            ]
        )
        submit_lines.append(" ".join(shlex.quote(part) for part in sbatch_parts))
        if partition["name"] == "8b8" and int(args.stagger_seconds) > 0:
            submit_lines.append(f"sleep {int(args.stagger_seconds)}")

    submit_path = suite_dir / "submit_commands.sh"
    submit_path.write_text("\n".join(submit_lines) + "\n", encoding="utf-8")
    submit_path.chmod(0o755)

    print(f"[OK] suite_dir={suite_dir}")
    print(f"[OK] manifest={manifest_path}")
    print(f"[OK] submit_commands={submit_path}")
    for partition in PARTITIONS:
        print(f"[OK] tasks={suite_dir / partition['name'] / 'tasks.txt'}")


if __name__ == "__main__":
    main()
