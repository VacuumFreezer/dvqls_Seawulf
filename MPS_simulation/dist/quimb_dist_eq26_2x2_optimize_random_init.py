#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quimb_dist_eq26_2x2_optimize import (
    Config,
    optimize,
    plot_history,
    resolve_output_paths,
    write_report,
)


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = THIS_DIR / "5qubits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 2x2 distributed Eq. (26) MPS optimizer with random angle "
            "initialization and sigma=lambda=1."
        )
    )
    parser.add_argument("--global-qubits", type=int, default=6)
    parser.add_argument("--local-qubits", type=int, default=5)
    parser.add_argument("--j-coupling", type=float, default=0.1)
    parser.add_argument("--kappa", type=float, default=20.0)
    parser.add_argument("--row-self-loop-weight", type=float, default=1.0)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--gate-max-bond", type=int, default=32)
    parser.add_argument("--gate-cutoff", type=float, default=1.0e-10)
    parser.add_argument("--apply-max-bond", type=int, default=64)
    parser.add_argument("--apply-cutoff", type=float, default=1.0e-10)
    parser.add_argument("--apply-no-compress", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--report-every", type=int, default=20)
    parser.add_argument("--init-seed", type=int, default=1234)
    parser.add_argument("--init-start", type=float, default=-3.141592653589793)
    parser.add_argument("--init-stop", type=float, default=3.141592653589793)
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(
            DEFAULT_OUT_DIR
            / "quimb_dist_eq26_2x2_optimize_random_init_n6_local5_k20_iter200.json"
        ),
    )
    parser.add_argument("--out-figure", type=str, default=None)
    parser.add_argument("--out-report", type=str, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        global_qubits=int(args.global_qubits),
        local_qubits=int(args.local_qubits),
        j_coupling=float(args.j_coupling),
        kappa=float(args.kappa),
        row_self_loop_weight=float(args.row_self_loop_weight),
        layers=int(args.layers),
        gate_max_bond=int(args.gate_max_bond),
        gate_cutoff=float(args.gate_cutoff),
        apply_max_bond=int(args.apply_max_bond),
        apply_cutoff=float(args.apply_cutoff),
        apply_no_compress=bool(args.apply_no_compress),
        learning_rate=float(args.learning_rate),
        adam_beta1=float(args.adam_beta1),
        adam_beta2=float(args.adam_beta2),
        adam_epsilon=float(args.adam_epsilon),
        iterations=int(args.iterations),
        report_every=int(args.report_every),
        init_mode="random_uniform",
        init_seed=int(args.init_seed),
        init_start=float(args.init_start),
        init_stop=float(args.init_stop),
        x_scale_init=1.0,
        z_scale_init=1.0,
        out_json=args.out_json,
        out_figure=args.out_figure,
        out_report=args.out_report,
    )


def main() -> None:
    cfg = build_config(parse_args())
    artifact_paths = resolve_output_paths(cfg)
    result = optimize(cfg)
    result["artifacts"] = {key: str(path.resolve()) for key, path in artifact_paths.items()}

    artifact_paths["json"].write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_history(result["history"], artifact_paths["figure"])
    write_report(artifact_paths["report"], result)

    print("\nFinal summary:")
    print(json.dumps(result["history"][-1], indent=2))
    print(f"\nWrote JSON to {artifact_paths['json']}")
    print(f"Wrote figure to {artifact_paths['figure']}")
    print(f"Wrote report to {artifact_paths['report']}")


if __name__ == "__main__":
    main()
