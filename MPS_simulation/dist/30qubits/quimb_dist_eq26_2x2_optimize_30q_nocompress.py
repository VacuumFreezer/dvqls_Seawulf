#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[2]
DIST_DIR = ROOT_DIR / "dist"
if str(ROOT_DIR.parents[0]) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR.parents[0]))
if str(DIST_DIR) not in sys.path:
    sys.path.insert(0, str(DIST_DIR))


def _load_smoke30_module():
    mod_path = THIS_DIR / "quimb_dist_eq26_2x2_direct_mpo_smoke_30q.py"
    spec = importlib.util.spec_from_file_location("smoke30", mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SMOKE30 = _load_smoke30_module()

from MPS_simulation.dist.quimb_dist_eq26_2x2_benchmark import (  # noqa: E402
    distributed_iteration,
    global_cost_jax,
    initialize_state,
    make_initial_parameters,
    to_jax_problem,
)


@dataclass
class Config:
    global_qubits: int
    local_qubits: int
    j_coupling: float
    kappa: float
    row_self_loop_weight: float
    layers: int
    gate_max_bond: int
    gate_cutoff: float
    apply_max_bond: int
    apply_cutoff: float
    apply_no_compress: bool
    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    iterations: int
    report_every: int
    init_mode: str
    init_seed: int
    init_start: float
    init_stop: float
    x_scale_init: float
    z_scale_init: float
    out_json: str | None
    out_figure: str | None
    out_report: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 30-qubit 2x2 distributed Eq. (26) optimizer using the "
            "low-memory direct-MPO workflow and no compression in MPO-on-MPS "
            "application."
        )
    )
    parser.add_argument("--global-qubits", type=int, default=30)
    parser.add_argument("--local-qubits", type=int, default=29)
    parser.add_argument("--j-coupling", type=float, default=0.1)
    parser.add_argument("--kappa", type=float, default=20.0)
    parser.add_argument("--row-self-loop-weight", type=float, default=1.0)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--gate-max-bond", type=int, default=32)
    parser.add_argument("--gate-cutoff", type=float, default=1.0e-10)
    parser.add_argument("--apply-max-bond", type=int, default=64)
    parser.add_argument("--apply-cutoff", type=float, default=1.0e-10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument(
        "--init-mode",
        type=str,
        choices=("structured_linspace", "random_uniform"),
        default="structured_linspace",
    )
    parser.add_argument("--init-seed", type=int, default=1234)
    parser.add_argument("--init-start", type=float, default=0.01)
    parser.add_argument("--init-stop", type=float, default=0.2)
    parser.add_argument("--x-scale-init", type=float, default=0.75)
    parser.add_argument("--z-scale-init", type=float, default=0.10)
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(
            THIS_DIR / "quimb_dist_eq26_2x2_optimize_30q_nocompress_iter200.json"
        ),
    )
    parser.add_argument("--out-figure", type=str, default=None)
    parser.add_argument("--out-report", type=str, default=None)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> Config:
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
        apply_no_compress=True,
        learning_rate=float(args.learning_rate),
        adam_beta1=float(args.adam_beta1),
        adam_beta2=float(args.adam_beta2),
        adam_epsilon=float(args.adam_epsilon),
        iterations=int(args.iterations),
        report_every=int(args.report_every),
        init_mode=str(args.init_mode),
        init_seed=int(args.init_seed),
        init_start=float(args.init_start),
        init_stop=float(args.init_stop),
        x_scale_init=float(args.x_scale_init),
        z_scale_init=float(args.z_scale_init),
        out_json=args.out_json,
        out_figure=args.out_figure,
        out_report=args.out_report,
    )


def resolve_output_paths(cfg: Config) -> dict[str, Path]:
    base = (
        Path(cfg.out_json).with_suffix("")
        if cfg.out_json is not None
        else THIS_DIR / "quimb_dist_eq26_2x2_optimize_30q_nocompress_iter200"
    )
    json_path = Path(cfg.out_json) if cfg.out_json is not None else base.with_suffix(".json")
    figure_path = (
        Path(cfg.out_figure)
        if cfg.out_figure is not None
        else base.with_name(base.name + "_cost").with_suffix(".png")
    )
    report_path = (
        Path(cfg.out_report)
        if cfg.out_report is not None
        else base.with_name(base.name + "_report").with_suffix(".md")
    )
    for path in (json_path, figure_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    return {"json": json_path, "figure": figure_path, "report": report_path}


def encode_array(array: np.ndarray) -> dict[str, object]:
    array_np = np.asarray(array)
    if np.iscomplexobj(array_np):
        return {
            "real": np.asarray(array_np.real, dtype=np.float64).tolist(),
            "imag": np.asarray(array_np.imag, dtype=np.float64).tolist(),
        }
    return {"real": np.asarray(array_np, dtype=np.float64).tolist()}


def plot_cost_history(history: list[dict[str, float | int]], figure_path: Path) -> None:
    iterations = [int(item["iteration"]) for item in history]
    costs = np.maximum([float(item["global_cost"]) for item in history], 1.0e-16)

    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=160)
    ax.plot(iterations, costs, linewidth=1.8, color="#005f73")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Global cost")
    ax.set_title("Distributed Optimization")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def write_report(report_path: Path, result: dict[str, object]) -> None:
    history = result["history"]
    final = history[-1]
    best_cost = min(float(item["global_cost"]) for item in history)
    lines = [
        "# Distributed Optimization Report",
        "",
        "## Setup",
        f"- Global system: `{result['problem']['global_qubits']}` qubits.",
        f"- Local block size: `{result['problem']['local_qubits']}` qubits.",
        f"- Iterations: `{result['optimization']['iterations']}`.",
        f"- Learning rate: `{result['config']['learning_rate']}`.",
        f"- Initialization mode: `{result['config']['init_mode']}`.",
        f"- No-compression MPO apply: `{result['config']['apply_no_compress']}`.",
        f"- Row Laplacian: `{result['problem']['row_laplacian']}`.",
        f"- `eta = {result['problem']['eta']:.12g}`.",
        f"- `zeta = {result['problem']['zeta']:.12g}`.",
        "",
        "## Outcome",
        f"- Final global cost: `{final['global_cost']:.12g}`.",
        f"- Best global cost in run: `{best_cost:.12g}`.",
        f"- Final alpha gradient L2: `{final['alpha_grad_l2']:.12g}`.",
        f"- Final beta gradient L2: `{final['beta_grad_l2']:.12g}`.",
        f"- Final alpha step L2: `{final['alpha_step_l2']:.12g}`.",
        f"- Final beta step L2: `{final['beta_step_l2']:.12g}`.",
        f"- Total elapsed time: `{result['optimization']['elapsed_s']:.6f} s`.",
        f"- Mean time per iteration: `{result['optimization']['elapsed_s'] / max(result['optimization']['iterations'], 1):.6f} s`.",
        "",
        "## Notes",
        "- This low-memory 30-qubit run does not build the dense global matrix.",
        "- The optimization uses direct MPO block construction and `compress=False` in `A_ij |x_ij>`.",
        "- No dense residual or exact `L2` error is reported in this workflow.",
        "",
        "## Artifacts",
        f"- JSON: `{result['artifacts']['json']}`",
        f"- Figure: `{result['artifacts']['figure']}`",
        f"- Report: `{result['artifacts']['report']}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_progress_artifacts(
    paths: dict[str, Path],
    cfg: Config,
    problem_np: dict[str, object],
    history: list[dict[str, float | int]],
    state,
    start_time: float,
) -> None:
    alpha_np = np.asarray(state["alpha"])
    beta_np = np.asarray(state["beta"])
    result = {
        "config": asdict(cfg),
        "problem": {
            "global_qubits": cfg.global_qubits,
            "local_qubits": cfg.local_qubits,
            "row_laplacian": np.asarray(problem_np["row_laplacian"], dtype=np.float64).tolist(),
            "eta": float(problem_np["eta"]),
            "zeta": float(problem_np["zeta"]),
            "column_mix": np.asarray(problem_np["column_mix"], dtype=np.float64).tolist(),
            "block_formula": problem_np["block_formula"],
        },
        "optimization": {
            "iterations": int(history[-1]["iteration"]) if history else 0,
            "elapsed_s": time.perf_counter() - start_time,
        },
        "history": history,
        "final_state": {
            "alpha": encode_array(alpha_np),
            "beta": encode_array(beta_np),
        },
        "artifacts": {key: str(path.resolve()) for key, path in paths.items()},
    }
    paths["json"].write_text(json.dumps(result, indent=2), encoding="utf-8")
    if history:
        plot_cost_history(history, paths["figure"])
        write_report(paths["report"], result)


def main() -> None:
    import jax

    jax.config.update("jax_enable_x64", True)

    cfg = make_config(parse_args())
    paths = resolve_output_paths(cfg)

    problem_np = SMOKE30.build_direct_problem(cfg)
    problem_jax = to_jax_problem(problem_np)
    alpha_init_np, beta_init_np = make_initial_parameters(cfg)

    cost_fn = lambda a, b: global_cost_jax(a, b, cfg, problem_jax)
    full_grad_fn = jax.value_and_grad(cost_fn, argnums=(0, 1))
    alpha_grad_fn = jax.grad(cost_fn, argnums=0)

    start_time = time.perf_counter()
    state = initialize_state(cfg, alpha_grad_fn, alpha_init_np, beta_init_np)
    history: list[dict[str, float | int]] = []

    initial_cost = float(cost_fn(state["alpha"], state["beta"]))
    print(f"Initial cost: {initial_cost:.12f}", flush=True)

    for iteration in range(1, cfg.iterations + 1):
        state, diag = distributed_iteration(
            state=state,
            cfg=cfg,
            problem_jax=problem_jax,
            full_grad_fn=full_grad_fn,
            alpha_grad_fn=alpha_grad_fn,
        )

        entry = {
            "iteration": iteration,
            "global_cost": float(diag["current_cost"]),
            "alpha_grad_l2": float(diag["alpha_grad_l2"]),
            "beta_grad_l2": float(diag["beta_grad_l2"]),
            "alpha_step_l2": float(diag["alpha_step_l2"]),
            "beta_step_l2": float(diag["beta_step_l2"]),
            "elapsed_s": time.perf_counter() - start_time,
        }
        history.append(entry)

        if (iteration % cfg.report_every == 0) or (iteration == cfg.iterations):
            print(
                f"[iter {iteration:4d}] cost={entry['global_cost']:.12f} "
                f"alpha_grad={entry['alpha_grad_l2']:.6f} "
                f"beta_grad={entry['beta_grad_l2']:.6f} "
                f"elapsed={entry['elapsed_s']:.2f}s",
                flush=True,
            )
            write_progress_artifacts(paths, cfg, problem_np, history, state, start_time)

    print(f"Wrote JSON to {paths['json']}", flush=True)
    print(f"Wrote figure to {paths['figure']}", flush=True)
    print(f"Wrote report to {paths['report']}", flush=True)


if __name__ == "__main__":
    main()
