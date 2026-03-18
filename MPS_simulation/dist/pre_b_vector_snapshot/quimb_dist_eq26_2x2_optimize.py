#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from scipy.sparse.linalg import spsolve

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from quimb_dist_eq26_2x2_benchmark import (  # noqa: E402
    build_circuit_numpy,
    build_partitioned_problem_numpy,
    distributed_iteration,
    global_cost_jax,
    global_cost_numpy,
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
    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    iterations: int
    report_every: int
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
            "Run 200 iterations of the 2x2 distributed Eq. (26) MPS optimizer "
            "and record cost, residual, consensus error, and relative solution error."
        )
    )
    parser.add_argument("--global-qubits", type=int, default=11)
    parser.add_argument("--local-qubits", type=int, default=10)
    parser.add_argument("--j-coupling", type=float, default=0.1)
    parser.add_argument("--kappa", type=float, default=20.0)
    parser.add_argument("--row-self-loop-weight", type=float, default=1.0)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--gate-max-bond", type=int, default=32)
    parser.add_argument("--gate-cutoff", type=float, default=1.0e-10)
    parser.add_argument("--apply-max-bond", type=int, default=64)
    parser.add_argument("--apply-cutoff", type=float, default=1.0e-10)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--report-every", type=int, default=20)
    parser.add_argument("--init-start", type=float, default=0.01)
    parser.add_argument("--init-stop", type=float, default=0.2)
    parser.add_argument("--x-scale-init", type=float, default=0.75)
    parser.add_argument("--z-scale-init", type=float, default=0.10)
    parser.add_argument("--out-json", type=str, default=None)
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
        learning_rate=float(args.learning_rate),
        adam_beta1=float(args.adam_beta1),
        adam_beta2=float(args.adam_beta2),
        adam_epsilon=float(args.adam_epsilon),
        iterations=int(args.iterations),
        report_every=int(args.report_every),
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
        else THIS_DIR
        / f"quimb_dist_eq26_2x2_optimize_n{cfg.global_qubits}_local{cfg.local_qubits}_iter{cfg.iterations}"
    )
    json_path = Path(cfg.out_json) if cfg.out_json is not None else base.with_suffix(".json")
    figure_path = (
        Path(cfg.out_figure)
        if cfg.out_figure is not None
        else base.with_name(base.name + "_history").with_suffix(".png")
    )
    report_path = (
        Path(cfg.out_report)
        if cfg.out_report is not None
        else base.with_name(base.name + "_report").with_suffix(".md")
    )
    for path in (json_path, figure_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    return {"json": json_path, "figure": figure_path, "report": report_path}


def build_global_sparse_problem(cfg: Config, eta: float, zeta: float):
    import quimb as qu
    import quimb.tensor as qtn

    builder = qtn.SpinHam1D(cyclic=False)
    builder += 1.0 / zeta, qu.pauli("X")
    builder += cfg.j_coupling / zeta, qu.pauli("Z"), qu.pauli("Z")
    builder += eta / zeta, qu.eye(2)
    return builder.build_sparse(cfg.global_qubits).tocsr()


def exact_solution(a_sparse, n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    dim = 2**n_qubits
    b_dense = np.full(dim, 1.0 / math.sqrt(dim), dtype=np.complex128)
    x_true = spsolve(a_sparse, b_dense)
    x_true = np.asarray(x_true, dtype=np.complex128)
    return x_true, b_dense


def agent_block_vector(alpha_agent: np.ndarray, cfg: Config) -> np.ndarray:
    sigma = float(alpha_agent[0])
    state = build_circuit_numpy(cfg.local_qubits, alpha_agent[1:], cfg).psi
    vector = np.asarray(state.to_dense(), dtype=np.complex128).reshape(-1)
    return sigma * vector


def reconstruct_row_solutions(alpha: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    row_vectors = []
    for i in range(2):
        blocks = [agent_block_vector(alpha[i, j], cfg) for j in range(2)]
        row_vectors.append(np.concatenate(blocks))
    return row_vectors[0], row_vectors[1]


def encode_array(array: np.ndarray) -> dict[str, object]:
    array_np = np.asarray(array)
    if np.iscomplexobj(array_np):
        return {
            "real": np.asarray(array_np.real, dtype=np.float64).tolist(),
            "imag": np.asarray(array_np.imag, dtype=np.float64).tolist(),
        }
    return {"real": np.asarray(array_np, dtype=np.float64).tolist()}


def decode_array(payload: object) -> np.ndarray:
    if isinstance(payload, dict) and "real" in payload:
        real = np.asarray(payload["real"])
        imag = np.asarray(payload.get("imag", 0.0))
        if np.any(imag):
            return real + 1.0j * imag
        return real
    return np.asarray(payload)


def format_array(array: np.ndarray) -> str:
    array_np = np.real_if_close(np.asarray(array), tol=1000)
    formatter = None
    if np.iscomplexobj(array_np):
        formatter = {
            "complex_kind": lambda z: f"{complex(z).real:.12g}{complex(z).imag:+.12g}j"
        }
    elif np.issubdtype(array_np.dtype, np.floating):
        formatter = {"float_kind": lambda x: f"{float(x):.12g}"}
    return np.array2string(
        array_np,
        separator=", ",
        max_line_width=160,
        precision=12,
        suppress_small=False,
        threshold=array_np.size,
        formatter=formatter,
    )


def compute_metrics(
    alpha: np.ndarray,
    beta: np.ndarray,
    cfg: Config,
    problem_np: dict[str, object],
    a_sparse,
    b_dense: np.ndarray,
    x_true: np.ndarray,
) -> dict[str, float]:
    row1, row2 = reconstruct_row_solutions(alpha, cfg)
    x_avg = 0.5 * (row1 + row2)
    residual = a_sparse @ x_avg - b_dense
    consensus_gap = row1 - row2

    return {
        "global_cost": float(global_cost_numpy(alpha, beta, cfg, problem_np)),
        "residual_norm_l2": float(np.linalg.norm(residual)),
        "consensus_error_l2": float(np.linalg.norm(consensus_gap)),
        "row_variance_mse": float(np.mean(np.abs(consensus_gap) ** 2)),
        "relative_solution_error_l2": float(
            np.linalg.norm(x_avg - x_true) / max(np.linalg.norm(x_true), 1.0e-12)
        ),
    }


def compute_rescaling_diagnostics(
    x_reconstructed: np.ndarray,
    x_true: np.ndarray,
    a_dense: np.ndarray,
    b_dense: np.ndarray,
) -> dict[str, float]:
    denom = np.vdot(x_reconstructed, x_reconstructed)
    if abs(denom) <= 1.0e-15:
        best_scale = 0.0
    else:
        best_scale = np.vdot(x_reconstructed, x_true) / denom

    x_rescaled = best_scale * x_reconstructed
    return {
        "best_scale_to_true_real": float(np.real(best_scale)),
        "best_scale_to_true_imag": float(np.imag(best_scale)),
        "raw_x_norm_l2": float(np.linalg.norm(x_reconstructed)),
        "true_x_norm_l2": float(np.linalg.norm(x_true)),
        "rescaled_x_norm_l2": float(np.linalg.norm(x_rescaled)),
        "cosine_similarity_to_true": float(
            abs(np.vdot(x_reconstructed, x_true))
            / max(np.linalg.norm(x_reconstructed) * np.linalg.norm(x_true), 1.0e-15)
        ),
        "rescaled_relative_solution_error_l2": float(
            np.linalg.norm(x_rescaled - x_true) / max(np.linalg.norm(x_true), 1.0e-12)
        ),
        "rescaled_residual_norm_l2": float(np.linalg.norm(a_dense @ x_rescaled - b_dense)),
    }


def plot_history(history: list[dict[str, float | int]], figure_path: Path) -> None:
    iterations = [int(item["iteration"]) for item in history]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), dpi=160)
    series = [
        ("global_cost", "Global cost"),
        ("residual_norm_l2", "Residual norm ||Ax-b||_2"),
        ("consensus_error_l2", "Consensus error ||x_row1-x_row2||_2"),
        ("relative_solution_error_l2", "Relative solution error"),
    ]

    for ax, (key, title) in zip(axes.flat, series):
        values = np.maximum([float(item[key]) for item in history], 1.0e-16)
        ax.plot(iterations, values, linewidth=1.8, color="#005f73")
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(title)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25, linewidth=0.8)

    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def write_report(report_path: Path, result: dict[str, object]) -> None:
    final = result["history"][-1]
    alpha_final = decode_array(result["final_state"]["alpha"])
    beta_final = decode_array(result["final_state"]["beta"])
    x_reconstructed = decode_array(result["final_state"]["x_reconstructed"])
    x_true = decode_array(result["final_state"]["x_true"])
    a_dense = decode_array(result["linear_system"]["A_dense"])
    b_dense = decode_array(result["linear_system"]["b"])
    rescaled = result["rescaled_diagnostics"]
    lines = [
        "# Distributed Eq. (26) 2x2 Optimization Report",
        "",
        "## Setup",
        f"- Iterations: `{result['optimization']['iterations']}`",
        f"- Learning rate: `{result['config']['learning_rate']}`",
        f"- Row Laplacian: `{result['problem']['row_laplacian']}`",
        f"- Scaled spectrum check: `lambda_min={result['problem']['scaled_lambda_min']:.12g}`, `lambda_max={result['problem']['scaled_lambda_max']:.12g}`",
        "- Figure y-axes use log scale.",
        "",
        "## Final Metrics",
        f"- Global cost: `{final['global_cost']:.12g}`",
        f"- Residual norm: `{final['residual_norm_l2']:.12g}`",
        f"- Consensus error: `{final['consensus_error_l2']:.12g}`",
        f"- Relative solution error: `{final['relative_solution_error_l2']:.12g}`",
        f"- Elapsed time: `{result['optimization']['elapsed_s']:.6f} s`",
        "",
        "## Rescaled-x Diagnostic",
        f"- Best scalar to match true `x`: `{rescaled['best_scale_to_true_real']:.12g}{rescaled['best_scale_to_true_imag']:+.12g}j`",
        f"- Raw reconstructed `||x||_2`: `{rescaled['raw_x_norm_l2']:.12g}`",
        f"- True `||x_true||_2`: `{rescaled['true_x_norm_l2']:.12g}`",
        f"- Rescaled `||x||_2`: `{rescaled['rescaled_x_norm_l2']:.12g}`",
        f"- Cosine similarity to true `x`: `{rescaled['cosine_similarity_to_true']:.12g}`",
        f"- Rescaled relative solution error: `{rescaled['rescaled_relative_solution_error_l2']:.12g}`",
        f"- Rescaled residual norm: `{rescaled['rescaled_residual_norm_l2']:.12g}`",
        "",
        "## Final Trainable Parameters",
        "The first entry in each `alpha[i, j, :]` block is `sigma_ij`.",
        "",
        "### alpha",
        "```text",
        format_array(alpha_final),
        "```",
        "",
        "### beta",
        "```text",
        format_array(beta_final),
        "```",
        "",
        "## Final Reconstructed x",
        "```text",
        format_array(x_reconstructed),
        "```",
        "",
        "## True Solution x",
        "```text",
        format_array(x_true),
        "```",
        "",
        "## True Matrix A",
        "```text",
        format_array(a_dense),
        "```",
        "",
        "## True Vector b",
        "```text",
        format_array(b_dense),
        "```",
        "",
        "## Artifacts",
        f"- JSON: `{result['artifacts']['json']}`",
        f"- Figure: `{result['artifacts']['figure']}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def optimize(cfg: Config) -> dict[str, object]:
    import jax

    jax.config.update("jax_enable_x64", True)

    problem_np = build_partitioned_problem_numpy(cfg)
    problem_jax = to_jax_problem(problem_np)

    cost_fn = lambda a, b: global_cost_jax(a, b, cfg, problem_jax)
    full_grad_fn = jax.value_and_grad(cost_fn, argnums=(0, 1))
    alpha_grad_fn = jax.grad(cost_fn, argnums=0)

    alpha_init_np, beta_init_np = make_initial_parameters(cfg)
    state = initialize_state(cfg, alpha_grad_fn, alpha_init_np, beta_init_np)

    a_sparse = build_global_sparse_problem(cfg, problem_np["eta"], problem_np["zeta"])
    x_true, b_dense = exact_solution(a_sparse, cfg.global_qubits)

    history: list[dict[str, float | int]] = []
    initial_metrics = compute_metrics(alpha_init_np, beta_init_np, cfg, problem_np, a_sparse, b_dense, x_true)
    initial_metrics["iteration"] = 0
    initial_metrics["elapsed_s"] = 0.0
    history.append(initial_metrics)
    print(
        "[iter=0] "
        f"cost={initial_metrics['global_cost']:.12f} "
        f"residual={initial_metrics['residual_norm_l2']:.12f} "
        f"consensus={initial_metrics['consensus_error_l2']:.12f} "
        f"rel_err={initial_metrics['relative_solution_error_l2']:.12f}"
    )

    t0 = time.perf_counter()
    for iteration in range(1, cfg.iterations + 1):
        state, _ = distributed_iteration(state, cfg, problem_jax, full_grad_fn, alpha_grad_fn)

        alpha_np = np.asarray(state["alpha"], dtype=np.float64)
        beta_np = np.asarray(state["beta"], dtype=np.float64)
        metrics = compute_metrics(alpha_np, beta_np, cfg, problem_np, a_sparse, b_dense, x_true)
        metrics["iteration"] = iteration
        metrics["elapsed_s"] = time.perf_counter() - t0
        history.append(metrics)

        if (iteration % cfg.report_every == 0) or (iteration == cfg.iterations):
            print(
                f"[iter={iteration}] "
                f"cost={metrics['global_cost']:.12f} "
                f"residual={metrics['residual_norm_l2']:.12f} "
                f"consensus={metrics['consensus_error_l2']:.12f} "
                f"rel_err={metrics['relative_solution_error_l2']:.12f} "
                f"elapsed={metrics['elapsed_s']:.2f}s"
            )

    total_elapsed = time.perf_counter() - t0
    alpha_final = np.asarray(state["alpha"], dtype=np.float64)
    beta_final = np.asarray(state["beta"], dtype=np.float64)
    row1, row2 = reconstruct_row_solutions(alpha_final, cfg)
    x_reconstructed = 0.5 * (row1 + row2)
    a_dense = np.asarray(a_sparse.toarray(), dtype=np.complex128)
    rescaled_diagnostics = compute_rescaling_diagnostics(x_reconstructed, x_true, a_dense, b_dense)

    return {
        "config": asdict(cfg),
        "problem": {
            "lambda_min": float(problem_np["lambda_min"]),
            "lambda_max": float(problem_np["lambda_max"]),
            "eta": float(problem_np["eta"]),
            "zeta": float(problem_np["zeta"]),
            "scaled_lambda_min": float(problem_np["scaled_lambda_min"]),
            "scaled_lambda_max": float(problem_np["scaled_lambda_max"]),
            "row_laplacian": problem_np["row_laplacian"].tolist(),
        },
        "optimization": {
            "optimizer": "distributed_adam_gradient_tracking",
            "iterations": cfg.iterations,
            "elapsed_s": total_elapsed,
        },
        "rescaled_diagnostics": rescaled_diagnostics,
        "final_state": {
            "alpha": encode_array(alpha_final),
            "beta": encode_array(beta_final),
            "x_reconstructed": encode_array(x_reconstructed),
            "row1_solution": encode_array(row1),
            "row2_solution": encode_array(row2),
            "x_true": encode_array(x_true),
        },
        "linear_system": {
            "A_dense": encode_array(a_dense),
            "b": encode_array(b_dense),
        },
        "history": history,
    }


def main() -> None:
    cfg = make_config(parse_args())
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
