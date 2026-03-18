#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from quimb_dist_eq26_2x2_benchmark import (  # noqa: E402
    adam_learning_rate,
    build_circuit_numpy,
    build_partitioned_problem_numpy,
    global_cost_numpy,
    make_initial_parameters,
    wrap_params_numpy,
)
from quimb_dist_eq26_2x2_optimize import (  # noqa: E402
    build_final_diagnostics,
    build_global_sparse_problem,
    compute_metrics,
    compute_rescaling_diagnostics,
    encode_array,
    exact_solution,
    plot_history,
    reconstruct_row_solutions,
    resolve_output_paths,
    write_report,
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
    init_mode: str
    init_seed: int
    init_start: float
    init_stop: float
    x_scale_init: float
    z_scale_init: float
    spsa_seed: int
    spsa_c: float
    spsa_directions: int
    spsa_full_params: bool
    out_json: str | None
    out_figure: str | None
    out_report: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 2x2 distributed Eq. (26) optimizer with SPSA angle "
            "gradients and analytic sigma/lambda gradients."
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
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--report-every", type=int, default=20)
    parser.add_argument(
        "--init-mode",
        type=str,
        choices=("structured_linspace",),
        default="structured_linspace",
    )
    parser.add_argument("--init-seed", type=int, default=1234)
    parser.add_argument("--init-start", type=float, default=0.01)
    parser.add_argument("--init-stop", type=float, default=0.2)
    parser.add_argument("--x-scale-init", type=float, default=0.75)
    parser.add_argument("--z-scale-init", type=float, default=0.10)
    parser.add_argument("--spsa-seed", type=int, default=1234)
    parser.add_argument("--spsa-c", type=float, default=0.05)
    parser.add_argument("--spsa-directions", type=int, default=1)
    parser.add_argument("--spsa-full-params", action="store_true")
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(
            THIS_DIR
            / "5qubits"
            / "quimb_dist_eq26_2x2_optimize_spsa_n6_local5_k20_iter200.json"
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
        spsa_seed=int(args.spsa_seed),
        spsa_c=float(args.spsa_c),
        spsa_directions=int(args.spsa_directions),
        spsa_full_params=bool(args.spsa_full_params),
        out_json=args.out_json,
        out_figure=args.out_figure,
        out_report=args.out_report,
    )


def mix_columns_numpy(values: np.ndarray, column_mix: np.ndarray) -> np.ndarray:
    return np.einsum("rk,kjp->rjp", column_mix, values)


def cost_and_scale_grads(
    alpha: np.ndarray,
    beta: np.ndarray,
    cfg: Config,
    problem_np: dict[str, object],
) -> tuple[float, np.ndarray, np.ndarray]:
    total = 0.0
    alpha_grad = np.zeros_like(alpha, dtype=np.float64)
    beta_grad = np.zeros_like(beta, dtype=np.float64)
    b_states = problem_np["b_states"]
    b_norms = np.asarray(problem_np["b_norms"], dtype=np.float64)
    laplacian = np.asarray(problem_np["row_laplacian"], dtype=np.float64)

    for i in range(2):
        z_states = []
        lambdas = np.zeros(2, dtype=np.float64)
        for k in range(2):
            lambdas[k] = float(beta[i, k, 0])
            z_states.append(build_circuit_numpy(cfg.local_qubits, beta[i, k, 1:], cfg).psi)

        z_overlaps = np.zeros((2, 2), dtype=np.float64)
        for k in range(2):
            for p in range(2):
                z_overlaps[k, p] = float(np.real(z_states[k].overlap(z_states[p])))

        sigmas = np.zeros(2, dtype=np.float64)
        ax_norm_terms = np.zeros(2, dtype=np.float64)
        ax_b_overlaps = np.zeros(2, dtype=np.float64)
        ax_z_overlaps = np.zeros((2, 2), dtype=np.float64)
        b_z_overlaps = np.zeros((2, 2), dtype=np.float64)

        for j in range(2):
            sigmas[j] = float(alpha[i, j, 0])
            b_state = b_states[i][j]
            x_state = build_circuit_numpy(cfg.local_qubits, alpha[i, j, 1:], cfg).psi
            ax_state = problem_np["blocks"][i][j].apply(
                x_state,
                contract=True,
                compress=True,
                max_bond=cfg.apply_max_bond,
                cutoff=cfg.apply_cutoff,
            )

            ax_norm_terms[j] = float(np.real(ax_state.overlap(ax_state)))
            ax_b_overlaps[j] = float(np.real(b_state.overlap(ax_state)))
            for k in range(2):
                ax_z_overlaps[j, k] = float(np.real(ax_state.overlap(z_states[k])))
                b_z_overlaps[j, k] = float(np.real(b_state.overlap(z_states[k])))

            az_term = 0.0
            bz_term = 0.0
            zz_term = 0.0
            for k in range(2):
                az_term += float(laplacian[j, k]) * lambdas[k] * ax_z_overlaps[j, k]
                bz_term += float(laplacian[j, k]) * lambdas[k] * b_z_overlaps[j, k]
                for p in range(2):
                    zz_term += (
                        float(laplacian[j, k])
                        * float(laplacian[j, p])
                        * lambdas[k]
                        * lambdas[p]
                        * z_overlaps[k, p]
                    )

            b_norm = float(b_norms[i, j])
            total += (
                sigmas[j] * sigmas[j] * ax_norm_terms[j]
                + zz_term
                + b_norm * b_norm
                - 2.0 * sigmas[j] * az_term
                - 2.0 * b_norm * sigmas[j] * ax_b_overlaps[j]
                + 2.0 * b_norm * bz_term
            )
            alpha_grad[i, j, 0] = (
                2.0 * sigmas[j] * ax_norm_terms[j]
                - 2.0 * az_term
                - 2.0 * b_norm * ax_b_overlaps[j]
            )

        for r in range(2):
            lambda_grad = 0.0
            for j in range(2):
                ljr = float(laplacian[j, r])
                zz_partial = 0.0
                for p in range(2):
                    zz_partial += (
                        float(laplacian[j, p]) * lambdas[p] * z_overlaps[r, p]
                    )
                lambda_grad += 2.0 * ljr * zz_partial
                lambda_grad -= 2.0 * sigmas[j] * ljr * ax_z_overlaps[j, r]
                lambda_grad += 2.0 * float(b_norms[i, j]) * ljr * b_z_overlaps[j, r]
            beta_grad[i, r, 0] = lambda_grad

    return float(total), alpha_grad, beta_grad


def spsa_angle_grads(
    alpha: np.ndarray,
    beta: np.ndarray,
    cfg: Config,
    problem_np: dict[str, object],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    alpha_delta = rng.choice(np.array([-1.0, 1.0]), size=alpha[..., 1:].shape)
    beta_delta = rng.choice(np.array([-1.0, 1.0]), size=beta[..., 1:].shape)

    alpha_plus = np.array(alpha, copy=True)
    alpha_minus = np.array(alpha, copy=True)
    beta_plus = np.array(beta, copy=True)
    beta_minus = np.array(beta, copy=True)

    alpha_plus[..., 1:] += cfg.spsa_c * alpha_delta
    alpha_minus[..., 1:] -= cfg.spsa_c * alpha_delta
    beta_plus[..., 1:] += cfg.spsa_c * beta_delta
    beta_minus[..., 1:] -= cfg.spsa_c * beta_delta

    alpha_plus = wrap_params_numpy(alpha_plus)
    alpha_minus = wrap_params_numpy(alpha_minus)
    beta_plus = wrap_params_numpy(beta_plus)
    beta_minus = wrap_params_numpy(beta_minus)

    f_plus = global_cost_numpy(alpha_plus, beta_plus, cfg, problem_np)
    f_minus = global_cost_numpy(alpha_minus, beta_minus, cfg, problem_np)
    scale = (f_plus - f_minus) / (2.0 * cfg.spsa_c)

    alpha_grad = np.zeros_like(alpha, dtype=np.float64)
    beta_grad = np.zeros_like(beta, dtype=np.float64)
    alpha_grad[..., 1:] = scale * alpha_delta
    beta_grad[..., 1:] = scale * beta_delta
    return alpha_grad, beta_grad


def spsa_full_grads(
    alpha: np.ndarray,
    beta: np.ndarray,
    cfg: Config,
    problem_np: dict[str, object],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    alpha_grad = np.zeros_like(alpha, dtype=np.float64)
    beta_grad = np.zeros_like(beta, dtype=np.float64)

    for _ in range(cfg.spsa_directions):
        alpha_delta = rng.choice(np.array([-1.0, 1.0]), size=alpha.shape)
        beta_delta = rng.choice(np.array([-1.0, 1.0]), size=beta.shape)

        alpha_plus = wrap_params_numpy(alpha + cfg.spsa_c * alpha_delta)
        alpha_minus = wrap_params_numpy(alpha - cfg.spsa_c * alpha_delta)
        beta_plus = wrap_params_numpy(beta + cfg.spsa_c * beta_delta)
        beta_minus = wrap_params_numpy(beta - cfg.spsa_c * beta_delta)

        f_plus = global_cost_numpy(alpha_plus, beta_plus, cfg, problem_np)
        f_minus = global_cost_numpy(alpha_minus, beta_minus, cfg, problem_np)
        scale = (f_plus - f_minus) / (2.0 * cfg.spsa_c)
        alpha_grad += scale * alpha_delta
        beta_grad += scale * beta_delta

    alpha_grad /= float(cfg.spsa_directions)
    beta_grad /= float(cfg.spsa_directions)
    return alpha_grad, beta_grad


def spsa_alpha_angle_grad(
    alpha: np.ndarray,
    beta: np.ndarray,
    cfg: Config,
    problem_np: dict[str, object],
    rng: np.random.Generator,
) -> np.ndarray:
    alpha_delta = rng.choice(np.array([-1.0, 1.0]), size=alpha[..., 1:].shape)
    alpha_plus = np.array(alpha, copy=True)
    alpha_minus = np.array(alpha, copy=True)
    alpha_plus[..., 1:] += cfg.spsa_c * alpha_delta
    alpha_minus[..., 1:] -= cfg.spsa_c * alpha_delta
    alpha_plus = wrap_params_numpy(alpha_plus)
    alpha_minus = wrap_params_numpy(alpha_minus)

    f_plus = global_cost_numpy(alpha_plus, beta, cfg, problem_np)
    f_minus = global_cost_numpy(alpha_minus, beta, cfg, problem_np)
    scale = (f_plus - f_minus) / (2.0 * cfg.spsa_c)

    alpha_grad = np.zeros_like(alpha, dtype=np.float64)
    alpha_grad[..., 1:] = scale * alpha_delta
    return alpha_grad


def spsa_full_alpha_grad(
    alpha: np.ndarray,
    beta: np.ndarray,
    cfg: Config,
    problem_np: dict[str, object],
    rng: np.random.Generator,
) -> np.ndarray:
    alpha_grad = np.zeros_like(alpha, dtype=np.float64)
    for _ in range(cfg.spsa_directions):
        alpha_delta = rng.choice(np.array([-1.0, 1.0]), size=alpha.shape)
        alpha_plus = wrap_params_numpy(alpha + cfg.spsa_c * alpha_delta)
        alpha_minus = wrap_params_numpy(alpha - cfg.spsa_c * alpha_delta)

        f_plus = global_cost_numpy(alpha_plus, beta, cfg, problem_np)
        f_minus = global_cost_numpy(alpha_minus, beta, cfg, problem_np)
        scale = (f_plus - f_minus) / (2.0 * cfg.spsa_c)
        alpha_grad += scale * alpha_delta

    alpha_grad /= float(cfg.spsa_directions)
    return alpha_grad


def hybrid_value_and_grads(
    alpha: np.ndarray,
    beta: np.ndarray,
    cfg: Config,
    problem_np: dict[str, object],
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, np.ndarray]:
    cost_value, alpha_grad, beta_grad = cost_and_scale_grads(alpha, beta, cfg, problem_np)
    alpha_angle_grad, beta_angle_grad = spsa_angle_grads(alpha, beta, cfg, problem_np, rng)
    alpha_grad[..., 1:] = alpha_angle_grad[..., 1:]
    beta_grad[..., 1:] = beta_angle_grad[..., 1:]
    return cost_value, alpha_grad, beta_grad


def hybrid_alpha_grad(
    alpha: np.ndarray,
    beta: np.ndarray,
    cfg: Config,
    problem_np: dict[str, object],
    rng: np.random.Generator,
) -> np.ndarray:
    _, alpha_grad, _ = cost_and_scale_grads(alpha, beta, cfg, problem_np)
    alpha_angle_grad = spsa_alpha_angle_grad(alpha, beta, cfg, problem_np, rng)
    alpha_grad[..., 1:] = alpha_angle_grad[..., 1:]
    return alpha_grad


def spsa_value_and_grads(
    alpha: np.ndarray,
    beta: np.ndarray,
    cfg: Config,
    problem_np: dict[str, object],
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, np.ndarray]:
    if cfg.spsa_full_params:
        cost_value = global_cost_numpy(alpha, beta, cfg, problem_np)
        alpha_grad, beta_grad = spsa_full_grads(alpha, beta, cfg, problem_np, rng)
        return cost_value, alpha_grad, beta_grad
    return hybrid_value_and_grads(alpha, beta, cfg, problem_np, rng)


def spsa_alpha_grad(
    alpha: np.ndarray,
    beta: np.ndarray,
    cfg: Config,
    problem_np: dict[str, object],
    rng: np.random.Generator,
) -> np.ndarray:
    if cfg.spsa_full_params:
        return spsa_full_alpha_grad(alpha, beta, cfg, problem_np, rng)
    return hybrid_alpha_grad(alpha, beta, cfg, problem_np, rng)


def initialize_state(
    cfg: Config,
    problem_np: dict[str, object],
    rng: np.random.Generator,
    alpha_init_np: np.ndarray,
    beta_init_np: np.ndarray,
) -> dict[str, object]:
    y = spsa_alpha_grad(alpha_init_np, beta_init_np, cfg, problem_np, rng)
    return {
        "step": 0,
        "alpha": np.array(alpha_init_np, copy=True),
        "beta": np.array(beta_init_np, copy=True),
        "y": y,
        "a_alpha": np.zeros_like(alpha_init_np, dtype=np.float64),
        "b_alpha": np.zeros_like(alpha_init_np, dtype=np.float64),
        "a_beta": np.zeros_like(beta_init_np, dtype=np.float64),
        "b_beta": np.zeros_like(beta_init_np, dtype=np.float64),
        "last_cost": None,
    }


def distributed_iteration_spsa(
    state: dict[str, object],
    cfg: Config,
    problem_np: dict[str, object],
    rng: np.random.Generator,
) -> tuple[dict[str, object], dict[str, float]]:
    step = int(state["step"]) + 1
    current_cost, g_alpha_old, h_beta_old = spsa_value_and_grads(
        state["alpha"], state["beta"], cfg, problem_np, rng
    )
    lr_t = adam_learning_rate(step, cfg)

    a_beta = cfg.adam_beta1 * state["a_beta"] + (1.0 - cfg.adam_beta1) * h_beta_old
    b_beta = cfg.adam_beta2 * state["b_beta"] + (1.0 - cfg.adam_beta2) * (h_beta_old * h_beta_old)
    beta_step = lr_t * a_beta / (np.sqrt(b_beta) + cfg.adam_epsilon)
    beta_new = wrap_params_numpy(state["beta"] - beta_step)

    a_alpha = cfg.adam_beta1 * state["a_alpha"] + (1.0 - cfg.adam_beta1) * state["y"]
    b_alpha = cfg.adam_beta2 * state["b_alpha"] + (1.0 - cfg.adam_beta2) * (state["y"] * state["y"])
    alpha_step = lr_t * a_alpha / (np.sqrt(b_alpha) + cfg.adam_epsilon)
    mixed_alpha = mix_columns_numpy(state["alpha"], np.asarray(problem_np["column_mix"], dtype=np.float64))
    alpha_new = wrap_params_numpy(mixed_alpha - alpha_step)

    g_alpha_new = spsa_alpha_grad(alpha_new, beta_new, cfg, problem_np, rng)
    mixed_y = mix_columns_numpy(state["y"], np.asarray(problem_np["column_mix"], dtype=np.float64))
    y_new = mixed_y + g_alpha_new - g_alpha_old

    diagnostics = {
        "step": float(step),
        "current_cost": float(current_cost),
        "alpha_grad_l2": float(np.linalg.norm(g_alpha_old)),
        "beta_grad_l2": float(np.linalg.norm(h_beta_old)),
        "alpha_step_l2": float(np.linalg.norm(alpha_step)),
        "beta_step_l2": float(np.linalg.norm(beta_step)),
    }

    return {
        "step": step,
        "alpha": alpha_new,
        "beta": beta_new,
        "y": y_new,
        "a_alpha": a_alpha,
        "b_alpha": b_alpha,
        "a_beta": a_beta,
        "b_beta": b_beta,
        "last_cost": float(current_cost),
    }, diagnostics


def optimize(cfg: Config) -> dict[str, object]:
    problem_np = build_partitioned_problem_numpy(cfg)
    alpha_init_np, beta_init_np = make_initial_parameters(cfg)
    rng = np.random.default_rng(cfg.spsa_seed)
    state = initialize_state(cfg, problem_np, rng, alpha_init_np, beta_init_np)

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
        state, _ = distributed_iteration_spsa(state, cfg, problem_np, rng)
        metrics = compute_metrics(state["alpha"], state["beta"], cfg, problem_np, a_sparse, b_dense, x_true)
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
    b_row_norms = [float(np.linalg.norm(row)) for row in problem_np["b_rows"]]
    final_diagnostics = build_final_diagnostics(alpha_final, cfg, problem_np, a_dense)

    result = {
        "config": asdict(cfg),
        "problem": {
            "lambda_min": float(problem_np["lambda_min"]),
            "lambda_max": float(problem_np["lambda_max"]),
            "eta": float(problem_np["eta"]),
            "zeta": float(problem_np["zeta"]),
            "scaled_lambda_min": float(problem_np["scaled_lambda_min"]),
            "scaled_lambda_max": float(problem_np["scaled_lambda_max"]),
            "row_laplacian": problem_np["row_laplacian"].tolist(),
            "b_row_norms": b_row_norms,
            "b_agent_norms": np.asarray(problem_np["b_norms"], dtype=np.float64).tolist(),
            "b_column_split": np.asarray(problem_np["column_split"], dtype=np.float64).tolist(),
            "b_rows": [encode_array(row) for row in problem_np["b_rows"]],
            "b_vectors": [
                [encode_array(problem_np["b_vectors"][i][j]) for j in range(2)]
                for i in range(2)
            ],
        },
        "optimization": {
            "optimizer": (
                "distributed_adam_gradient_tracking_spsa_all_params"
                if cfg.spsa_full_params
                else "distributed_adam_gradient_tracking_spsa_angles_analytic_scales"
            ),
            "iterations": cfg.iterations,
            "elapsed_s": total_elapsed,
            "spsa_seed": cfg.spsa_seed,
            "spsa_c": cfg.spsa_c,
            "spsa_directions": cfg.spsa_directions,
            "spsa_full_params": cfg.spsa_full_params,
        },
        "initialization": {
            "mode": cfg.init_mode,
            "sigma_init": cfg.x_scale_init,
            "lambda_init": cfg.z_scale_init,
            "angle_range": [cfg.init_start, cfg.init_stop],
        },
        "rescaled_diagnostics": rescaled_diagnostics,
        "final_diagnostics": final_diagnostics,
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
    return result


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
