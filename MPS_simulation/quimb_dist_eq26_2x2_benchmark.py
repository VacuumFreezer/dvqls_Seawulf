#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import eigsh

THIS_DIR = Path(__file__).resolve().parent
CEN_DIR = THIS_DIR.parent / "cen"
if str(CEN_DIR) not in sys.path:
    sys.path.insert(0, str(CEN_DIR))

from quimb_vqls_eq26_benchmark import (  # noqa: E402
    build_circuit_numpy,
    build_circuit_jax,
    time_callable,
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
    forward_repeats: int
    gradient_repeats: int
    iteration_repeats: int
    init_mode: str
    init_seed: int
    init_start: float
    init_stop: float
    x_scale_init: float
    z_scale_init: float
    out_json: str | None
    out_report: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a 2x2 distributed MPS implementation of the Eq. (26) "
            "global linear system using 10-qubit local block operators."
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
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--forward-repeats", type=int, default=3)
    parser.add_argument("--gradient-repeats", type=int, default=1)
    parser.add_argument("--iteration-repeats", type=int, default=1)
    parser.add_argument(
        "--init-mode",
        type=str,
        choices=("random_uniform", "structured_linspace"),
        default="random_uniform",
    )
    parser.add_argument("--init-seed", type=int, default=1234)
    parser.add_argument("--init-start", type=float, default=-math.pi)
    parser.add_argument("--init-stop", type=float, default=math.pi)
    parser.add_argument("--x-scale-init", type=float, default=1.0)
    parser.add_argument("--z-scale-init", type=float, default=1.0)
    parser.add_argument("--out-json", type=str, default=None)
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
        forward_repeats=int(args.forward_repeats),
        gradient_repeats=int(args.gradient_repeats),
        iteration_repeats=int(args.iteration_repeats),
        init_mode=str(args.init_mode),
        init_seed=int(args.init_seed),
        init_start=float(args.init_start),
        init_stop=float(args.init_stop),
        x_scale_init=float(args.x_scale_init),
        z_scale_init=float(args.z_scale_init),
        out_json=args.out_json,
        out_report=args.out_report,
    )


def resolve_output_paths(cfg: Config) -> dict[str, Path]:
    base = (
        Path(cfg.out_json).with_suffix("")
        if cfg.out_json is not None
        else THIS_DIR / "quimb_dist_eq26_2x2_benchmark_results_n11"
    )
    json_path = (
        Path(cfg.out_json) if cfg.out_json is not None else base.with_suffix(".json")
    )
    report_path = (
        Path(cfg.out_report)
        if cfg.out_report is not None
        else base.with_name(base.name + "_report").with_suffix(".md")
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    return {"json": json_path, "report": report_path}


def _matrix_to_mpo(matrix, local_qubits: int, cfg: Config):
    import quimb.tensor as qtn

    return qtn.MatrixProductOperator.from_dense(
        matrix,
        dims=[2] * local_qubits,
        cutoff=cfg.apply_cutoff,
        max_bond=cfg.apply_max_bond,
    )


def build_partitioned_problem_numpy(cfg: Config) -> dict[str, object]:
    import quimb as qu
    import quimb.tensor as qtn

    if cfg.global_qubits != cfg.local_qubits + 1:
        raise ValueError(
            "For the current 2x2 benchmark, global_qubits must equal local_qubits + 1."
        )

    h0_builder = qtn.SpinHam1D(cyclic=False)
    h0_builder += 1.0, qu.pauli("X")
    h0_builder += cfg.j_coupling, qu.pauli("Z"), qu.pauli("Z")
    h0_sparse = h0_builder.build_sparse(cfg.global_qubits).tocsr()

    lambda_min = float(eigsh(h0_sparse, k=1, which="SA", return_eigenvectors=False)[0])
    lambda_max = float(eigsh(h0_sparse, k=1, which="LA", return_eigenvectors=False)[0])
    eta = (lambda_max - cfg.kappa * lambda_min) / (cfg.kappa - 1.0)
    zeta = lambda_max + eta

    a_sparse = ((h0_sparse + eta * identity(h0_sparse.shape[0], format="csr")) / zeta).tocsr()
    scaled_lambda_min = float(eigsh(a_sparse, k=1, which="SA", return_eigenvectors=False)[0])
    scaled_lambda_max = float(eigsh(a_sparse, k=1, which="LA", return_eigenvectors=False)[0])

    half_dim = 2**cfg.local_qubits
    a_dense = np.asarray(a_sparse.toarray(), dtype=np.complex128)
    a11 = a_dense[:half_dim, :half_dim]
    a12 = a_dense[:half_dim, half_dim:]
    a21 = a_dense[half_dim:, :half_dim]
    a22 = a_dense[half_dim:, half_dim:]

    blocks = (
        (_matrix_to_mpo(a11, cfg.local_qubits, cfg), _matrix_to_mpo(a12, cfg.local_qubits, cfg)),
        (_matrix_to_mpo(a21, cfg.local_qubits, cfg), _matrix_to_mpo(a22, cfg.local_qubits, cfg)),
    )

    global_dim = 2**cfg.global_qubits
    b_dense = np.full(global_dim, 1.0 / math.sqrt(global_dim), dtype=np.complex128)
    b_rows = (
        np.array(b_dense[:half_dim], copy=True),
        np.array(b_dense[half_dim:], copy=True),
    )
    column_split = np.full((2, 2), 0.5, dtype=np.float64)
    b_vectors = []
    b_states = []
    b_norms = np.zeros((2, 2), dtype=np.float64)
    for i in range(2):
        row_vectors = []
        row_states = []
        for j in range(2):
            vector = column_split[i, j] * b_rows[i]
            norm = float(np.linalg.norm(vector))
            if norm <= 1.0e-15:
                state_vector = np.zeros_like(vector)
                state_vector[0] = 1.0
            else:
                state_vector = vector / norm
            row_vectors.append(vector)
            row_states.append(qtn.MatrixProductState.from_dense(state_vector, dims=[2] * cfg.local_qubits))
            b_norms[i, j] = norm
        b_vectors.append(tuple(row_vectors))
        b_states.append(tuple(row_states))

    row_scale = 1.0 / (1.0 + cfg.row_self_loop_weight)
    row_laplacian = row_scale * np.asarray(
        [[1.0, -1.0], [-1.0, 1.0]],
        dtype=np.float64,
    )
    column_mix = np.asarray([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)

    return {
        "blocks": blocks,
        "b_rows": b_rows,
        "b_vectors": tuple(b_vectors),
        "b_states": tuple(b_states),
        "b_norms": b_norms,
        "column_split": column_split,
        "row_laplacian": row_laplacian,
        "column_mix": column_mix,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "eta": eta,
        "zeta": zeta,
        "scaled_lambda_min": scaled_lambda_min,
        "scaled_lambda_max": scaled_lambda_max,
        "block_descriptions": {
            "A11": "Exact upper-left local block of the globally scaled simulator matrix.",
            "A12": "Exact upper-right local block of the globally scaled simulator matrix.",
            "A21": "Exact lower-left local block of the globally scaled simulator matrix.",
            "A22": "Exact lower-right local block of the globally scaled simulator matrix.",
        },
    }


def to_jax_problem(problem_np: dict[str, object]):
    import jax.numpy as jnp

    blocks = []
    for row in problem_np["blocks"]:
        converted_row = []
        for mpo in row:
            mpo_jax = mpo.copy()
            mpo_jax.apply_to_arrays(jnp.asarray)
            converted_row.append(mpo_jax)
        blocks.append(tuple(converted_row))

    b_states = []
    for row in problem_np["b_states"]:
        converted_row = []
        for state in row:
            state_jax = state.copy()
            state_jax.apply_to_arrays(jnp.asarray)
            converted_row.append(state_jax)
        b_states.append(tuple(converted_row))

    return {
        "blocks": tuple(blocks),
        "b_states": tuple(b_states),
        "b_norms": jnp.asarray(problem_np["b_norms"], dtype=jnp.float64),
        "row_laplacian": jnp.asarray(problem_np["row_laplacian"]),
        "column_mix": jnp.asarray(problem_np["column_mix"]),
    }


def wrap_params_numpy(params: np.ndarray) -> np.ndarray:
    wrapped = np.array(params, copy=True)
    wrapped[..., 1:] = ((wrapped[..., 1:] + math.pi) % (2.0 * math.pi)) - math.pi
    return wrapped


def wrap_params_jax(params):
    import jax.numpy as jnp

    scales = params[..., :1]
    angles = ((params[..., 1:] + math.pi) % (2.0 * math.pi)) - math.pi
    return jnp.concatenate((scales, angles), axis=-1)


def make_initial_parameters(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    angle_count = 2 * cfg.layers * cfg.local_qubits
    alpha = np.empty((2, 2, angle_count + 1), dtype=np.float64)
    beta = np.empty((2, 2, angle_count + 1), dtype=np.float64)

    if cfg.init_mode == "random_uniform":
        rng = np.random.default_rng(cfg.init_seed)
        for i in range(2):
            for j in range(2):
                alpha[i, j, 0] = cfg.x_scale_init
                alpha[i, j, 1:] = rng.uniform(cfg.init_start, cfg.init_stop, size=angle_count)
                beta[i, j, 0] = cfg.z_scale_init
                beta[i, j, 1:] = rng.uniform(cfg.init_start, cfg.init_stop, size=angle_count)
    elif cfg.init_mode == "structured_linspace":
        base_angles = np.linspace(cfg.init_start, cfg.init_stop, angle_count, dtype=np.float64)
        for i in range(2):
            for j in range(2):
                offset = 0.01 * (2 * i + j)
                alpha[i, j, 0] = cfg.x_scale_init + 0.05 * (i - j)
                alpha[i, j, 1:] = base_angles + offset
                beta[i, j, 0] = cfg.z_scale_init + 0.02 * (i + j + 1)
                beta[i, j, 1:] = base_angles[::-1] - 0.5 * offset
    else:
        raise ValueError(f"Unsupported init_mode: {cfg.init_mode}")

    return wrap_params_numpy(alpha), wrap_params_numpy(beta)


def global_cost_jax(alpha, beta, cfg: Config, problem: dict[str, object]):
    import jax.numpy as jnp

    total = jnp.asarray(0.0, dtype=jnp.float64)
    b_states = problem["b_states"]
    b_norms = problem["b_norms"]
    laplacian = problem["row_laplacian"]

    for i in range(2):
        z_states = []
        z_scales = []
        for k in range(2):
            z_scales.append(beta[i, k, 0])
            z_states.append(
                build_circuit_jax(cfg.local_qubits, beta[i, k, 1:], cfg).psi
            )

        z_overlaps = [[None, None], [None, None]]
        for k in range(2):
            for p in range(2):
                z_overlaps[k][p] = z_states[k].overlap(z_states[p])

        for j in range(2):
            b_state = b_states[i][j]
            b_norm = b_norms[i, j]
            sigma = alpha[i, j, 0]
            x_state = build_circuit_jax(cfg.local_qubits, alpha[i, j, 1:], cfg).psi
            ax_state = problem["blocks"][i][j].apply(
                x_state,
                contract=True,
                compress=True,
                max_bond=cfg.apply_max_bond,
                cutoff=cfg.apply_cutoff,
            )

            ax_norm = sigma * sigma * jnp.real(ax_state.overlap(ax_state))
            ax_b = sigma * jnp.real(b_state.overlap(ax_state))

            zz_term = jnp.asarray(0.0, dtype=jnp.float64)
            ax_z_term = jnp.asarray(0.0, dtype=jnp.float64)
            b_z_term = jnp.asarray(0.0, dtype=jnp.float64)

            for k in range(2):
                lk = laplacian[j, k]
                b_z_overlap = jnp.real(b_state.overlap(z_states[k]))
                ax_z_term = ax_z_term + lk * z_scales[k] * jnp.real(ax_state.overlap(z_states[k]))
                b_z_term = b_z_term + lk * z_scales[k] * b_z_overlap
                for p in range(2):
                    zz_term = zz_term + (
                        lk
                        * laplacian[j, p]
                        * z_scales[k]
                        * z_scales[p]
                        * jnp.real(z_overlaps[k][p])
                    )

            total = total + (
                ax_norm
                + zz_term
                + b_norm * b_norm
                - 2.0 * ax_z_term * sigma
                - 2.0 * b_norm * ax_b
                + 2.0 * b_norm * b_z_term
            )

    return jnp.real(total)


def global_cost_numpy(alpha: np.ndarray, beta: np.ndarray, cfg: Config, problem: dict[str, object]) -> float:
    total = 0.0
    b_states = problem["b_states"]
    b_norms = problem["b_norms"]
    laplacian = problem["row_laplacian"]

    for i in range(2):
        z_states = []
        z_scales = []
        for k in range(2):
            z_scales.append(float(beta[i, k, 0]))
            z_states.append(
                build_circuit_numpy(cfg.local_qubits, beta[i, k, 1:], cfg).psi
            )

        z_overlaps = [[None, None], [None, None]]
        for k in range(2):
            for p in range(2):
                z_overlaps[k][p] = float(np.real(z_states[k].overlap(z_states[p])))

        for j in range(2):
            b_state = b_states[i][j]
            b_norm = float(b_norms[i, j])
            sigma = float(alpha[i, j, 0])
            x_state = build_circuit_numpy(cfg.local_qubits, alpha[i, j, 1:], cfg).psi
            ax_state = problem["blocks"][i][j].apply(
                x_state,
                contract=True,
                compress=True,
                max_bond=cfg.apply_max_bond,
                cutoff=cfg.apply_cutoff,
            )

            ax_norm = sigma * sigma * float(np.real(ax_state.overlap(ax_state)))
            ax_b = sigma * float(np.real(b_state.overlap(ax_state)))

            zz_term = 0.0
            ax_z_term = 0.0
            b_z_term = 0.0
            for k in range(2):
                lk = float(laplacian[j, k])
                b_z_overlap = float(np.real(b_state.overlap(z_states[k])))
                ax_z_term += lk * z_scales[k] * float(np.real(ax_state.overlap(z_states[k])))
                b_z_term += lk * z_scales[k] * b_z_overlap
                for p in range(2):
                    zz_term += lk * float(laplacian[j, p]) * z_scales[k] * z_scales[p] * z_overlaps[k][p]

            total += (
                ax_norm
                + zz_term
                + b_norm * b_norm
                - 2.0 * sigma * ax_z_term
                - 2.0 * b_norm * ax_b
                + 2.0 * b_norm * b_z_term
            )

    return float(total)


def mix_columns(values, column_mix):
    import jax.numpy as jnp

    return jnp.einsum("rk,kjp->rjp", column_mix, values)


def adam_learning_rate(step: int, cfg: Config) -> float:
    return cfg.learning_rate * math.sqrt(1.0 - cfg.adam_beta2**step) / (1.0 - cfg.adam_beta1**step)


def distributed_iteration(
    state: dict[str, object],
    cfg: Config,
    problem_jax: dict[str, object],
    full_grad_fn,
    alpha_grad_fn,
):
    import jax.numpy as jnp

    step = int(state["step"]) + 1
    current_cost, (g_alpha_old, h_beta_old) = full_grad_fn(state["alpha"], state["beta"])
    lr_t = adam_learning_rate(step, cfg)

    a_beta = cfg.adam_beta1 * state["a_beta"] + (1.0 - cfg.adam_beta1) * h_beta_old
    b_beta = cfg.adam_beta2 * state["b_beta"] + (1.0 - cfg.adam_beta2) * (h_beta_old * h_beta_old)
    beta_step = lr_t * a_beta / (jnp.sqrt(b_beta) + cfg.adam_epsilon)
    beta_new = wrap_params_jax(state["beta"] - beta_step)

    a_alpha = cfg.adam_beta1 * state["a_alpha"] + (1.0 - cfg.adam_beta1) * state["y"]
    b_alpha = cfg.adam_beta2 * state["b_alpha"] + (1.0 - cfg.adam_beta2) * (state["y"] * state["y"])
    alpha_step = lr_t * a_alpha / (jnp.sqrt(b_alpha) + cfg.adam_epsilon)
    mixed_alpha = mix_columns(state["alpha"], problem_jax["column_mix"])
    alpha_new = wrap_params_jax(mixed_alpha - alpha_step)

    g_alpha_new = alpha_grad_fn(alpha_new, beta_new)
    mixed_y = mix_columns(state["y"], problem_jax["column_mix"])
    y_new = mixed_y + g_alpha_new - g_alpha_old

    diagnostics = {
        "step": step,
        "current_cost": current_cost,
        "alpha_grad_l2": jnp.linalg.norm(g_alpha_old),
        "beta_grad_l2": jnp.linalg.norm(h_beta_old),
        "alpha_step_l2": jnp.linalg.norm(alpha_step),
        "beta_step_l2": jnp.linalg.norm(beta_step),
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
        "last_cost": current_cost,
    }, diagnostics


def initialize_state(cfg: Config, alpha_grad_fn, alpha_init_np: np.ndarray, beta_init_np: np.ndarray):
    import jax.numpy as jnp

    alpha = jnp.asarray(alpha_init_np)
    beta = jnp.asarray(beta_init_np)
    y = alpha_grad_fn(alpha, beta)

    return {
        "step": 0,
        "alpha": alpha,
        "beta": beta,
        "y": y,
        "a_alpha": jnp.zeros_like(alpha),
        "b_alpha": jnp.zeros_like(alpha),
        "a_beta": jnp.zeros_like(beta),
        "b_beta": jnp.zeros_like(beta),
        "last_cost": None,
    }


def write_results_report(report_path: Path, result: dict[str, object]) -> None:
    timing = result["timings"]
    iteration = timing["one_iteration"]
    forward = timing["forward_cost"]
    gradient = timing["reverse_mode_gradient"]
    summary = result["iteration_summary"]

    lines = [
        "# Distributed Eq. (26) 2x2 MPS Benchmark",
        "",
        "## Setup",
        f"- Global system: `{result['problem']['global_qubits']}`-qubit Eq. (26) Ising-inspired linear system.",
        f"- Partition: 2x2 block decomposition with `{result['problem']['local_qubits']}`-qubit local operators.",
        f"- Row graph Laplacian: `{result['problem']['row_laplacian']}`.",
        "- Column consensus weights: `[[0.5, 0.5], [0.5, 0.5]]`.",
        f"- Scaled spectrum check: `lambda_min={result['problem']['scaled_lambda_min']:.12g}`, `lambda_max={result['problem']['scaled_lambda_max']:.12g}`.",
        "",
        "## Timings",
        f"- Forward global cost mean: `{forward['mean_s']:.6f} s`.",
        f"- Reverse-mode global gradient mean: `{gradient['mean_s']:.6f} s`.",
        f"- One distributed iteration mean: `{iteration['mean_s']:.6f} s`.",
        "",
        "## One Iteration Diagnostics",
        f"- Current global cost: `{summary['current_cost']:.12g}`.",
        f"- Alpha gradient L2 norm: `{summary['alpha_grad_l2']:.12g}`.",
        f"- Beta gradient L2 norm: `{summary['beta_grad_l2']:.12g}`.",
        f"- Alpha update L2 norm: `{summary['alpha_step_l2']:.12g}`.",
        f"- Beta update L2 norm: `{summary['beta_step_l2']:.12g}`.",
        "",
        "## Artifacts",
        f"- JSON: `{result['artifacts']['json']}`",
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import jax
    import numpy as onp

    jax.config.update("jax_enable_x64", True)

    cfg = make_config(parse_args())
    artifact_paths = resolve_output_paths(cfg)

    print("Building 11-qubit scaled problem and 10-qubit local block MPOs...")
    problem_np = build_partitioned_problem_numpy(cfg)
    problem_jax = to_jax_problem(problem_np)

    param_dim = 1 + 2 * cfg.layers * cfg.local_qubits
    alpha_init_np, beta_init_np = make_initial_parameters(cfg)

    cost_fn = lambda a, b: global_cost_jax(a, b, cfg, problem_jax)
    full_grad_fn = jax.value_and_grad(cost_fn, argnums=(0, 1))
    alpha_grad_fn = jax.grad(cost_fn, argnums=0)

    initial_state = initialize_state(cfg, alpha_grad_fn, alpha_init_np, beta_init_np)

    alpha_init = initial_state["alpha"]
    beta_init = initial_state["beta"]

    print("Timing forward global cost evaluation...")
    forward = time_callable(
        lambda: global_cost_numpy(alpha_init_np, beta_init_np, cfg, problem_np),
        repeats=cfg.forward_repeats,
        warmup=1,
    )

    print("Timing reverse-mode gradient of the full distributed objective...")
    reverse = time_callable(
        lambda: full_grad_fn(alpha_init, beta_init),
        repeats=cfg.gradient_repeats,
        warmup=1,
    )

    print("Timing one full distributed iteration (gradient + Adam + consensus/tracking update)...")
    iteration = time_callable(
        lambda: distributed_iteration(initial_state, cfg, problem_jax, full_grad_fn, alpha_grad_fn),
        repeats=cfg.iteration_repeats,
        warmup=1,
    )

    _, iteration_info = iteration["last_value"]

    result = {
        "config": asdict(cfg),
        "problem": {
            "global_qubits": cfg.global_qubits,
            "local_qubits": cfg.local_qubits,
            "lambda_min": float(problem_np["lambda_min"]),
            "lambda_max": float(problem_np["lambda_max"]),
            "eta": float(problem_np["eta"]),
            "zeta": float(problem_np["zeta"]),
            "scaled_lambda_min": float(problem_np["scaled_lambda_min"]),
            "scaled_lambda_max": float(problem_np["scaled_lambda_max"]),
            "row_laplacian": problem_np["row_laplacian"].tolist(),
            "param_dim_per_agent_alpha": int(param_dim),
            "param_dim_per_agent_beta": int(param_dim),
            "total_trainable_parameters": int(8 * param_dim),
            "block_descriptions": problem_np["block_descriptions"],
        },
        "timings": {
            "forward_cost": {
                "value": float(onp.asarray(forward["last_value"])),
                **forward["timing"],
            },
            "reverse_mode_gradient": {
                "value": float(onp.asarray(reverse["last_value"][0])),
                "alpha_grad_l2": float(onp.linalg.norm(onp.asarray(reverse["last_value"][1][0]))),
                "beta_grad_l2": float(onp.linalg.norm(onp.asarray(reverse["last_value"][1][1]))),
                **reverse["timing"],
            },
            "one_iteration": {
                **iteration["timing"],
            },
        },
        "iteration_summary": {
            "current_cost": float(onp.asarray(iteration_info["current_cost"])),
            "alpha_grad_l2": float(onp.asarray(iteration_info["alpha_grad_l2"])),
            "beta_grad_l2": float(onp.asarray(iteration_info["beta_grad_l2"])),
            "alpha_step_l2": float(onp.asarray(iteration_info["alpha_step_l2"])),
            "beta_step_l2": float(onp.asarray(iteration_info["beta_step_l2"])),
        },
    }

    result["artifacts"] = {key: str(path.resolve()) for key, path in artifact_paths.items()}

    artifact_paths["json"].write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_results_report(artifact_paths["report"], result)

    print("\nBenchmark summary:")
    print(json.dumps(result, indent=2))
    print(f"\nWrote JSON to {artifact_paths['json']}")
    print(f"Wrote report to {artifact_paths['report']}")


if __name__ == "__main__":
    main()
