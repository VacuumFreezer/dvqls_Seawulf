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


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MPS_simulation.dist.quimb_dist_eq26_2x2_benchmark import (  # noqa: E402
    apply_block_mpo,
    distributed_iteration,
    initialize_state,
    to_jax_problem,
)


@dataclass
class Config:
    global_qubits: int
    local_qubits: int
    j_coupling: float
    kappa: float
    row_self_loop_weight: float
    gate_max_bond: int | None
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
    init_seed: int
    sigma_init: float
    angle_init_radius: float
    out_json: str | None
    out_figure: str | None
    out_report: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 13-qubit distributed J=0.1 study with an exact "
            "Hadamard -> (random Ry, CZ) -> (random Ry, CZ) ansatz and true-scale sigma/lambda init."
        )
    )
    parser.add_argument("--global-qubits", type=int, default=13)
    parser.add_argument("--local-qubits", type=int, default=12)
    parser.add_argument("--j-coupling", type=float, default=0.1)
    parser.add_argument("--kappa", type=float, default=20.0)
    parser.add_argument("--row-self-loop-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--init-seed", type=int, default=1234)
    parser.add_argument("--sigma-init", type=float, default=1.0 / math.sqrt(2.0))
    parser.add_argument("--angle-init-radius", type=float, default=0.1)
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
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
        gate_max_bond=None,
        gate_cutoff=0.0,
        apply_max_bond=64,
        apply_cutoff=1.0e-10,
        apply_no_compress=True,
        learning_rate=float(args.learning_rate),
        adam_beta1=float(args.adam_beta1),
        adam_beta2=float(args.adam_beta2),
        adam_epsilon=float(args.adam_epsilon),
        iterations=int(args.iterations),
        report_every=int(args.report_every),
        init_seed=int(args.init_seed),
        sigma_init=float(args.sigma_init),
        angle_init_radius=float(args.angle_init_radius),
        out_json=args.out_json,
        out_figure=args.out_figure,
        out_report=args.out_report,
    )


def resolve_output_paths(cfg: Config) -> dict[str, Path]:
    run_dir = THIS_DIR / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    default_base = run_dir / (
        f"j0p1_two_layer_ry_cz_true_scales_12q_near0p{str(cfg.angle_init_radius).replace('.', 'p')}_"
        f"lr{str(cfg.learning_rate).replace('.', 'p')}_"
        f"iter{cfg.iterations}_seed{cfg.init_seed}"
    )
    base = Path(cfg.out_json).with_suffix("") if cfg.out_json is not None else default_base
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


def scale_and_add_identity(mpo, coeff: float, nsites: int):
    import quimb.tensor as qtn

    ident = qtn.MPO_identity(nsites)
    return mpo.add_MPO(ident.multiply(coeff, inplace=False), inplace=False)


def build_base_local_mpo(local_qubits: int, j_coupling: float):
    import quimb as qu
    import quimb.tensor as qtn

    builder = qtn.SpinHam1D(cyclic=False)
    builder += 1.0, qu.pauli("X")
    builder += j_coupling, qu.pauli("Z"), qu.pauli("Z")
    return builder.build_mpo(local_qubits)


def build_boundary_z_mpo(local_qubits: int, coeff: float):
    import quimb as qu
    import quimb.tensor as qtn

    builder = qtn.SpinHam1D(cyclic=False)
    builder += coeff, qu.pauli("Z"), 0
    return builder.build_mpo(local_qubits)


def build_h0_mpo(global_qubits: int, j_coupling: float):
    import quimb as qu
    import quimb.tensor as qtn

    builder = qtn.SpinHam1D(cyclic=False)
    builder += 1.0, qu.pauli("X")
    builder += j_coupling, qu.pauli("Z"), qu.pauli("Z")
    return builder.build_mpo(global_qubits)


def estimate_scaled_spectrum(global_qubits: int, j_coupling: float, kappa: float) -> tuple[float, float]:
    import quimb.tensor as qtn

    if abs(j_coupling) <= 1.0e-15:
        lambda_min = -float(global_qubits)
        lambda_max = float(global_qubits)
        eta = (lambda_max - kappa * lambda_min) / (kappa - 1.0)
        zeta = lambda_max + eta
        return eta, zeta

    h0_mpo = build_h0_mpo(global_qubits, j_coupling)

    dmrg_min = qtn.DMRG2(
        h0_mpo,
        which="SA",
        bond_dims=[8, 16, 32, 64],
        cutoffs=[1.0e-8, 1.0e-10, 1.0e-12],
    )
    dmrg_min.solve(tol=1.0e-8, max_sweeps=6, verbosity=0)

    dmrg_max = qtn.DMRG2(
        h0_mpo,
        which="LA",
        bond_dims=[8, 16, 32, 64],
        cutoffs=[1.0e-8, 1.0e-10, 1.0e-12],
    )
    dmrg_max.solve(tol=1.0e-8, max_sweeps=6, verbosity=0)

    lambda_min = float(dmrg_min.energy)
    lambda_max = float(dmrg_max.energy)
    eta = (lambda_max - kappa * lambda_min) / (kappa - 1.0)
    zeta = lambda_max + eta
    return eta, zeta


def build_direct_problem(cfg: Config) -> dict[str, object]:
    import quimb.tensor as qtn

    eta, zeta = estimate_scaled_spectrum(cfg.global_qubits, cfg.j_coupling, cfg.kappa)
    local_base = build_base_local_mpo(cfg.local_qubits, cfg.j_coupling)

    if abs(cfg.j_coupling) <= 1.0e-15:
        a_diag = scale_and_add_identity(
            local_base.multiply(1.0 / zeta, inplace=False),
            eta / zeta,
            cfg.local_qubits,
        )
        a11 = a_diag.copy()
        a22 = a_diag.copy()
        a11_formula = "(H_rest + eta I) / zeta"
        a22_formula = "(H_rest + eta I) / zeta"
    else:
        boundary_z = build_boundary_z_mpo(cfg.local_qubits, cfg.j_coupling)
        a11 = scale_and_add_identity(
            local_base.add_MPO(boundary_z, inplace=False).multiply(1.0 / zeta, inplace=False),
            eta / zeta,
            cfg.local_qubits,
        )
        a22 = scale_and_add_identity(
            local_base.add_MPO(boundary_z.multiply(-1.0, inplace=False), inplace=False).multiply(
                1.0 / zeta,
                inplace=False,
            ),
            eta / zeta,
            cfg.local_qubits,
        )
        a11_formula = "(H_rest + J Z_1 + eta I) / zeta"
        a22_formula = "(H_rest - J Z_1 + eta I) / zeta"
    ident_local = qtn.MPO_identity(cfg.local_qubits).multiply(1.0 / zeta, inplace=False)
    blocks = ((a11, ident_local.copy()), (ident_local.copy(), a22))

    b_state = qtn.MPS_computational_state("+" * cfg.local_qubits, dtype="float64")
    b_norm = 0.5 / math.sqrt(2.0)
    b_states = ((b_state.copy(), b_state.copy()), (b_state.copy(), b_state.copy()))
    b_norms = np.full((2, 2), b_norm, dtype=np.float64)

    row_scale = 1.0 / (1.0 + cfg.row_self_loop_weight)
    row_laplacian = row_scale * np.asarray([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
    column_mix = np.asarray([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)

    return {
        "blocks": blocks,
        "b_states": b_states,
        "b_norms": b_norms,
        "row_laplacian": row_laplacian,
        "column_mix": column_mix,
        "eta": float(eta),
        "zeta": float(zeta),
        "block_formula": {
            "A11": a11_formula,
            "A12": "I / zeta",
            "A21": "I / zeta",
            "A22": a22_formula,
            "b_ij": "(1 / (2 sqrt(2))) |+^{29}>",
        },
    }


def wrap_params_numpy(params: np.ndarray) -> np.ndarray:
    wrapped = np.array(params, copy=True)
    wrapped[..., 1:] = ((wrapped[..., 1:] + math.pi) % (2.0 * math.pi)) - math.pi
    return wrapped


def build_circuit_numpy(n: int, angles: np.ndarray, cfg: Config):
    import quimb.tensor as qtn

    if angles.shape != (2 * n,):
        raise ValueError(f"Expected {2 * n} angles, got {angles.shape}.")

    circ = qtn.CircuitMPS(
        n,
        cutoff=cfg.gate_cutoff,
        max_bond=cfg.gate_max_bond,
    )
    for i in range(n):
        circ.h(i)
    for layer in range(2):
        offset = layer * n
        for i in range(n):
            circ.ry(float(angles[offset + i]), i)
        for i in range(n - 1):
            circ.cz(i, i + 1)
    return circ


def build_circuit_jax(n: int, angles, cfg: Config):
    import quimb.tensor as qtn

    circ = qtn.CircuitMPS(
        n,
        cutoff=cfg.gate_cutoff,
        max_bond=cfg.gate_max_bond,
    )
    for i in range(n):
        circ.h(i)
    for layer in range(2):
        offset = layer * n
        for i in range(n):
            circ.ry(angles[offset + i], i)
        for i in range(n - 1):
            circ.cz(i, i + 1)
    return circ


def make_initial_parameters(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    angle_count = 2 * cfg.local_qubits
    rng = np.random.default_rng(cfg.init_seed)
    alpha = np.empty((2, 2, angle_count + 1), dtype=np.float64)
    beta = np.empty((2, 2, angle_count + 1), dtype=np.float64)
    lambda_init = np.asarray(
        [
            [0.3423575332244867, -0.3423575332244867],
            [-0.34235753322448675, 0.34235753322448675],
        ],
        dtype=np.float64,
    )
    for i in range(2):
        for j in range(2):
            alpha[i, j, 0] = cfg.sigma_init
            beta[i, j, 0] = lambda_init[i, j]
            alpha[i, j, 1:] = rng.uniform(-cfg.angle_init_radius, cfg.angle_init_radius, size=angle_count)
            beta[i, j, 1:] = rng.uniform(-cfg.angle_init_radius, cfg.angle_init_radius, size=angle_count)
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
            z_states.append(build_circuit_jax(cfg.local_qubits, beta[i, k, 1:], cfg).psi)

        z_overlaps = [[None, None], [None, None]]
        for k in range(2):
            for p in range(2):
                z_overlaps[k][p] = z_states[k].overlap(z_states[p])

        for j in range(2):
            b_state = b_states[i][j]
            b_norm = b_norms[i, j]
            sigma = alpha[i, j, 0]
            x_state = build_circuit_jax(cfg.local_qubits, alpha[i, j, 1:], cfg).psi
            ax_state = apply_block_mpo(problem["blocks"][i][j], x_state, cfg)

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
                - 2.0 * sigma * ax_z_term
                - 2.0 * b_norm * ax_b
                + 2.0 * b_norm * b_z_term
            )

    return jnp.real(total)


def first_n_amplitudes(state, nsites: int, count: int) -> np.ndarray:
    values = np.empty(count, dtype=np.complex128)
    for idx in range(count):
        bitstring = format(idx, f"0{nsites}b")
        values[idx] = complex(state.amplitude(bitstring))
    return values


def stack_prefix(block1_prefix: np.ndarray, block2_prefix: np.ndarray, prefix_len: int, block_size: int) -> np.ndarray:
    if prefix_len <= block_size:
        return np.asarray(block1_prefix[:prefix_len], dtype=np.complex128)
    remaining = prefix_len - block_size
    return np.concatenate(
        [
            np.asarray(block1_prefix, dtype=np.complex128),
            np.asarray(block2_prefix[:remaining], dtype=np.complex128),
        ]
    )


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


def build_final_diagnostics(alpha: np.ndarray, beta: np.ndarray, cfg: Config, problem_np: dict[str, object]) -> dict[str, object]:
    prefix_len = 100
    block_size = 1 << cfg.local_qubits

    x_states = [[None, None], [None, None]]
    z_states = [[None, None], [None, None]]
    row_actions = [None, None]
    row_copy_actions = [None, None]
    row_action_prefixes: list[np.ndarray] = []
    row_residual_norms: list[float] = []
    row_copy_action_prefixes: list[np.ndarray] = []
    row_copy_residual_norms: list[float] = []
    b_i_prefixes: list[np.ndarray] = []
    b_ij_prefixes: list[list[np.ndarray]] = [[], []]
    x_ij_prefixes: list[list[np.ndarray]] = [[], []]
    x_unit_norms = np.zeros((2, 2), dtype=np.float64)
    x_scaled_norms = np.zeros((2, 2), dtype=np.float64)

    for i in range(2):
        for j in range(2):
            x_states[i][j] = build_circuit_numpy(cfg.local_qubits, alpha[i, j, 1:], cfg).psi
            z_states[i][j] = build_circuit_numpy(cfg.local_qubits, beta[i, j, 1:], cfg).psi
            x_unit_norms[i, j] = math.sqrt(
                max(float(np.real(x_states[i][j].overlap(x_states[i][j]))), 0.0)
            )
            x_scaled_norms[i, j] = abs(float(alpha[i, j, 0])) * x_unit_norms[i, j]
            x_amp = first_n_amplitudes(x_states[i][j], cfg.local_qubits, prefix_len)
            x_ij_prefixes[i].append(float(alpha[i, j, 0]) * x_amp)
            b_amp = first_n_amplitudes(problem_np["b_states"][i][j], cfg.local_qubits, prefix_len)
            b_ij_prefixes[i].append(float(problem_np["b_norms"][i, j]) * b_amp)

    x1_state = x_states[0][0].multiply(float(alpha[0, 0, 0]), inplace=False).add_MPS(
        x_states[1][0].multiply(float(alpha[1, 0, 0]), inplace=False),
        inplace=False,
        compress=False,
    ).multiply(0.5, inplace=False)
    x2_state = x_states[0][1].multiply(float(alpha[0, 1, 0]), inplace=False).add_MPS(
        x_states[1][1].multiply(float(alpha[1, 1, 0]), inplace=False),
        inplace=False,
        compress=False,
    ).multiply(0.5, inplace=False)

    for i in range(2):
        row_copy_action = None
        for j in range(2):
            ax_state = apply_block_mpo(problem_np["blocks"][i][j], x_states[i][j], cfg)
            scaled_ax = ax_state.multiply(float(alpha[i, j, 0]), inplace=False)
            row_copy_action = (
                scaled_ax
                if row_copy_action is None
                else row_copy_action.add_MPS(scaled_ax, inplace=False, compress=False)
            )
        row_copy_actions[i] = row_copy_action
        row_copy_action_prefixes.append(first_n_amplitudes(row_copy_action, cfg.local_qubits, prefix_len))

        b_i_state = problem_np["b_states"][i][0]
        b_i_scale = float(problem_np["b_norms"][i, 0] + problem_np["b_norms"][i, 1])
        b_i_prefix = b_i_scale * first_n_amplitudes(b_i_state, cfg.local_qubits, prefix_len)
        b_i_prefixes.append(b_i_prefix)

        row_copy_residual = row_copy_action.add_MPS(
            b_i_state.multiply(-b_i_scale, inplace=False),
            inplace=False,
            compress=False,
        )
        row_copy_norm_sq = float(np.real(row_copy_residual.overlap(row_copy_residual)))
        row_copy_residual_norms.append(math.sqrt(max(row_copy_norm_sq, 0.0)))

    for i in range(2):
        row_action = apply_block_mpo(problem_np["blocks"][i][0], x1_state, cfg)
        row_action = row_action.add_MPS(
            apply_block_mpo(problem_np["blocks"][i][1], x2_state, cfg),
            inplace=False,
            compress=False,
        )
        row_actions[i] = row_action
        row_action_prefixes.append(first_n_amplitudes(row_action, cfg.local_qubits, prefix_len))

        b_i_state = problem_np["b_states"][i][0]
        b_i_scale = float(problem_np["b_norms"][i, 0] + problem_np["b_norms"][i, 1])
        row_residual = row_action.add_MPS(
            b_i_state.multiply(-b_i_scale, inplace=False),
            inplace=False,
            compress=False,
        )
        row_norm_sq = float(np.real(row_residual.overlap(row_residual)))
        row_residual_norms.append(math.sqrt(max(row_norm_sq, 0.0)))

    x1_prefix = 0.5 * (
        float(alpha[0, 0, 0]) * first_n_amplitudes(x_states[0][0], cfg.local_qubits, prefix_len)
        + float(alpha[1, 0, 0]) * first_n_amplitudes(x_states[1][0], cfg.local_qubits, prefix_len)
    )
    x2_prefix = 0.5 * (
        float(alpha[0, 1, 0]) * first_n_amplitudes(x_states[0][1], cfg.local_qubits, prefix_len)
        + float(alpha[1, 1, 0]) * first_n_amplitudes(x_states[1][1], cfg.local_qubits, prefix_len)
    )
    x_reconstructed_prefix = stack_prefix(x1_prefix, x2_prefix, prefix_len, block_size)

    global_b_state = problem_np["b_states"][0][0].copy()
    global_b_prefix = (1.0 / math.sqrt(2.0)) * first_n_amplitudes(global_b_state, cfg.local_qubits, prefix_len)

    return {
        "sigma": np.asarray(alpha[:, :, 0], dtype=np.float64).tolist(),
        "lambda": np.asarray(beta[:, :, 0], dtype=np.float64).tolist(),
        "x_unit_norms": np.asarray(x_unit_norms, dtype=np.float64).tolist(),
        "x_scaled_norms": np.asarray(x_scaled_norms, dtype=np.float64).tolist(),
        "row_action_residual_norms": row_residual_norms,
        "row_copy_residual_norms": row_copy_residual_norms,
        "global_residual": math.sqrt(sum(v * v for v in row_residual_norms)),
        "x_reconstructed_prefix_100": np.real_if_close(x_reconstructed_prefix).tolist(),
        "x_ij_prefix_100": [
            [np.real_if_close(v).tolist() for v in row]
            for row in x_ij_prefixes
        ],
        "global_b_prefix_100": np.real_if_close(global_b_prefix).tolist(),
        "b_i_prefix_100": [np.real_if_close(v).tolist() for v in b_i_prefixes],
        "b_ij_prefix_100": [
            [np.real_if_close(v).tolist() for v in row]
            for row in b_ij_prefixes
        ],
        "row_action_prefix_100": [np.real_if_close(v).tolist() for v in row_action_prefixes],
        "row_copy_action_prefix_100": [np.real_if_close(v).tolist() for v in row_copy_action_prefixes],
    }


def plot_cost_history(history: list[dict[str, float | int]], figure_path: Path) -> None:
    iterations = [int(item["iteration"]) for item in history]
    costs = np.maximum([float(item["global_cost"]) for item in history], 1.0e-16)

    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=160)
    ax.plot(iterations, costs, linewidth=1.8, color="#005f73")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Global cost")
    ax.set_title("J=0.1 Two-Layer Ry+CZ Exact-Circuit Optimization")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def write_report(report_path: Path, result: dict[str, object]) -> None:
    history = result["history"]
    final = history[-1]
    diag = result["final_diagnostics"]
    lines = [
        "# J=0.1 Two-Layer Ry+CZ Exact-Circuit Optimization Report",
        "",
        "## Setup",
        f"- Global system: `{result['problem']['global_qubits']}` qubits.",
        f"- Local block size: `{result['problem']['local_qubits']}` qubits.",
        f"- `J = {result['config']['j_coupling']}`.",
        f"- Learning rate: `{result['config']['learning_rate']}`.",
        f"- Iterations completed: `{result['optimization']['iterations_completed']}` / `{result['optimization']['iterations_target']}`.",
        f"- Initialization seed: `{result['config']['init_seed']}`.",
        f"- Random angle init radius: `{result['config']['angle_init_radius']}`.",
        f"- Initial `sigma`: `{result['config']['sigma_init']}`.",
        "- Initial `lambda`: `[[0.3423575332244867, -0.3423575332244867], [-0.34235753322448675, 0.34235753322448675]]`.",
        f"- No-compression MPO apply: `{result['config']['apply_no_compress']}`.",
        f"- Exact circuit gate setting: `max_bond={result['config']['gate_max_bond']}`, `cutoff={result['config']['gate_cutoff']}`.",
        f"- `eta = {result['problem']['eta']:.12g}`.",
        f"- `zeta = {result['problem']['zeta']:.12g}`.",
        "",
        "## Outcome",
        f"- Final global cost: `{final['global_cost']:.12g}`.",
        f"- Best global cost in run: `{min(float(item['global_cost']) for item in history):.12g}`.",
        f"- Final global residual: `{diag['global_residual']:.12g}`.",
        f"- Final reconstructed-row residual norms: `{diag['row_action_residual_norms']}`.",
        f"- Final row-copy residual norms: `{diag['row_copy_residual_norms']}`.",
        f"- Final alpha gradient L2: `{final['alpha_grad_l2']:.12g}`.",
        f"- Final beta gradient L2: `{final['beta_grad_l2']:.12g}`.",
        f"- Total elapsed time: `{result['optimization']['elapsed_s']:.6f} s`.",
        "",
        "## sigma and lambda",
        "### sigma",
        "```text",
        format_array(np.asarray(diag["sigma"])),
        "```",
        "",
        "### lambda",
        "```text",
        format_array(np.asarray(diag["lambda"])),
        "```",
        "",
        "## State Norm Checks",
        "### || |X_ij> ||_2",
        "```text",
        format_array(np.asarray(diag["x_unit_norms"])),
        "```",
        "",
        "### ||x_ij||_2 = ||sigma_ij |X_ij>||_2",
        "```text",
        format_array(np.asarray(diag["x_scaled_norms"])),
        "```",
        "",
        "## First 100 Entries",
        "### reconstructed x",
        "```text",
        format_array(np.asarray(diag["x_reconstructed_prefix_100"])),
        "```",
        "",
        "### x_11",
        "```text",
        format_array(np.asarray(diag["x_ij_prefix_100"][0][0])),
        "```",
        "",
        "### x_12",
        "```text",
        format_array(np.asarray(diag["x_ij_prefix_100"][0][1])),
        "```",
        "",
        "### x_21",
        "```text",
        format_array(np.asarray(diag["x_ij_prefix_100"][1][0])),
        "```",
        "",
        "### x_22",
        "```text",
        format_array(np.asarray(diag["x_ij_prefix_100"][1][1])),
        "```",
        "",
        "### global b",
        "```text",
        format_array(np.asarray(diag["global_b_prefix_100"])),
        "```",
        "",
        "### b_1",
        "```text",
        format_array(np.asarray(diag["b_i_prefix_100"][0])),
        "```",
        "",
        "### b_2",
        "```text",
        format_array(np.asarray(diag["b_i_prefix_100"][1])),
        "```",
        "",
        "### b_11",
        "```text",
        format_array(np.asarray(diag["b_ij_prefix_100"][0][0])),
        "```",
        "",
        "### b_12",
        "```text",
        format_array(np.asarray(diag["b_ij_prefix_100"][0][1])),
        "```",
        "",
        "### b_21",
        "```text",
        format_array(np.asarray(diag["b_ij_prefix_100"][1][0])),
        "```",
        "",
        "### b_22",
        "```text",
        format_array(np.asarray(diag["b_ij_prefix_100"][1][1])),
        "```",
        "",
        "### A_11 x_1 + A_12 x_2",
        "```text",
        format_array(np.asarray(diag["row_action_prefix_100"][0])),
        "```",
        "",
        "### A_21 x_1 + A_22 x_2",
        "```text",
        format_array(np.asarray(diag["row_action_prefix_100"][1])),
        "```",
        "",
        "### A_11 x_11 + A_12 x_12",
        "```text",
        format_array(np.asarray(diag["row_copy_action_prefix_100"][0])),
        "```",
        "",
        "### A_21 x_21 + A_22 x_22",
        "```text",
        format_array(np.asarray(diag["row_copy_action_prefix_100"][1])),
        "```",
        "",
        "## Artifacts",
        f"- JSON: `{result['artifacts']['json']}`",
        f"- Figure: `{result['artifacts']['figure']}`",
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
    final_diagnostics = build_final_diagnostics(alpha_np, beta_np, cfg, problem_np)
    result = {
        "config": asdict(cfg),
        "problem": {
            "global_qubits": cfg.global_qubits,
            "local_qubits": cfg.local_qubits,
            "row_laplacian": np.asarray(problem_np["row_laplacian"], dtype=np.float64).tolist(),
            "b_norms": np.asarray(problem_np["b_norms"], dtype=np.float64).tolist(),
            "eta": float(problem_np["eta"]),
            "zeta": float(problem_np["zeta"]),
            "column_mix": np.asarray(problem_np["column_mix"], dtype=np.float64).tolist(),
            "block_formula": problem_np["block_formula"],
        },
        "optimization": {
            "iterations_target": cfg.iterations,
            "iterations_completed": int(history[-1]["iteration"]) if history else 0,
            "elapsed_s": time.perf_counter() - start_time,
        },
        "history": history,
        "final_state": {
            "alpha": np.asarray(alpha_np, dtype=np.float64).tolist(),
            "beta": np.asarray(beta_np, dtype=np.float64).tolist(),
        },
        "final_diagnostics": final_diagnostics,
        "artifacts": {key: str(path.resolve()) for key, path in paths.items()},
    }
    paths["json"].write_text(json.dumps(result, indent=2), encoding="utf-8")
    if history:
        plot_cost_history(history, paths["figure"])
        write_report(paths["report"], result)


def main() -> None:
    import jax
    import numpy as onp

    jax.config.update("jax_enable_x64", True)

    cfg = make_config(parse_args())
    paths = resolve_output_paths(cfg)

    problem_np = build_direct_problem(cfg)
    problem_jax = to_jax_problem(problem_np)
    alpha_init_np, beta_init_np = make_initial_parameters(cfg)

    cost_fn = lambda a, b: global_cost_jax(a, b, cfg, problem_jax)
    full_grad_fn = jax.value_and_grad(cost_fn, argnums=(0, 1))
    alpha_grad_fn = jax.grad(cost_fn, argnums=0)

    state = initialize_state(cfg, alpha_grad_fn, alpha_init_np, beta_init_np)
    history: list[dict[str, float | int]] = []
    start_time = time.perf_counter()

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
            "global_cost": float(onp.asarray(diag["current_cost"])),
            "alpha_grad_l2": float(onp.asarray(diag["alpha_grad_l2"])),
            "beta_grad_l2": float(onp.asarray(diag["beta_grad_l2"])),
            "alpha_step_l2": float(onp.asarray(diag["alpha_step_l2"])),
            "beta_step_l2": float(onp.asarray(diag["beta_step_l2"])),
            "elapsed_s": time.perf_counter() - start_time,
        }

        finite_state = (
            np.isfinite(np.asarray(state["alpha"])).all()
            and np.isfinite(np.asarray(state["beta"])).all()
            and all(np.isfinite(entry[k]) for k in ("global_cost", "alpha_grad_l2", "beta_grad_l2"))
        )
        if not finite_state:
            history.append(entry)
            write_progress_artifacts(paths, cfg, problem_np, history, state, start_time)
            raise RuntimeError(f"Non-finite value detected at iteration {iteration}.")

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
