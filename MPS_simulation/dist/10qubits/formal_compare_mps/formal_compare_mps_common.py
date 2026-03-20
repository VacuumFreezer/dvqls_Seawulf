#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MPS_simulation.quimb_dist_eq26_common import (  # noqa: E402
    JsonlWriter,
    atomic_write_json,
    dump_yaml_config,
    encode_array,
    format_array_preview,
    merge_section_config,
    sanitize_jsonable,
)


DEFAULT_PARAM_PATH = THIS_DIR / "param.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "global_qubits": 13,
    "local_qubits": 12,
    "j_coupling": 0.1,
    "kappa": 20.0,
    "layers": 12,
    "gate_max_bond": None,
    "gate_cutoff": 0.0,
    "apply_max_bond": 64,
    "apply_cutoff": 1.0e-10,
    "apply_no_compress": True,
    "learning_rate": 0.01,
    "decay": 0.9999,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1.0e-8,
    "iterations": 2000,
    "report_every": 10,
    "init_seed": 0,
    "angle_init_start": -math.pi,
    "angle_init_stop": math.pi,
    "sigma_init_low": 0.0,
    "sigma_init_high": 2.0,
    "lambda_init_low": 0.0,
    "lambda_init_high": 2.0,
    "preview_elements": 200,
    "out_dir": None,
    "out_json": None,
    "out_report": None,
    "out_history": None,
    "out_checkpoint": None,
    "out_config": None,
}


@dataclass
class Config:
    case_name: str
    global_qubits: int
    local_qubits: int
    j_coupling: float
    kappa: float
    layers: int
    gate_max_bond: int | None
    gate_cutoff: float
    apply_max_bond: int
    apply_cutoff: float
    apply_no_compress: bool
    learning_rate: float
    decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    iterations: int
    report_every: int
    init_seed: int
    angle_init_start: float
    angle_init_stop: float
    sigma_init_low: float
    sigma_init_high: float
    lambda_init_low: float
    lambda_init_high: float
    preview_elements: int
    out_dir: str | None
    out_json: str | None
    out_report: str | None
    out_history: str | None
    out_checkpoint: str | None
    out_config: str | None

    @property
    def local_dim(self) -> int:
        return 2**self.local_qubits

    @property
    def global_dim(self) -> int:
        return 2**self.global_qubits


def make_config(args) -> Config:
    merged = merge_section_config(DEFAULT_CONFIG, args.config, "optimize", args.case)

    overrides = {
        "iterations": args.iterations,
        "report_every": args.report_every,
        "init_seed": args.init_seed,
        "out_dir": args.out_dir,
        "out_json": args.out_json,
        "out_report": args.out_report,
        "out_history": args.out_history,
        "out_checkpoint": args.out_checkpoint,
        "out_config": args.out_config,
    }
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value

    cfg = Config(
        case_name=str(args.case),
        global_qubits=int(merged["global_qubits"]),
        local_qubits=int(merged["local_qubits"]),
        j_coupling=float(merged["j_coupling"]),
        kappa=float(merged["kappa"]),
        layers=int(merged["layers"]),
        gate_max_bond=None if merged.get("gate_max_bond") is None else int(merged["gate_max_bond"]),
        gate_cutoff=float(merged["gate_cutoff"]),
        apply_max_bond=int(merged["apply_max_bond"]),
        apply_cutoff=float(merged["apply_cutoff"]),
        apply_no_compress=bool(merged["apply_no_compress"]),
        learning_rate=float(merged["learning_rate"]),
        decay=float(merged["decay"]),
        adam_beta1=float(merged["adam_beta1"]),
        adam_beta2=float(merged["adam_beta2"]),
        adam_epsilon=float(merged["adam_epsilon"]),
        iterations=int(merged["iterations"]),
        report_every=int(merged["report_every"]),
        init_seed=int(merged["init_seed"]),
        angle_init_start=float(merged["angle_init_start"]),
        angle_init_stop=float(merged["angle_init_stop"]),
        sigma_init_low=float(merged["sigma_init_low"]),
        sigma_init_high=float(merged["sigma_init_high"]),
        lambda_init_low=float(merged["lambda_init_low"]),
        lambda_init_high=float(merged["lambda_init_high"]),
        preview_elements=int(merged["preview_elements"]),
        out_dir=merged.get("out_dir"),
        out_json=merged.get("out_json"),
        out_report=merged.get("out_report"),
        out_history=merged.get("out_history"),
        out_checkpoint=merged.get("out_checkpoint"),
        out_config=merged.get("out_config"),
    )

    if cfg.global_qubits != cfg.local_qubits + 1:
        raise ValueError("This 2x2 MPS comparison expects global_qubits = local_qubits + 1.")
    return cfg


def resolve_output_paths(cfg: Config) -> dict[str, Path]:
    if cfg.out_dir is not None:
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return {
            "json": Path(cfg.out_json) if cfg.out_json is not None else out_dir / "optimize.json",
            "report": Path(cfg.out_report) if cfg.out_report is not None else out_dir / "optimize_report.md",
            "history": Path(cfg.out_history) if cfg.out_history is not None else out_dir / "optimize_metrics.jsonl",
            "checkpoint": (
                Path(cfg.out_checkpoint)
                if cfg.out_checkpoint is not None
                else out_dir / "optimize_checkpoint.json"
            ),
            "config": Path(cfg.out_config) if cfg.out_config is not None else out_dir / "optimize_config_used.yaml",
        }

    base = THIS_DIR / f"formal_compare_mps_{cfg.case_name}"
    return {
        "json": Path(cfg.out_json) if cfg.out_json is not None else base.with_suffix(".json"),
        "report": Path(cfg.out_report) if cfg.out_report is not None else base.with_name(base.name + "_report").with_suffix(".md"),
        "history": Path(cfg.out_history) if cfg.out_history is not None else base.with_name(base.name + "_metrics").with_suffix(".jsonl"),
        "checkpoint": (
            Path(cfg.out_checkpoint)
            if cfg.out_checkpoint is not None
            else base.with_name(base.name + "_checkpoint").with_suffix(".json")
        ),
        "config": Path(cfg.out_config) if cfg.out_config is not None else base.with_name(base.name + "_config_used").with_suffix(".yaml"),
    }


def ensure_output_dirs(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def wrap_angles_numpy(array: np.ndarray) -> np.ndarray:
    return ((array + math.pi) % (2.0 * math.pi)) - math.pi


def wrap_angles_jax(array):
    import jax.numpy as jnp

    return ((array + math.pi) % (2.0 * math.pi)) - math.pi


def apply_block_mpo(block, state, cfg: Config):
    apply_kwargs = {
        "contract": True,
        "compress": not cfg.apply_no_compress,
    }
    if apply_kwargs["compress"]:
        apply_kwargs["max_bond"] = cfg.apply_max_bond
        apply_kwargs["cutoff"] = cfg.apply_cutoff
    return block.apply(state, **apply_kwargs)


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


def build_direct_problem(cfg: Config) -> dict[str, Any]:
    import quimb.tensor as qtn

    eta, zeta = estimate_scaled_spectrum(cfg.global_qubits, cfg.j_coupling, cfg.kappa)
    local_base = build_base_local_mpo(cfg.local_qubits, cfg.j_coupling)
    boundary_z = build_boundary_z_mpo(cfg.local_qubits, cfg.j_coupling)

    if abs(cfg.j_coupling) <= 1.0e-15:
        a11 = scale_and_add_identity(
            local_base.multiply(1.0 / zeta, inplace=False),
            eta / zeta,
            cfg.local_qubits,
        )
        a22 = a11.copy()
    else:
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
    ident_local = qtn.MPO_identity(cfg.local_qubits).multiply(1.0 / zeta, inplace=False)
    blocks = ((a11, ident_local.copy()), (ident_local.copy(), a22))

    b_state = qtn.MPS_computational_state("+" * cfg.local_qubits, dtype="float64")
    b_norm = 0.5 / math.sqrt(2.0)
    b_states = ((b_state.copy(), b_state.copy()), (b_state.copy(), b_state.copy()))
    b_norms = np.full((2, 2), b_norm, dtype=np.float64)
    b_row_state = b_state.copy()
    b_row_norm = math.sqrt(2.0) * b_norm

    return {
        "blocks": blocks,
        "b_states": b_states,
        "b_norms": b_norms,
        "b_row_state": b_row_state,
        "b_row_norm": float(b_row_norm),
        "eta": float(eta),
        "zeta": float(zeta),
        "column_mix": np.asarray([[2.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0]], dtype=np.float64),
        "row_coeffs": np.asarray([[-1.0, 1.0], [1.0, -1.0]], dtype=np.float64),
    }


def build_circuit_numpy(n: int, angles: np.ndarray, cfg: Config):
    import quimb.tensor as qtn

    if angles.shape != (cfg.layers, n):
        raise ValueError(f"Expected angle tensor {(cfg.layers, n)}, got {angles.shape}.")

    circ = qtn.CircuitMPS(n, cutoff=cfg.gate_cutoff, max_bond=cfg.gate_max_bond)
    for layer in range(cfg.layers):
        for wire in range(n):
            circ.ry(float(angles[layer, wire]), wire)
        if n == 2:
            circ.cnot(0, 1)
        elif n > 2:
            for wire in range(n):
                circ.cnot(wire, (wire + 1) % n)
    return circ


def build_circuit_jax(n: int, angles, cfg: Config):
    import quimb.tensor as qtn

    circ = qtn.CircuitMPS(n, cutoff=cfg.gate_cutoff, max_bond=cfg.gate_max_bond)
    for layer in range(cfg.layers):
        for wire in range(n):
            circ.ry(angles[layer, wire], wire)
        if n == 2:
            circ.cnot(0, 1)
        elif n > 2:
            for wire in range(n):
                circ.cnot(wire, (wire + 1) % n)
    return circ


def make_initial_parameters(cfg: Config) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.init_seed)
    alpha = rng.uniform(
        cfg.angle_init_start,
        cfg.angle_init_stop,
        size=(2, 2, cfg.layers, cfg.local_qubits),
    ).astype(np.float64)
    beta = rng.uniform(
        cfg.angle_init_start,
        cfg.angle_init_stop,
        size=(2, 2, cfg.layers, cfg.local_qubits),
    ).astype(np.float64)
    sigma = rng.uniform(
        cfg.sigma_init_low,
        cfg.sigma_init_high,
        size=(2, 2),
    ).astype(np.float64)
    lam = rng.uniform(
        cfg.lambda_init_low,
        cfg.lambda_init_high,
        size=(2, 2),
    ).astype(np.float64)

    return {
        "alpha": wrap_angles_numpy(alpha),
        "beta": wrap_angles_numpy(beta),
        "sigma": sigma,
        "lambda": lam,
    }


def to_jax_params(params_np: dict[str, np.ndarray]):
    import jax.numpy as jnp

    return {
        "alpha": jnp.asarray(params_np["alpha"], dtype=jnp.float64),
        "beta": jnp.asarray(params_np["beta"], dtype=jnp.float64),
        "sigma": jnp.asarray(params_np["sigma"], dtype=jnp.float64),
        "lambda": jnp.asarray(params_np["lambda"], dtype=jnp.float64),
    }


def to_numpy_params(params_jax: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "alpha": np.asarray(params_jax["alpha"], dtype=np.float64),
        "beta": np.asarray(params_jax["beta"], dtype=np.float64),
        "sigma": np.asarray(params_jax["sigma"], dtype=np.float64),
        "lambda": np.asarray(params_jax["lambda"], dtype=np.float64),
    }


def to_jax_problem(problem_np: dict[str, Any]):
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

    b_row_state = problem_np["b_row_state"].copy()
    b_row_state.apply_to_arrays(jnp.asarray)

    return {
        "blocks": tuple(blocks),
        "b_states": tuple(b_states),
        "b_norms": jnp.asarray(problem_np["b_norms"], dtype=jnp.float64),
        "b_row_state": b_row_state,
        "b_row_norm": jnp.asarray(problem_np["b_row_norm"], dtype=jnp.float64),
        "column_mix": jnp.asarray(problem_np["column_mix"], dtype=jnp.float64),
        "row_coeffs": jnp.asarray(problem_np["row_coeffs"], dtype=jnp.float64),
    }


def local_loss_numpy(i: int, j: int, params: dict[str, np.ndarray], cfg: Config, problem_np: dict[str, Any]) -> float:
    sigma = float(params["sigma"][i, j])
    lam_self = float(params["lambda"][i, j])
    lam_neigh = float(params["lambda"][i, 1 - j])
    coeffs = problem_np["row_coeffs"][j]
    t = coeffs * np.asarray([lam_self, lam_neigh], dtype=np.float64)

    x_state = build_circuit_numpy(cfg.local_qubits, params["alpha"][i, j], cfg).psi
    z_self = build_circuit_numpy(cfg.local_qubits, params["beta"][i, j], cfg).psi
    z_neigh = build_circuit_numpy(cfg.local_qubits, params["beta"][i, 1 - j], cfg).psi

    b_state = problem_np["b_states"][i][j]
    b_norm = float(problem_np["b_norms"][i, j])
    ax_state = apply_block_mpo(problem_np["blocks"][i][j], x_state, cfg)

    beta_re = float(np.real(ax_state.overlap(ax_state)))
    tau_re = float(np.real(b_state.overlap(ax_state)))
    zeta_vec = np.asarray(
        [
            float(np.real(ax_state.overlap(z_self))),
            float(np.real(ax_state.overlap(z_neigh))),
        ],
        dtype=np.float64,
    )
    delta_vec = np.asarray(
        [
            float(np.real(b_state.overlap(z_self))),
            float(np.real(b_state.overlap(z_neigh))),
        ],
        dtype=np.float64,
    )
    omega_offdiag = float(np.real(z_self.overlap(z_neigh)))

    s_norm_sq = sigma * sigma * beta_re
    s_norm_sq += float(np.sum(t * t))
    s_norm_sq += 2.0 * sigma * float(np.sum(t * zeta_vec))
    s_norm_sq += 2.0 * t[0] * t[1] * omega_offdiag

    overlap_s_b = sigma * tau_re + float(np.sum(t * delta_vec))
    return float(s_norm_sq + b_norm * b_norm - 2.0 * b_norm * overlap_s_b)


def global_cost_numpy(params: dict[str, np.ndarray], cfg: Config, problem_np: dict[str, Any]) -> float:
    total = 0.0
    for i in range(2):
        for j in range(2):
            total += local_loss_numpy(i, j, params, cfg, problem_np)
    return float(total)


def local_loss_jax(i: int, j: int, params, cfg: Config, problem_jax: dict[str, Any]):
    import jax.numpy as jnp

    sigma = params["sigma"][i, j]
    lam_self = params["lambda"][i, j]
    lam_neigh = params["lambda"][i, 1 - j]
    t = problem_jax["row_coeffs"][j] * jnp.asarray([lam_self, lam_neigh], dtype=jnp.float64)

    x_state = build_circuit_jax(cfg.local_qubits, params["alpha"][i, j], cfg).psi
    z_self = build_circuit_jax(cfg.local_qubits, params["beta"][i, j], cfg).psi
    z_neigh = build_circuit_jax(cfg.local_qubits, params["beta"][i, 1 - j], cfg).psi

    b_state = problem_jax["b_states"][i][j]
    b_norm = problem_jax["b_norms"][i, j]
    ax_state = apply_block_mpo(problem_jax["blocks"][i][j], x_state, cfg)

    beta_re = jnp.real(ax_state.overlap(ax_state))
    tau_re = jnp.real(b_state.overlap(ax_state))
    zeta_vec = jnp.asarray(
        [
            jnp.real(ax_state.overlap(z_self)),
            jnp.real(ax_state.overlap(z_neigh)),
        ],
        dtype=jnp.float64,
    )
    delta_vec = jnp.asarray(
        [
            jnp.real(b_state.overlap(z_self)),
            jnp.real(b_state.overlap(z_neigh)),
        ],
        dtype=jnp.float64,
    )
    omega_offdiag = jnp.real(z_self.overlap(z_neigh))

    s_norm_sq = sigma * sigma * beta_re
    s_norm_sq = s_norm_sq + jnp.sum(t * t)
    s_norm_sq = s_norm_sq + 2.0 * sigma * jnp.sum(t * zeta_vec)
    s_norm_sq = s_norm_sq + 2.0 * t[0] * t[1] * omega_offdiag

    overlap_s_b = sigma * tau_re + jnp.sum(t * delta_vec)
    return s_norm_sq + b_norm * b_norm - 2.0 * b_norm * overlap_s_b


def global_cost_jax(params, cfg: Config, problem_jax: dict[str, Any]):
    import jax.numpy as jnp

    total = jnp.asarray(0.0, dtype=jnp.float64)
    for i in range(2):
        for j in range(2):
            total = total + local_loss_jax(i, j, params, cfg, problem_jax)
    return total


def consensus_mix(params, column_mix):
    import jax.numpy as jnp

    mixed_alpha = jnp.einsum("rk,kjln->rjln", column_mix, params["alpha"])
    mixed_sigma = jnp.einsum("rk,kj->rj", column_mix, params["sigma"])
    return {
        "alpha": mixed_alpha,
        "sigma": mixed_sigma,
        "beta": params["beta"],
        "lambda": params["lambda"],
    }


def lr_at_step(step: int, cfg: Config) -> float:
    return cfg.learning_rate * (cfg.decay ** max(step - 1, 0))


def adam_zero_like(params):
    import jax.numpy as jnp

    return {key: jnp.zeros_like(value) for key, value in params.items()}


def grad_tracker_init(grads):
    return {
        "alpha": grads["alpha"],
        "sigma": grads["sigma"],
        "beta": grads["beta"],
        "lambda": grads["lambda"],
    }


def update_gradient_tracker(tracker, grads, prev_grads, column_mix):
    import jax.numpy as jnp

    mixed_alpha = jnp.einsum("rk,kjln->rjln", column_mix, tracker["alpha"])
    mixed_sigma = jnp.einsum("rk,kj->rj", column_mix, tracker["sigma"])
    return {
        "alpha": mixed_alpha + grads["alpha"] - prev_grads["alpha"],
        "sigma": mixed_sigma + grads["sigma"] - prev_grads["sigma"],
        "beta": grads["beta"],
        "lambda": grads["lambda"],
    }


def adam_apply(params, tracker, moments1, moments2, step: int, cfg: Config):
    import jax.numpy as jnp

    lr = lr_at_step(step, cfg)
    new_m1 = {}
    new_m2 = {}
    new_params = {}

    for key in ("alpha", "beta", "sigma", "lambda"):
        grad = tracker[key]
        m1 = cfg.adam_beta1 * moments1[key] + (1.0 - cfg.adam_beta1) * grad
        m2 = cfg.adam_beta2 * moments2[key] + (1.0 - cfg.adam_beta2) * (grad * grad)
        m1_hat = m1 / (1.0 - cfg.adam_beta1**step)
        m2_hat = m2 / (1.0 - cfg.adam_beta2**step)
        update = lr * m1_hat / (jnp.sqrt(m2_hat) + cfg.adam_epsilon)
        new_value = params[key] - update
        if key in ("alpha", "beta"):
            new_value = wrap_angles_jax(new_value)
        new_params[key] = new_value
        new_m1[key] = m1
        new_m2[key] = m2

    return new_params, new_m1, new_m2


def initialize_state(cfg: Config, params_np: dict[str, np.ndarray], grad_fn):
    params = to_jax_params(params_np)
    grads = grad_fn(params)
    tracker = grad_tracker_init(grads)
    return {
        "step": 0,
        "params": params,
        "tracker": tracker,
        "prev_grads": grads,
        "moments1": adam_zero_like(params),
        "moments2": adam_zero_like(params),
    }


def distributed_iteration(state: dict[str, Any], cfg: Config, problem_jax: dict[str, Any], value_and_grad_fn):
    step = int(state["step"]) + 1
    mixed_params = consensus_mix(state["params"], problem_jax["column_mix"])
    new_params, new_m1, new_m2 = adam_apply(
        mixed_params,
        state["tracker"],
        state["moments1"],
        state["moments2"],
        step,
        cfg,
    )
    current_cost, current_grads = value_and_grad_fn(new_params)
    new_tracker = update_gradient_tracker(
        state["tracker"],
        current_grads,
        state["prev_grads"],
        problem_jax["column_mix"],
    )

    diagnostics = {
        "step": step,
        "current_cost": current_cost,
        "alpha_grad_l2": current_grads["alpha"],
        "beta_grad_l2": current_grads["beta"],
        "sigma_grad_l2": current_grads["sigma"],
        "lambda_grad_l2": current_grads["lambda"],
        "learning_rate": lr_at_step(step, cfg),
    }

    return {
        "step": step,
        "params": new_params,
        "tracker": new_tracker,
        "prev_grads": current_grads,
        "moments1": new_m1,
        "moments2": new_m2,
    }, diagnostics


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


def reconstruct_diagnostics(params_np: dict[str, np.ndarray], cfg: Config, problem_np: dict[str, Any]) -> dict[str, Any]:
    x_states = [[None, None], [None, None]]
    row_copy_residual_norms = []
    row_action_residual_norms = []

    for i in range(2):
        for j in range(2):
            x_states[i][j] = build_circuit_numpy(cfg.local_qubits, params_np["alpha"][i, j], cfg).psi

    x1_state = x_states[0][0].multiply(float(params_np["sigma"][0, 0]), inplace=False).add_MPS(
        x_states[1][0].multiply(float(params_np["sigma"][1, 0]), inplace=False),
        inplace=False,
        compress=False,
    ).multiply(0.5, inplace=False)
    x2_state = x_states[0][1].multiply(float(params_np["sigma"][0, 1]), inplace=False).add_MPS(
        x_states[1][1].multiply(float(params_np["sigma"][1, 1]), inplace=False),
        inplace=False,
        compress=False,
    ).multiply(0.5, inplace=False)

    row_action_prefixes = []
    row_copy_action_prefixes = []
    for i in range(2):
        row_copy_action = None
        for j in range(2):
            ax_state = apply_block_mpo(problem_np["blocks"][i][j], x_states[i][j], cfg)
            scaled_ax = ax_state.multiply(float(params_np["sigma"][i, j]), inplace=False)
            row_copy_action = scaled_ax if row_copy_action is None else row_copy_action.add_MPS(
                scaled_ax,
                inplace=False,
                compress=False,
            )
        row_copy_action_prefixes.append(first_n_amplitudes(row_copy_action, cfg.local_qubits, cfg.preview_elements))
        row_copy_residual = row_copy_action.add_MPS(
            problem_np["b_row_state"].multiply(-problem_np["b_row_norm"], inplace=False),
            inplace=False,
            compress=False,
        )
        row_copy_residual_norms.append(
            math.sqrt(max(float(np.real(row_copy_residual.overlap(row_copy_residual))), 0.0))
        )

        row_action = apply_block_mpo(problem_np["blocks"][i][0], x1_state, cfg)
        row_action = row_action.add_MPS(
            apply_block_mpo(problem_np["blocks"][i][1], x2_state, cfg),
            inplace=False,
            compress=False,
        )
        row_action_prefixes.append(first_n_amplitudes(row_action, cfg.local_qubits, cfg.preview_elements))
        row_residual = row_action.add_MPS(
            problem_np["b_row_state"].multiply(-problem_np["b_row_norm"], inplace=False),
            inplace=False,
            compress=False,
        )
        row_action_residual_norms.append(
            math.sqrt(max(float(np.real(row_residual.overlap(row_residual))), 0.0))
        )

    x1_prefix = 0.5 * (
        float(params_np["sigma"][0, 0]) * first_n_amplitudes(x_states[0][0], cfg.local_qubits, cfg.preview_elements)
        + float(params_np["sigma"][1, 0]) * first_n_amplitudes(x_states[1][0], cfg.local_qubits, cfg.preview_elements)
    )
    x2_prefix = 0.5 * (
        float(params_np["sigma"][0, 1]) * first_n_amplitudes(x_states[0][1], cfg.local_qubits, cfg.preview_elements)
        + float(params_np["sigma"][1, 1]) * first_n_amplitudes(x_states[1][1], cfg.local_qubits, cfg.preview_elements)
    )
    x_prefix = stack_prefix(x1_prefix, x2_prefix, cfg.preview_elements, cfg.local_dim)

    return {
        "row_action_residual_norms": row_action_residual_norms,
        "row_copy_residual_norms": row_copy_residual_norms,
        "global_residual_l2": float(math.sqrt(sum(v * v for v in row_action_residual_norms))),
        "reconstructed_x_preview": format_array_preview(x_prefix, max_elements=cfg.preview_elements),
        "row_action_previews": [
            format_array_preview(np.asarray(v), max_elements=cfg.preview_elements)
            for v in row_action_prefixes
        ],
        "row_copy_action_previews": [
            format_array_preview(np.asarray(v), max_elements=cfg.preview_elements)
            for v in row_copy_action_prefixes
        ],
    }


def consensus_error_numpy(params_np: dict[str, np.ndarray], cfg: Config) -> float:
    states = [[None, None], [None, None]]
    total = 0.0
    for i in range(2):
        for j in range(2):
            states[i][j] = build_circuit_numpy(cfg.local_qubits, params_np["alpha"][i, j], cfg).psi
    for j in range(2):
        left = np.asarray(first_n_amplitudes(states[0][j], cfg.local_qubits, cfg.local_dim), dtype=np.complex128)
        right = np.asarray(first_n_amplitudes(states[1][j], cfg.local_qubits, cfg.local_dim), dtype=np.complex128)
        diff = float(params_np["sigma"][0, j]) * left - float(params_np["sigma"][1, j]) * right
        total += float(np.vdot(diff, diff).real)
    return float(math.sqrt(max(total, 0.0)))


def compute_metrics(params_np: dict[str, np.ndarray], cfg: Config, problem_np: dict[str, Any]) -> dict[str, Any]:
    diag = reconstruct_diagnostics(params_np, cfg, problem_np)
    return {
        "global_cost": float(global_cost_numpy(params_np, cfg, problem_np)),
        "global_residual_l2": float(diag["global_residual_l2"]),
        "consensus_error_l2": float(consensus_error_numpy(params_np, cfg)),
    }


def gradient_norms(grads_np: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "alpha_grad_l2": float(np.linalg.norm(grads_np["alpha"])),
        "beta_grad_l2": float(np.linalg.norm(grads_np["beta"])),
        "sigma_grad_l2": float(np.linalg.norm(grads_np["sigma"])),
        "lambda_grad_l2": float(np.linalg.norm(grads_np["lambda"])),
    }


def metrics_entry(iteration: int, metrics: dict[str, Any], grad_norm_info: dict[str, float] | None, *, learning_rate: float | None = None) -> dict[str, Any]:
    payload = {
        "iteration": int(iteration),
        "global_cost": float(metrics["global_cost"]),
        "global_residual_l2": float(metrics["global_residual_l2"]),
        "consensus_error_l2": float(metrics["consensus_error_l2"]),
    }
    if grad_norm_info is not None:
        payload.update(grad_norm_info)
    if learning_rate is not None:
        payload["learning_rate"] = float(learning_rate)
    return payload


def checkpoint_payload(
    iteration: int,
    metrics: dict[str, Any],
    params_np: dict[str, np.ndarray],
    *,
    state_np: dict[str, Any] | None = None,
    failed: bool = False,
    error_message: str | None = None,
) -> dict[str, Any]:
    payload = {
        "iteration": int(iteration),
        "latest_metrics": sanitize_jsonable(metrics),
        "params": {
            "alpha": encode_array(params_np["alpha"]),
            "beta": encode_array(params_np["beta"]),
            "sigma": encode_array(params_np["sigma"]),
            "lambda": encode_array(params_np["lambda"]),
        },
        "failed": bool(failed),
        "error_message": error_message,
    }
    if state_np is not None:
        payload["optimizer_state"] = sanitize_jsonable(state_np)
    return payload


def state_to_numpy_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": int(state["step"]),
        "params": {key: encode_array(np.asarray(value)) for key, value in state["params"].items()},
        "tracker": {key: encode_array(np.asarray(value)) for key, value in state["tracker"].items()},
        "prev_grads": {key: encode_array(np.asarray(value)) for key, value in state["prev_grads"].items()},
        "moments1": {key: encode_array(np.asarray(value)) for key, value in state["moments1"].items()},
        "moments2": {key: encode_array(np.asarray(value)) for key, value in state["moments2"].items()},
    }


def write_report(report_path: Path, result: dict[str, Any]) -> None:
    final = result["history"][-1]
    lines = [
        "# Formal vs MPS Apples-to-Apples Comparison",
        "",
        "## Setup",
        f"- Case: `{result['case']}`",
        f"- Global qubits: `{result['problem']['global_qubits']}`",
        f"- Local qubits: `{result['problem']['local_qubits']}`",
        f"- Ansatz: `BasicEntangler-style (RY + ring CNOT)`",
        f"- Layers: `{result['config']['layers']}`",
        f"- Coupling `J`: `{result['problem']['j_coupling']}`",
        f"- Learning rate init: `{result['config']['learning_rate']}`",
        f"- Decay: `{result['config']['decay']}`",
        f"- Iterations completed: `{result['optimization']['iterations_completed']}` / `{result['optimization']['iterations_requested']}`",
        f"- Metropolis column mixing: `{result['problem']['column_mix']}`",
        f"- Local row coefficients: `{result['problem']['row_coeffs']}`",
        "",
        "## Final Metrics",
        f"- Global cost: `{final['global_cost']:.12g}`",
        f"- Global residual: `{final['global_residual_l2']:.12g}`",
        f"- Consensus error: `{final['consensus_error_l2']:.12g}`",
        "",
        "## Final Parameters",
        "### sigma",
        "```text",
        format_array_preview(np.asarray(result["final_state"]["sigma"]), max_elements=result["config"]["preview_elements"]),
        "```",
        "",
        "### lambda",
        "```text",
        format_array_preview(np.asarray(result["final_state"]["lambda"]), max_elements=result["config"]["preview_elements"]),
        "```",
        "",
        "## Reconstruction",
        "### reconstructed x",
        "```text",
        result["final_diagnostics"]["reconstructed_x_preview"],
        "```",
        "",
        "### row action 1",
        "```text",
        result["final_diagnostics"]["row_action_previews"][0],
        "```",
        "",
        "### row action 2",
        "```text",
        result["final_diagnostics"]["row_action_previews"][1],
        "```",
        "",
        "### row copy action 1",
        "```text",
        result["final_diagnostics"]["row_copy_action_previews"][0],
        "```",
        "",
        "### row copy action 2",
        "```text",
        result["final_diagnostics"]["row_copy_action_previews"][1],
        "```",
        "",
        "## Artifacts",
        f"- JSON: `{result['artifacts']['json']}`",
        f"- Report: `{result['artifacts']['report']}`",
        f"- History: `{result['artifacts']['history']}`",
        f"- Checkpoint: `{result['artifacts']['checkpoint']}`",
        f"- Config: `{result['artifacts']['config']}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
