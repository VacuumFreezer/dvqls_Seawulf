#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import eigsh, spsolve


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
    "local_qubits": 13,
    "j_coupling": 0.1,
    "kappa": 20.0,
    "ansatz": "basic_entangler_ry",
    "layers": 3,
    "forward_mode": "dense_statevector",
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
    ansatz: str
    forward_mode: str
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
        ansatz=str(merged["ansatz"]),
        forward_mode=str(merged["forward_mode"]),
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

    if cfg.global_qubits != cfg.local_qubits:
        raise ValueError("This comparison is single-agent only, so global_qubits must equal local_qubits.")
    if cfg.ansatz != "basic_entangler_ry":
        raise ValueError(f"Unsupported ansatz: {cfg.ansatz}")
    if cfg.forward_mode not in {"dense_statevector", "mps_overlap"}:
        raise ValueError(f"Unsupported forward_mode: {cfg.forward_mode}")
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

    base = THIS_DIR / f"formal_single_agent_{cfg.case_name}"
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


def wrap_angles_numpy(angles: np.ndarray) -> np.ndarray:
    return ((angles + math.pi) % (2.0 * math.pi)) - math.pi


def wrap_angles_jax(angles):
    import jax.numpy as jnp

    return ((angles + math.pi) % (2.0 * math.pi)) - math.pi


def complex_dtype(xp):
    return getattr(xp, "complex128", np.complex128)


def apply_single_qubit_gate(state, gate, wire: int, n_qubits: int, xp):
    reshaped = xp.reshape(state, (2,) * n_qubits)
    moved = xp.moveaxis(reshaped, wire, 0)
    merged = xp.reshape(moved, (2, -1))
    updated = gate @ merged
    restored = xp.reshape(updated, (2,) + (2,) * (n_qubits - 1))
    restored = xp.moveaxis(restored, 0, wire)
    return xp.reshape(restored, (-1,))


def apply_two_qubit_gate(state, gate, left: int, right: int, n_qubits: int, xp):
    reshaped = xp.reshape(state, (2,) * n_qubits)
    moved = xp.moveaxis(reshaped, (left, right), (0, 1))
    merged = xp.reshape(moved, (4, -1))
    updated = gate @ merged
    restored = xp.reshape(updated, (2, 2) + (2,) * (n_qubits - 2))
    restored = xp.moveaxis(restored, (0, 1), (left, right))
    return xp.reshape(restored, (-1,))


def ry_gate(theta, xp):
    half = theta / 2.0
    c = xp.cos(half)
    s = xp.sin(half)
    return xp.asarray([[c, -s], [s, c]], dtype=complex_dtype(xp))


def cnot_gate(xp):
    return xp.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=complex_dtype(xp),
    )


def circuit_state_dense_numpy(alpha_angles: np.ndarray, cfg: Config) -> np.ndarray:
    angles = np.asarray(alpha_angles, dtype=np.float64)
    if angles.shape != (cfg.layers, cfg.local_qubits):
        raise ValueError(f"Expected angle tensor {(cfg.layers, cfg.local_qubits)}, got {angles.shape}.")

    state = np.zeros((cfg.global_dim,), dtype=np.complex128)
    state[0] = 1.0
    cnot = cnot_gate(np)

    for layer in range(cfg.layers):
        for wire in range(cfg.local_qubits):
            state = apply_single_qubit_gate(state, ry_gate(angles[layer, wire], np), wire, cfg.local_qubits, np)
        if cfg.local_qubits == 2:
            state = apply_two_qubit_gate(state, cnot, 0, 1, cfg.local_qubits, np)
        elif cfg.local_qubits > 2:
            for wire in range(cfg.local_qubits):
                state = apply_two_qubit_gate(state, cnot, wire, (wire + 1) % cfg.local_qubits, cfg.local_qubits, np)
    return state


def circuit_state_dense_jax(alpha_angles, cfg: Config):
    import jax.numpy as jnp

    if alpha_angles.shape != (cfg.layers, cfg.local_qubits):
        raise ValueError(f"Expected angle tensor {(cfg.layers, cfg.local_qubits)}, got {alpha_angles.shape}.")

    state = jnp.zeros((cfg.global_dim,), dtype=jnp.complex128)
    state = state.at[0].set(1.0 + 0.0j)
    cnot = cnot_gate(jnp)

    for layer in range(cfg.layers):
        for wire in range(cfg.local_qubits):
            state = apply_single_qubit_gate(state, ry_gate(alpha_angles[layer, wire], jnp), wire, cfg.local_qubits, jnp)
        if cfg.local_qubits == 2:
            state = apply_two_qubit_gate(state, cnot, 0, 1, cfg.local_qubits, jnp)
        elif cfg.local_qubits > 2:
            for wire in range(cfg.local_qubits):
                state = apply_two_qubit_gate(state, cnot, wire, (wire + 1) % cfg.local_qubits, cfg.local_qubits, jnp)
    return state


def build_circuit_mps_numpy(alpha_angles: np.ndarray, cfg: Config):
    import quimb.tensor as qtn

    angles = np.asarray(alpha_angles, dtype=np.float64)
    if angles.shape != (cfg.layers, cfg.local_qubits):
        raise ValueError(f"Expected angle tensor {(cfg.layers, cfg.local_qubits)}, got {angles.shape}.")

    circ = qtn.CircuitMPS(cfg.local_qubits, cutoff=cfg.gate_cutoff, max_bond=cfg.gate_max_bond)
    for layer in range(cfg.layers):
        for wire in range(cfg.local_qubits):
            circ.ry(float(angles[layer, wire]), wire)
        if cfg.local_qubits == 2:
            circ.cnot(0, 1)
        elif cfg.local_qubits > 2:
            for wire in range(cfg.local_qubits):
                circ.cnot(wire, (wire + 1) % cfg.local_qubits)
    return circ


def build_circuit_mps_jax(alpha_angles, cfg: Config):
    import quimb.tensor as qtn

    circ = qtn.CircuitMPS(cfg.local_qubits, cutoff=cfg.gate_cutoff, max_bond=cfg.gate_max_bond)
    for layer in range(cfg.layers):
        for wire in range(cfg.local_qubits):
            circ.ry(alpha_angles[layer, wire], wire)
        if cfg.local_qubits == 2:
            circ.cnot(0, 1)
        elif cfg.local_qubits > 2:
            for wire in range(cfg.local_qubits):
                circ.cnot(wire, (wire + 1) % cfg.local_qubits)
    return circ


def scale_and_add_identity(mpo, coeff: float, nsites: int):
    import quimb.tensor as qtn

    ident = qtn.MPO_identity(nsites)
    return mpo.add_MPO(ident.multiply(coeff, inplace=False), inplace=False)


def build_global_sparse_problem(cfg: Config):
    import quimb as qu
    import quimb.tensor as qtn

    builder = qtn.SpinHam1D(cyclic=False)
    builder += 1.0, qu.pauli("X")
    builder += cfg.j_coupling, qu.pauli("Z"), qu.pauli("Z")
    h0_sparse = builder.build_sparse(cfg.global_qubits).tocsr()

    lambda_min = float(eigsh(h0_sparse, k=1, which="SA", return_eigenvectors=False)[0])
    lambda_max = float(eigsh(h0_sparse, k=1, which="LA", return_eigenvectors=False)[0])
    eta = (lambda_max - cfg.kappa * lambda_min) / (cfg.kappa - 1.0)
    zeta = lambda_max + eta
    a_sparse = ((h0_sparse + eta * identity(h0_sparse.shape[0], format="csr")) / zeta).tocsr()
    scaled_lambda_min = float(eigsh(a_sparse, k=1, which="SA", return_eigenvectors=False)[0])
    scaled_lambda_max = float(eigsh(a_sparse, k=1, which="LA", return_eigenvectors=False)[0])

    return {
        "h0_sparse": h0_sparse,
        "a_sparse": a_sparse,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "eta": float(eta),
        "zeta": float(zeta),
        "scaled_lambda_min": scaled_lambda_min,
        "scaled_lambda_max": scaled_lambda_max,
    }


def build_problem_numpy(cfg: Config) -> dict[str, Any]:
    import quimb as qu
    import quimb.tensor as qtn

    sparse_problem = build_global_sparse_problem(cfg)

    builder = qtn.SpinHam1D(cyclic=False)
    builder += 1.0, qu.pauli("X")
    builder += cfg.j_coupling, qu.pauli("Z"), qu.pauli("Z")
    h0_mpo = builder.build_mpo(cfg.global_qubits)
    a_mpo = scale_and_add_identity(
        h0_mpo.multiply(1.0 / sparse_problem["zeta"], inplace=False),
        sparse_problem["eta"] / sparse_problem["zeta"],
        cfg.global_qubits,
    )

    b_dense = np.full(cfg.global_dim, 1.0 / math.sqrt(cfg.global_dim), dtype=np.float64)
    b_state = qtn.MPS_computational_state("+" * cfg.global_qubits, dtype="float64")
    x_true = np.asarray(np.real_if_close(spsolve(sparse_problem["a_sparse"], b_dense), tol=1000), dtype=np.float64)

    return {
        "a_sparse": sparse_problem["a_sparse"],
        "a_mpo": a_mpo,
        "b_dense": b_dense,
        "b_state": b_state,
        "b_norm": 1.0,
        "x_true": x_true,
        **sparse_problem,
    }


def to_jax_problem(problem_np: dict[str, Any], cfg: Config):
    import jax.numpy as jnp

    if cfg.forward_mode == "dense_statevector":
        return {
            "a_dense": jnp.asarray(problem_np["a_sparse"].toarray(), dtype=jnp.complex128),
            "b_dense": jnp.asarray(problem_np["b_dense"], dtype=jnp.complex128),
        }

    a_mpo = problem_np["a_mpo"].copy()
    a_mpo.apply_to_arrays(jnp.asarray)
    b_state = problem_np["b_state"].copy()
    b_state.apply_to_arrays(jnp.asarray)

    return {
        "a_mpo": a_mpo,
        "b_state": b_state,
        "b_norm": jnp.asarray(problem_np["b_norm"], dtype=jnp.float64),
    }


def apply_global_mpo(mpo, state, cfg: Config):
    apply_kwargs = {
        "contract": True,
        "compress": not cfg.apply_no_compress,
    }
    if apply_kwargs["compress"]:
        apply_kwargs["max_bond"] = cfg.apply_max_bond
        apply_kwargs["cutoff"] = cfg.apply_cutoff
    return mpo.apply(state, **apply_kwargs)


def make_initial_parameters(cfg: Config) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.init_seed)
    return {
        "alpha": wrap_angles_numpy(
            rng.uniform(cfg.angle_init_start, cfg.angle_init_stop, size=(cfg.layers, cfg.local_qubits)).astype(np.float64)
        ),
        "beta": wrap_angles_numpy(
            rng.uniform(cfg.angle_init_start, cfg.angle_init_stop, size=(cfg.layers, cfg.local_qubits)).astype(np.float64)
        ),
        "sigma": np.asarray(rng.uniform(cfg.sigma_init_low, cfg.sigma_init_high), dtype=np.float64),
        "lambda": np.asarray(rng.uniform(cfg.lambda_init_low, cfg.lambda_init_high), dtype=np.float64),
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


def global_cost_numpy(params: dict[str, np.ndarray], cfg: Config, problem_np: dict[str, Any]) -> float:
    sigma = float(params["sigma"])
    if cfg.forward_mode == "dense_statevector":
        x = sigma * circuit_state_dense_numpy(params["alpha"], cfg)
        residual = np.asarray(problem_np["a_sparse"] @ x - problem_np["b_dense"], dtype=np.complex128)
        return float(np.real(np.vdot(residual, residual)))

    x_state = build_circuit_mps_numpy(params["alpha"], cfg).psi
    ax_state = apply_global_mpo(problem_np["a_mpo"], x_state, cfg)
    ax_norm = sigma * sigma * float(np.real(ax_state.overlap(ax_state)))
    ax_b = sigma * float(np.real(problem_np["b_state"].overlap(ax_state)))
    return float(ax_norm + problem_np["b_norm"] ** 2 - 2.0 * problem_np["b_norm"] * ax_b)


def global_cost_jax(params, cfg: Config, problem_jax: dict[str, Any]):
    import jax.numpy as jnp

    sigma = params["sigma"]
    if cfg.forward_mode == "dense_statevector":
        x = sigma * circuit_state_dense_jax(params["alpha"], cfg)
        residual = problem_jax["a_dense"] @ x - problem_jax["b_dense"]
        return jnp.real(jnp.vdot(residual, residual))

    x_state = build_circuit_mps_jax(params["alpha"], cfg).psi
    ax_state = apply_global_mpo(problem_jax["a_mpo"], x_state, cfg)
    ax_norm = sigma * sigma * jnp.real(ax_state.overlap(ax_state))
    ax_b = sigma * jnp.real(problem_jax["b_state"].overlap(ax_state))
    return ax_norm + problem_jax["b_norm"] ** 2 - 2.0 * problem_jax["b_norm"] * ax_b


def reconstruct_solution_numpy(params: dict[str, np.ndarray], cfg: Config) -> np.ndarray:
    return float(params["sigma"]) * circuit_state_dense_numpy(params["alpha"], cfg)


def compute_metrics(params_np: dict[str, np.ndarray], cfg: Config, problem_np: dict[str, Any]) -> dict[str, Any]:
    x_est = reconstruct_solution_numpy(params_np, cfg)
    residual = np.asarray(problem_np["a_sparse"] @ x_est - problem_np["b_dense"], dtype=np.complex128)
    solution_error = x_est - problem_np["x_true"]
    return {
        "global_cost": float(global_cost_numpy(params_np, cfg, problem_np)),
        "global_residual_l2": float(np.linalg.norm(residual)),
        "consensus_error_l2": 0.0,
        "solution_error_l2": float(np.linalg.norm(solution_error)),
        "relative_solution_error_l2": float(
            np.linalg.norm(solution_error) / max(np.linalg.norm(problem_np["x_true"]), 1.0e-12)
        ),
        "x_estimate": x_est,
        "residual_vector": residual,
    }


def compute_rescaling_diagnostics(
    x_estimate: np.ndarray,
    x_true: np.ndarray,
    a_sparse,
    b_dense: np.ndarray,
) -> dict[str, Any]:
    x_est = np.asarray(x_estimate, dtype=np.complex128)
    x_ref = np.asarray(x_true, dtype=np.complex128)
    b_vec = np.asarray(b_dense, dtype=np.complex128)
    denom = complex(np.vdot(x_est, x_est))
    best_scale = 0.0j if abs(denom) <= 1.0e-15 else complex(np.vdot(x_est, x_ref) / denom)
    x_rescaled = best_scale * x_est
    return {
        "available": True,
        "best_scale_to_true_real": float(best_scale.real),
        "best_scale_to_true_imag": float(best_scale.imag),
        "raw_x_norm_l2": float(np.linalg.norm(x_est)),
        "true_x_norm_l2": float(np.linalg.norm(x_ref)),
        "rescaled_x_norm_l2": float(np.linalg.norm(x_rescaled)),
        "cosine_similarity_to_true": float(
            abs(np.vdot(x_est, x_ref)) / max(np.linalg.norm(x_est) * np.linalg.norm(x_ref), 1.0e-15)
        ),
        "rescaled_relative_solution_error_l2": float(
            np.linalg.norm(x_rescaled - x_ref) / max(np.linalg.norm(x_ref), 1.0e-12)
        ),
        "rescaled_residual_norm_l2": float(np.linalg.norm(a_sparse @ x_rescaled - b_vec)),
    }


def build_final_diagnostics(params_np: dict[str, np.ndarray], cfg: Config, problem_np: dict[str, Any]) -> dict[str, Any]:
    x_unit = circuit_state_dense_numpy(params_np["alpha"], cfg)
    x_est = float(params_np["sigma"]) * x_unit
    ax_vec = np.asarray(problem_np["a_sparse"] @ x_est, dtype=np.complex128)
    residual = np.asarray(ax_vec - problem_np["b_dense"], dtype=np.complex128)
    global_cost = float(global_cost_numpy(params_np, cfg, problem_np))
    return {
        "sigma": float(params_np["sigma"]),
        "lambda": float(params_np["lambda"]),
        "x_unit_norm_l2": float(np.linalg.norm(x_unit)),
        "x_norm_l2": float(np.linalg.norm(x_est)),
        "ax_norm_l2": float(np.linalg.norm(ax_vec)),
        "residual_norm_l2": float(np.linalg.norm(residual)),
        "residual_from_cost_l2": float(math.sqrt(max(global_cost, 0.0))),
        "alpha_preview": format_array_preview(params_np["alpha"], max_elements=cfg.preview_elements),
        "beta_preview": format_array_preview(params_np["beta"], max_elements=cfg.preview_elements),
        "x_estimate_preview": format_array_preview(x_est, max_elements=cfg.preview_elements),
        "x_true_preview": format_array_preview(problem_np["x_true"], max_elements=cfg.preview_elements),
        "ax_preview": format_array_preview(ax_vec, max_elements=cfg.preview_elements),
        "b_preview": format_array_preview(problem_np["b_dense"], max_elements=cfg.preview_elements),
        "residual_preview": format_array_preview(residual, max_elements=cfg.preview_elements),
    }


def checkpoint_payload(
    iteration: int,
    metrics: dict[str, Any],
    params_np: dict[str, np.ndarray],
    *,
    optimizer_state: dict[str, Any] | None = None,
    failed: bool = False,
    error_message: str | None = None,
) -> dict[str, Any]:
    payload = {
        "iteration": int(iteration),
        "latest_metrics": sanitize_jsonable(metrics),
        "params": {
            "alpha": encode_array(np.asarray(params_np["alpha"], dtype=np.float64)),
            "beta": encode_array(np.asarray(params_np["beta"], dtype=np.float64)),
            "sigma": float(np.asarray(params_np["sigma"], dtype=np.float64)),
            "lambda": float(np.asarray(params_np["lambda"], dtype=np.float64)),
        },
        "failed": bool(failed),
        "error_message": error_message,
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = sanitize_jsonable(optimizer_state)
    return payload


def sparse_matrix_preview(matrix, max_elements: int = 200) -> str:
    rows_needed = max(1, math.ceil(max_elements / matrix.shape[1]))
    dense_head = np.real_if_close(np.asarray(matrix[:rows_needed, :].toarray()), tol=1000)
    flat = dense_head.reshape(-1)[:max_elements]
    return (
        f"shape={matrix.shape}, showing first {len(flat)} flattened elements:\n"
        f"{format_array_preview(flat, max_elements=max_elements)}"
    )


def write_report(report_path: Path, result: dict[str, Any]) -> None:
    final = result["history"][-1]
    lines = [
        "# Formal-Style Single-Agent MPS Comparison",
        "",
        "## Setup",
        f"- Case: `{result['case']}`",
        f"- Ansatz: `{result['config']['ansatz']}`",
        f"- Layers: `{result['config']['layers']}`",
        f"- Forward mode: `{result['config']['forward_mode']}`",
        f"- Global qubits: `{result['problem']['global_qubits']}`",
        f"- Coupling `J`: `{result['problem']['j_coupling']}`",
        f"- Learning rate: `{result['config']['learning_rate']}`",
        f"- Decay: `{result['config']['decay']}`",
        f"- Iterations: `{result['optimization']['iterations_completed']}` / `{result['optimization']['iterations_requested']}`",
        f"- Init seed: `{result['config']['init_seed']}`",
        f"- Angle init range: `[{result['config']['angle_init_start']}, {result['config']['angle_init_stop']}]`",
        f"- Sigma init range: `[{result['config']['sigma_init_low']}, {result['config']['sigma_init_high']}]`",
        f"- Lambda init range: `[{result['config']['lambda_init_low']}, {result['config']['lambda_init_high']}]`",
        f"- Exact gate setting: `max_bond={result['config']['gate_max_bond']}`, `cutoff={result['config']['gate_cutoff']}`",
        "",
        "## Final Metrics",
        f"- Global cost: `{final['global_cost']:.12g}`",
        f"- Global residual: `{final['global_residual_l2']:.12g}`",
        f"- Solution L2 error: `{final['solution_error_l2']:.12g}`",
        f"- Relative solution L2 error: `{final['relative_solution_error_l2']:.12g}`",
        f"- Elapsed time: `{result['optimization']['elapsed_s']:.6f} s`",
        "",
        "## Sigma and Lambda",
        f"- Final sigma: `{result['final_diagnostics']['sigma']:.12g}`",
        f"- Final lambda placeholder: `{result['final_diagnostics']['lambda']:.12g}`",
        f"- `|| |X> ||_2`: `{result['final_diagnostics']['x_unit_norm_l2']:.12g}`",
        f"- `||x||_2`: `{result['final_diagnostics']['x_norm_l2']:.12g}`",
        f"- `||Ax||_2`: `{result['final_diagnostics']['ax_norm_l2']:.12g}`",
        f"- Residual from dense check: `{result['final_diagnostics']['residual_norm_l2']:.12g}`",
        f"- Residual from cost identity: `{result['final_diagnostics']['residual_from_cost_l2']:.12g}`",
        "",
        "## Rescaled-x Diagnostic",
        f"- Best scalar to match true `x`: `{result['rescaled_diagnostics']['best_scale_to_true_real']:.12g}{result['rescaled_diagnostics']['best_scale_to_true_imag']:+.12g}j`",
        f"- Cosine similarity to true `x`: `{result['rescaled_diagnostics']['cosine_similarity_to_true']:.12g}`",
        f"- Rescaled relative solution error: `{result['rescaled_diagnostics']['rescaled_relative_solution_error_l2']:.12g}`",
        f"- Rescaled residual norm: `{result['rescaled_diagnostics']['rescaled_residual_norm_l2']:.12g}`",
        "",
        "## Final Previews",
        "### alpha",
        "```text",
        result["final_diagnostics"]["alpha_preview"],
        "```",
        "",
        "### beta",
        "```text",
        result["final_diagnostics"]["beta_preview"],
        "```",
        "",
        "### reconstructed x",
        "```text",
        result["final_diagnostics"]["x_estimate_preview"],
        "```",
        "",
        "### true x",
        "```text",
        result["final_diagnostics"]["x_true_preview"],
        "```",
        "",
        "### A x",
        "```text",
        result["final_diagnostics"]["ax_preview"],
        "```",
        "",
        "### b",
        "```text",
        result["final_diagnostics"]["b_preview"],
        "```",
        "",
        "### A x - b",
        "```text",
        result["final_diagnostics"]["residual_preview"],
        "```",
        "",
        "## Linear System Preview",
        "```text",
        result["linear_system"]["A_sparse_preview"],
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
