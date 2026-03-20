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
    "j_coupling": 0.1,
    "kappa": 20.0,
    "layers": 2,
    "hadamard_prefix": True,
    "gate_max_bond": None,
    "gate_cutoff": 0.0,
    "apply_max_bond": 64,
    "apply_cutoff": 1.0e-10,
    "apply_no_compress": True,
    "learning_rate": 0.02,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1.0e-8,
    "iterations": 200,
    "report_every": 5,
    "init_mode": "uniform_block_norm_j0",
    "init_seed": 1234,
    "init_start": -0.1,
    "init_stop": 0.1,
    "x_scale_init": 1.0,
    "z_scale_init": 0.10,
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
    n_rows: int
    n_cols: int
    j_coupling: float
    kappa: float
    layers: int
    hadamard_prefix: bool
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
    init_mode: str
    init_seed: int
    init_start: float
    init_stop: float
    x_scale_init: float
    z_scale_init: float
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

    @property
    def local_dim(self) -> int:
        return 2**self.local_qubits

    @property
    def angle_count(self) -> int:
        return self.layers * self.local_qubits

    @property
    def param_dim(self) -> int:
        return 1 + self.angle_count

    @property
    def use_messenger(self) -> bool:
        return self.n_cols > 1

    @property
    def use_consensus(self) -> bool:
        return self.n_rows > 1


def make_config(args) -> Config:
    merged = merge_section_config(DEFAULT_CONFIG, args.config, "optimize", args.case)

    overrides = {
        "iterations": args.iterations,
        "report_every": args.report_every,
        "init_mode": args.init_mode,
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
        n_rows=int(merged["n_rows"]),
        n_cols=int(merged["n_cols"]),
        j_coupling=float(merged["j_coupling"]),
        kappa=float(merged["kappa"]),
        layers=int(merged["layers"]),
        hadamard_prefix=bool(merged["hadamard_prefix"]),
        gate_max_bond=None if merged.get("gate_max_bond") is None else int(merged["gate_max_bond"]),
        gate_cutoff=float(merged["gate_cutoff"]),
        apply_max_bond=int(merged["apply_max_bond"]),
        apply_cutoff=float(merged["apply_cutoff"]),
        apply_no_compress=bool(merged["apply_no_compress"]),
        learning_rate=float(merged["learning_rate"]),
        adam_beta1=float(merged["adam_beta1"]),
        adam_beta2=float(merged["adam_beta2"]),
        adam_epsilon=float(merged["adam_epsilon"]),
        iterations=int(merged["iterations"]),
        report_every=int(merged["report_every"]),
        init_mode=str(merged["init_mode"]),
        init_seed=int(merged["init_seed"]),
        init_start=float(merged["init_start"]),
        init_stop=float(merged["init_stop"]),
        x_scale_init=float(merged["x_scale_init"]),
        z_scale_init=float(merged["z_scale_init"]),
        preview_elements=int(merged["preview_elements"]),
        out_dir=merged.get("out_dir"),
        out_json=merged.get("out_json"),
        out_report=merged.get("out_report"),
        out_history=merged.get("out_history"),
        out_checkpoint=merged.get("out_checkpoint"),
        out_config=merged.get("out_config"),
    )

    if cfg.n_rows != cfg.n_cols:
        raise ValueError("This partition-compare workflow currently expects square agent grids.")
    if cfg.n_rows * cfg.local_dim != cfg.global_dim:
        raise ValueError(
            f"Case `{cfg.case_name}` is inconsistent: n_rows * 2^local_qubits = "
            f"{cfg.n_rows} * {cfg.local_dim} != 2^global_qubits = {cfg.global_dim}."
        )
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

    base = THIS_DIR / f"partition_compare_{cfg.case_name}"
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


def line_graph_metropolis_mix(size: int) -> np.ndarray:
    if size <= 0:
        raise ValueError("Graph size must be positive.")
    if size == 1:
        return np.asarray([[1.0]], dtype=np.float64)

    degrees = np.asarray([1] + [2] * (size - 2) + [1], dtype=np.int64)
    mix = np.zeros((size, size), dtype=np.float64)
    for idx in range(size - 1):
        weight = 1.0 / (1.0 + max(degrees[idx], degrees[idx + 1]))
        mix[idx, idx + 1] = weight
        mix[idx + 1, idx] = weight
    for idx in range(size):
        mix[idx, idx] = 1.0 - np.sum(mix[idx])
    return mix


def line_graph_laplacian(size: int) -> np.ndarray:
    if size <= 0:
        raise ValueError("Graph size must be positive.")
    if size == 1:
        return np.asarray([[0.0]], dtype=np.float64)

    laplacian = np.zeros((size, size), dtype=np.float64)
    for idx in range(size):
        degree = 0
        if idx > 0:
            laplacian[idx, idx - 1] = -1.0
            degree += 1
        if idx + 1 < size:
            laplacian[idx, idx + 1] = -1.0
            degree += 1
        laplacian[idx, idx] = float(degree)
    return laplacian


def wrap_params_numpy(params: np.ndarray) -> np.ndarray:
    wrapped = np.array(params, copy=True, dtype=np.float64)
    wrapped[..., 1:] = ((wrapped[..., 1:] + math.pi) % (2.0 * math.pi)) - math.pi
    return wrapped


def wrap_params_jax(params):
    import jax.numpy as jnp

    scales = params[..., :1]
    angles = ((params[..., 1:] + math.pi) % (2.0 * math.pi)) - math.pi
    return jnp.concatenate((scales, angles), axis=-1)


def apply_block_mpo(block, state, cfg: Config):
    apply_kwargs = {
        "contract": True,
        "compress": not cfg.apply_no_compress,
    }
    if apply_kwargs["compress"]:
        apply_kwargs["max_bond"] = cfg.apply_max_bond
        apply_kwargs["cutoff"] = cfg.apply_cutoff
    return block.apply(state, **apply_kwargs)


def matrix_to_mpo(matrix: np.ndarray, cfg: Config):
    import quimb.tensor as qtn

    return qtn.MatrixProductOperator.from_dense(
        matrix,
        dims=[2] * cfg.local_qubits,
        cutoff=cfg.apply_cutoff,
        max_bond=cfg.apply_max_bond,
    )


def make_initial_parameters(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.empty((cfg.n_rows, cfg.n_cols, cfg.param_dim), dtype=np.float64)
    beta = np.empty((cfg.n_rows, cfg.n_cols, cfg.param_dim), dtype=np.float64)

    if cfg.init_mode == "normal_angles":
        rng = np.random.default_rng(cfg.init_seed)
        alpha[..., 0] = cfg.x_scale_init
        beta[..., 0] = cfg.z_scale_init
        alpha[..., 1:] = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(cfg.n_rows, cfg.n_cols, cfg.angle_count),
        )
        beta[..., 1:] = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(cfg.n_rows, cfg.n_cols, cfg.angle_count),
        )
    elif cfg.init_mode == "uniform_block_norm_j0":
        rng = np.random.default_rng(cfg.init_seed)
        x_star_j0 = np.full(cfg.global_dim, 1.0 / math.sqrt(cfg.global_dim), dtype=np.float64)
        block_norms = np.asarray(
            [
                np.linalg.norm(x_star_j0[j * cfg.local_dim : (j + 1) * cfg.local_dim])
                for j in range(cfg.n_cols)
            ],
            dtype=np.float64,
        )
        alpha[..., 0] = block_norms[None, :]
        beta[..., 0] = block_norms[None, :]
        alpha[..., 1:] = rng.uniform(
            cfg.init_start,
            cfg.init_stop,
            size=(cfg.n_rows, cfg.n_cols, cfg.angle_count),
        )
        beta[..., 1:] = rng.uniform(
            cfg.init_start,
            cfg.init_stop,
            size=(cfg.n_rows, cfg.n_cols, cfg.angle_count),
        )
    elif cfg.init_mode == "normal_block_norm_j0":
        rng = np.random.default_rng(cfg.init_seed)
        x_star_j0 = np.full(cfg.global_dim, 1.0 / math.sqrt(cfg.global_dim), dtype=np.float64)
        block_norms = np.asarray(
            [
                np.linalg.norm(x_star_j0[j * cfg.local_dim : (j + 1) * cfg.local_dim])
                for j in range(cfg.n_cols)
            ],
            dtype=np.float64,
        )
        alpha[..., 0] = block_norms[None, :]
        beta[..., 0] = block_norms[None, :]
        alpha[..., 1:] = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(cfg.n_rows, cfg.n_cols, cfg.angle_count),
        )
        beta[..., 1:] = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(cfg.n_rows, cfg.n_cols, cfg.angle_count),
        )
    elif cfg.init_mode == "random_uniform":
        rng = np.random.default_rng(cfg.init_seed)
        alpha[..., 0] = cfg.x_scale_init
        beta[..., 0] = cfg.z_scale_init
        alpha[..., 1:] = rng.uniform(cfg.init_start, cfg.init_stop, size=(cfg.n_rows, cfg.n_cols, cfg.angle_count))
        beta[..., 1:] = rng.uniform(cfg.init_start, cfg.init_stop, size=(cfg.n_rows, cfg.n_cols, cfg.angle_count))
    elif cfg.init_mode == "structured_linspace":
        base_angles = np.linspace(cfg.init_start, cfg.init_stop, cfg.angle_count, dtype=np.float64)
        for i in range(cfg.n_rows):
            for j in range(cfg.n_cols):
                offset = 0.01 * (i + j)
                alpha[i, j, 0] = cfg.x_scale_init + 0.02 * (i - j)
                beta[i, j, 0] = cfg.z_scale_init + 0.01 * (i + j + 1)
                alpha[i, j, 1:] = base_angles + offset
                beta[i, j, 1:] = base_angles[::-1] - 0.5 * offset
    else:
        raise ValueError(f"Unsupported init_mode: {cfg.init_mode}")

    return wrap_params_numpy(alpha), wrap_params_numpy(beta)


def apply_single_qubit_gate(state, gate, wire: int, n_qubits: int, xp):
    reshaped = xp.reshape(state, (2,) * n_qubits)
    moved = xp.moveaxis(reshaped, wire, 0)
    merged = xp.reshape(moved, (2, -1))
    updated = gate @ merged
    restored = xp.reshape(updated, (2,) + (2,) * (n_qubits - 1))
    restored = xp.moveaxis(restored, 0, wire)
    return xp.reshape(restored, (-1,))


def apply_cz_gate(state, left: int, right: int, n_qubits: int, xp):
    phases = xp.asarray([1.0, 1.0, 1.0, -1.0], dtype=state.dtype)
    reshaped = xp.reshape(state, (2,) * n_qubits)
    moved = xp.moveaxis(reshaped, (left, right), (0, 1))
    merged = xp.reshape(moved, (4, -1))
    updated = phases[:, None] * merged
    restored = xp.reshape(updated, (2, 2) + (2,) * (n_qubits - 2))
    restored = xp.moveaxis(restored, (0, 1), (left, right))
    return xp.reshape(restored, (-1,))


def ry_gate(theta, xp):
    half = theta / 2.0
    c = xp.cos(half)
    s = xp.sin(half)
    return xp.asarray([[c, -s], [s, c]], dtype=xp.float64)


def hadamard_gate(xp):
    value = 1.0 / math.sqrt(2.0)
    return xp.asarray([[value, value], [value, -value]], dtype=xp.float64)


def reshape_angles(flat_angles, cfg: Config, xp):
    return xp.reshape(flat_angles, (cfg.layers, cfg.local_qubits))


def circuit_state_numpy(flat_angles, cfg: Config) -> np.ndarray:
    state = np.zeros(cfg.local_dim, dtype=np.float64)
    state[0] = 1.0
    if cfg.hadamard_prefix:
        had = hadamard_gate(np)
        for wire in range(cfg.local_qubits):
            state = apply_single_qubit_gate(state, had, wire, cfg.local_qubits, np)

    for layer_angles in reshape_angles(np.asarray(flat_angles, dtype=np.float64), cfg, np):
        for wire in range(cfg.local_qubits):
            state = apply_single_qubit_gate(state, ry_gate(layer_angles[wire], np), wire, cfg.local_qubits, np)
        for left in range(cfg.local_qubits - 1):
            state = apply_cz_gate(state, left, left + 1, cfg.local_qubits, np)
    return state


def circuit_state_jax(flat_angles, cfg: Config):
    import jax.numpy as jnp

    state = jnp.zeros((cfg.local_dim,), dtype=jnp.float64)
    state = state.at[0].set(1.0)
    if cfg.hadamard_prefix:
        had = hadamard_gate(jnp)
        for wire in range(cfg.local_qubits):
            state = apply_single_qubit_gate(state, had, wire, cfg.local_qubits, jnp)

    for layer_angles in reshape_angles(flat_angles, cfg, jnp):
        for wire in range(cfg.local_qubits):
            state = apply_single_qubit_gate(state, ry_gate(layer_angles[wire], jnp), wire, cfg.local_qubits, jnp)
        for left in range(cfg.local_qubits - 1):
            state = apply_cz_gate(state, left, left + 1, cfg.local_qubits, jnp)
    return state


def build_circuit_mps_numpy(flat_angles, cfg: Config):
    import quimb.tensor as qtn

    circ = qtn.CircuitMPS(
        cfg.local_qubits,
        cutoff=cfg.gate_cutoff,
        max_bond=cfg.gate_max_bond,
    )
    if cfg.hadamard_prefix:
        for wire in range(cfg.local_qubits):
            circ.h(wire)
    for layer_angles in reshape_angles(np.asarray(flat_angles, dtype=np.float64), cfg, np):
        for wire in range(cfg.local_qubits):
            circ.ry(float(layer_angles[wire]), wire)
        for left in range(cfg.local_qubits - 1):
            circ.cz(left, left + 1)
    return circ


def build_circuit_mps_jax(flat_angles, cfg: Config):
    import jax.numpy as jnp
    import quimb.tensor as qtn

    circ = qtn.CircuitMPS(
        cfg.local_qubits,
        cutoff=cfg.gate_cutoff,
        max_bond=cfg.gate_max_bond,
    )
    if cfg.hadamard_prefix:
        for wire in range(cfg.local_qubits):
            circ.h(wire)
    for layer_angles in reshape_angles(flat_angles, cfg, jnp):
        for wire in range(cfg.local_qubits):
            circ.ry(layer_angles[wire], wire)
        for left in range(cfg.local_qubits - 1):
            circ.cz(left, left + 1)
    return circ


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


def build_partitioned_problem_numpy(cfg: Config) -> dict[str, Any]:
    import quimb.tensor as qtn

    global_problem = build_global_sparse_problem(cfg)
    a_sparse = global_problem["a_sparse"]
    block_dim = cfg.local_dim

    blocks = []
    for row in range(cfg.n_rows):
        row_blocks = []
        row_slice = slice(row * block_dim, (row + 1) * block_dim)
        for col in range(cfg.n_cols):
            col_slice = slice(col * block_dim, (col + 1) * block_dim)
            block_dense = np.real_if_close(a_sparse[row_slice, col_slice].toarray(), tol=1000)
            row_blocks.append(matrix_to_mpo(np.asarray(block_dense, dtype=np.float64), cfg))
        blocks.append(tuple(row_blocks))

    b_dense = np.full(cfg.global_dim, 1.0 / math.sqrt(cfg.global_dim), dtype=np.float64)
    b_rows = []
    b_vectors = []
    b_states = []
    b_norms = np.zeros((cfg.n_rows, cfg.n_cols), dtype=np.float64)
    for row in range(cfg.n_rows):
        row_vec = np.array(b_dense[row * block_dim : (row + 1) * block_dim], copy=True)
        b_rows.append(row_vec)
        row_parts = []
        row_states = []
        for col in range(cfg.n_cols):
            vector = np.array(row_vec / cfg.n_cols, copy=True)
            norm = float(np.linalg.norm(vector))
            state_vector = np.zeros_like(vector)
            if norm <= 1.0e-15:
                state_vector[0] = 1.0
            else:
                state_vector = vector / norm
            row_parts.append(vector)
            row_states.append(qtn.MatrixProductState.from_dense(state_vector, dims=[2] * cfg.local_qubits))
            b_norms[row, col] = norm
        b_vectors.append(row_parts)
        b_states.append(tuple(row_states))

    column_mix = line_graph_metropolis_mix(cfg.n_rows)
    row_laplacian = line_graph_laplacian(cfg.n_cols)

    x_true = np.asarray(np.real_if_close(spsolve(a_sparse, b_dense), tol=1000), dtype=np.float64)

    return {
        "blocks": tuple(blocks),
        "a_sparse": a_sparse,
        "b_dense": b_dense,
        "b_rows": tuple(b_rows),
        "b_vectors": tuple(tuple(vec for vec in row_parts) for row_parts in b_vectors),
        "b_states": tuple(b_states),
        "b_norms": b_norms,
        "row_laplacian": row_laplacian,
        "column_mix": column_mix,
        "x_true": x_true,
        **global_problem,
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

    return {
        "blocks": tuple(blocks),
        "b_states": tuple(b_states),
        "b_norms": jnp.asarray(problem_np["b_norms"], dtype=jnp.float64),
        "row_laplacian": jnp.asarray(problem_np["row_laplacian"], dtype=jnp.float64),
        "column_mix": jnp.asarray(problem_np["column_mix"], dtype=jnp.float64),
    }


def build_state_blocks_numpy(alpha: np.ndarray, cfg: Config) -> np.ndarray:
    states = np.zeros((cfg.n_rows, cfg.n_cols, cfg.local_dim), dtype=np.float64)
    for i in range(cfg.n_rows):
        for j in range(cfg.n_cols):
            states[i, j] = float(alpha[i, j, 0]) * circuit_state_numpy(alpha[i, j, 1:], cfg)
    return states


def build_unit_state_blocks_numpy(alpha: np.ndarray, cfg: Config) -> np.ndarray:
    states = np.zeros((cfg.n_rows, cfg.n_cols, cfg.local_dim), dtype=np.float64)
    for i in range(cfg.n_rows):
        for j in range(cfg.n_cols):
            states[i, j] = circuit_state_numpy(alpha[i, j, 1:], cfg)
    return states


def global_cost_numpy(alpha: np.ndarray, beta: np.ndarray | None, cfg: Config, problem_np: dict[str, Any]) -> float:
    total = 0.0
    b_states = problem_np["b_states"]
    b_norms = problem_np["b_norms"]
    laplacian = problem_np["row_laplacian"]

    for i in range(cfg.n_rows):
        if cfg.use_messenger:
            z_states = []
            z_scales = []
            for k in range(cfg.n_cols):
                z_scales.append(float(beta[i, k, 0]))
                z_states.append(build_circuit_mps_numpy(beta[i, k, 1:], cfg).psi)

            z_overlaps = [[None for _ in range(cfg.n_cols)] for _ in range(cfg.n_cols)]
            for k in range(cfg.n_cols):
                for p in range(cfg.n_cols):
                    z_overlaps[k][p] = float(np.real(z_states[k].overlap(z_states[p])))
        else:
            z_states = []
            z_scales = []
            z_overlaps = []

        for j in range(cfg.n_cols):
            b_state = b_states[i][j]
            b_norm = float(b_norms[i, j])
            sigma = float(alpha[i, j, 0])
            x_state = build_circuit_mps_numpy(alpha[i, j, 1:], cfg).psi
            ax_state = apply_block_mpo(problem_np["blocks"][i][j], x_state, cfg)

            ax_norm = sigma * sigma * float(np.real(ax_state.overlap(ax_state)))
            ax_b = sigma * float(np.real(b_state.overlap(ax_state)))

            zz_term = 0.0
            ax_z_term = 0.0
            b_z_term = 0.0
            if cfg.use_messenger:
                for k in range(cfg.n_cols):
                    lk = float(laplacian[j, k])
                    b_z_overlap = float(np.real(b_state.overlap(z_states[k])))
                    ax_z_term += lk * z_scales[k] * float(np.real(ax_state.overlap(z_states[k])))
                    b_z_term += lk * z_scales[k] * b_z_overlap
                    for p in range(cfg.n_cols):
                        zz_term += (
                            lk
                            * float(laplacian[j, p])
                            * z_scales[k]
                            * z_scales[p]
                            * z_overlaps[k][p]
                        )

            total += (
                ax_norm
                + zz_term
                + b_norm * b_norm
                - 2.0 * sigma * ax_z_term
                - 2.0 * b_norm * ax_b
                + 2.0 * b_norm * b_z_term
            )
    return float(total)


def global_cost_jax(alpha, beta, cfg: Config, problem_jax: dict[str, Any]):
    import jax.numpy as jnp

    total = jnp.asarray(0.0, dtype=jnp.float64)
    b_states = problem_jax["b_states"]
    b_norms = problem_jax["b_norms"]
    laplacian = problem_jax["row_laplacian"]

    for i in range(cfg.n_rows):
        if cfg.use_messenger:
            z_states = []
            z_scales = []
            for k in range(cfg.n_cols):
                z_scales.append(beta[i, k, 0])
                z_states.append(build_circuit_mps_jax(beta[i, k, 1:], cfg).psi)

            z_overlaps = [[None for _ in range(cfg.n_cols)] for _ in range(cfg.n_cols)]
            for k in range(cfg.n_cols):
                for p in range(cfg.n_cols):
                    z_overlaps[k][p] = z_states[k].overlap(z_states[p])
        else:
            z_states = []
            z_scales = []
            z_overlaps = []

        for j in range(cfg.n_cols):
            b_state = b_states[i][j]
            b_norm = b_norms[i, j]
            sigma = alpha[i, j, 0]
            x_state = build_circuit_mps_jax(alpha[i, j, 1:], cfg).psi
            ax_state = apply_block_mpo(problem_jax["blocks"][i][j], x_state, cfg)

            ax_norm = sigma * sigma * jnp.real(ax_state.overlap(ax_state))
            ax_b = sigma * jnp.real(b_state.overlap(ax_state))

            zz_term = jnp.asarray(0.0, dtype=jnp.float64)
            ax_z_term = jnp.asarray(0.0, dtype=jnp.float64)
            b_z_term = jnp.asarray(0.0, dtype=jnp.float64)
            if cfg.use_messenger:
                for k in range(cfg.n_cols):
                    lk = laplacian[j, k]
                    b_z_overlap = jnp.real(b_state.overlap(z_states[k]))
                    ax_z_term = ax_z_term + lk * z_scales[k] * jnp.real(ax_state.overlap(z_states[k]))
                    b_z_term = b_z_term + lk * z_scales[k] * b_z_overlap
                    for p in range(cfg.n_cols):
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
    return total


def reconstruct_solution_numpy(alpha: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    x_states = build_state_blocks_numpy(alpha, cfg)
    averaged = np.mean(x_states, axis=0)
    x_est = np.concatenate([averaged[j] for j in range(cfg.n_cols)], axis=0)
    return x_states, x_est


def compute_metrics(alpha: np.ndarray, beta: np.ndarray, cfg: Config, problem_np: dict[str, Any]) -> dict[str, Any]:
    x_states, x_est = reconstruct_solution_numpy(alpha, cfg)
    residual = problem_np["a_sparse"] @ x_est - problem_np["b_dense"]

    consensus_sq = 0.0
    if cfg.use_consensus:
        for j in range(cfg.n_cols):
            for i in range(cfg.n_rows - 1):
                diff = x_states[i, j] - x_states[i + 1, j]
                consensus_sq += float(np.dot(diff, diff))

    solution_error = x_est - problem_np["x_true"]
    metrics = {
        "global_cost": float(global_cost_numpy(alpha, beta, cfg, problem_np)),
        "global_residual_l2": float(np.linalg.norm(residual)),
        "consensus_error_l2": float(math.sqrt(consensus_sq)),
        "solution_error_l2": float(np.linalg.norm(solution_error)),
        "relative_solution_error_l2": float(
            np.linalg.norm(solution_error) / max(np.linalg.norm(problem_np["x_true"]), 1.0e-12)
        ),
        "x_estimate": x_est,
        "x_states": x_states,
    }
    return metrics


def compute_rescaling_diagnostics(
    x_estimate: np.ndarray,
    x_true: np.ndarray,
    a_sparse,
    b_dense: np.ndarray,
) -> dict[str, Any]:
    x_est = np.asarray(x_estimate, dtype=np.float64)
    x_ref = np.asarray(x_true, dtype=np.float64)
    b_vec = np.asarray(b_dense, dtype=np.float64)
    denom = float(np.dot(x_est, x_est))
    best_scale = 0.0 if abs(denom) <= 1.0e-15 else float(np.dot(x_est, x_ref) / denom)
    x_rescaled = best_scale * x_est
    return {
        "available": True,
        "best_scale_to_true_real": float(best_scale),
        "best_scale_to_true_imag": 0.0,
        "raw_x_norm_l2": float(np.linalg.norm(x_est)),
        "true_x_norm_l2": float(np.linalg.norm(x_ref)),
        "rescaled_x_norm_l2": float(np.linalg.norm(x_rescaled)),
        "cosine_similarity_to_true": float(
            abs(np.dot(x_est, x_ref)) / max(np.linalg.norm(x_est) * np.linalg.norm(x_ref), 1.0e-15)
        ),
        "rescaled_relative_solution_error_l2": float(
            np.linalg.norm(x_rescaled - x_ref) / max(np.linalg.norm(x_ref), 1.0e-12)
        ),
        "rescaled_residual_norm_l2": float(np.linalg.norm(a_sparse @ x_rescaled - b_vec)),
    }


def build_final_diagnostics(
    alpha: np.ndarray,
    beta: np.ndarray,
    cfg: Config,
    problem_np: dict[str, Any],
) -> dict[str, Any]:
    unit_states = build_unit_state_blocks_numpy(alpha, cfg)
    x_states = build_state_blocks_numpy(alpha, cfg)
    averaged = np.mean(x_states, axis=0)
    x_est = np.concatenate([averaged[j] for j in range(cfg.n_cols)], axis=0)
    global_action = np.asarray(problem_np["a_sparse"] @ x_est, dtype=np.float64)

    row_action_vectors = []
    row_action_residual_norms = []
    for i in range(cfg.n_rows):
        row_slice = slice(i * cfg.local_dim, (i + 1) * cfg.local_dim)
        row_action = np.asarray(global_action[row_slice], dtype=np.float64)
        row_action_vectors.append(row_action)
        row_action_residual_norms.append(float(np.linalg.norm(row_action - problem_np["b_rows"][i])))

    return {
        "x_unit_state_norms": np.linalg.norm(unit_states, axis=2).tolist(),
        "x_block_norms": np.linalg.norm(x_states, axis=2).tolist(),
        "column_block_norms": np.linalg.norm(averaged, axis=1).tolist(),
        "row_copy_norms": np.linalg.norm(x_states.reshape(cfg.n_rows, -1), axis=1).tolist(),
        "row_action_residual_norms": row_action_residual_norms,
        "sigma": np.asarray(alpha[..., 0], dtype=np.float64).tolist(),
        "lambda": None if not cfg.use_messenger else np.asarray(beta[..., 0], dtype=np.float64).tolist(),
        "row_action_previews": [
            format_array_preview(np.asarray(row_action, dtype=np.float64), max_elements=cfg.preview_elements)
            for row_action in row_action_vectors
        ],
        "b_row_previews": [
            format_array_preview(np.asarray(row_vec, dtype=np.float64), max_elements=cfg.preview_elements)
            for row_vec in problem_np["b_rows"]
        ],
        "b_agent_previews": [
            [
                format_array_preview(np.asarray(agent_vec, dtype=np.float64), max_elements=cfg.preview_elements)
                for agent_vec in row_parts
            ]
            for row_parts in problem_np["b_vectors"]
        ],
    }


def mix_columns(values, column_mix):
    import jax.numpy as jnp

    return jnp.einsum("rk,kjp->rjp", column_mix, values)


def adam_learning_rate(step: int, cfg: Config) -> float:
    return cfg.learning_rate * math.sqrt(1.0 - cfg.adam_beta2**step) / (1.0 - cfg.adam_beta1**step)


def distributed_iteration(
    state: dict[str, Any],
    cfg: Config,
    problem_jax: dict[str, Any],
    full_grad_fn,
    alpha_grad_fn,
):
    import jax.numpy as jnp

    step = int(state["step"]) + 1
    if cfg.use_messenger:
        current_cost, (g_alpha_old, h_beta_old) = full_grad_fn(state["alpha"], state["beta"])
    else:
        current_cost, g_alpha_old = full_grad_fn(state["alpha"])
        h_beta_old = None
    lr_t = adam_learning_rate(step, cfg)

    if cfg.use_messenger:
        a_beta = cfg.adam_beta1 * state["a_beta"] + (1.0 - cfg.adam_beta1) * h_beta_old
        b_beta = cfg.adam_beta2 * state["b_beta"] + (1.0 - cfg.adam_beta2) * (h_beta_old * h_beta_old)
        beta_step = lr_t * a_beta / (jnp.sqrt(b_beta) + cfg.adam_epsilon)
        beta_new = wrap_params_jax(state["beta"] - beta_step)
    else:
        a_beta = state["a_beta"]
        b_beta = state["b_beta"]
        beta_step = jnp.zeros_like(state["beta"])
        beta_new = state["beta"]

    a_alpha = cfg.adam_beta1 * state["a_alpha"] + (1.0 - cfg.adam_beta1) * state["y"]
    b_alpha = cfg.adam_beta2 * state["b_alpha"] + (1.0 - cfg.adam_beta2) * (state["y"] * state["y"])
    alpha_step = lr_t * a_alpha / (jnp.sqrt(b_alpha) + cfg.adam_epsilon)
    alpha_new = wrap_params_jax(mix_columns(state["alpha"], problem_jax["column_mix"]) - alpha_step)

    if cfg.use_messenger:
        g_alpha_new = alpha_grad_fn(alpha_new, beta_new)
    else:
        g_alpha_new = alpha_grad_fn(alpha_new)
    y_new = mix_columns(state["y"], problem_jax["column_mix"]) + g_alpha_new - g_alpha_old

    diagnostics = {
        "step": step,
        "current_cost": current_cost,
        "alpha_grad_l2": jnp.linalg.norm(g_alpha_old),
        "beta_grad_l2": None if not cfg.use_messenger else jnp.linalg.norm(h_beta_old),
        "alpha_step_l2": jnp.linalg.norm(alpha_step),
        "beta_step_l2": None if not cfg.use_messenger else jnp.linalg.norm(beta_step),
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
    }, diagnostics


def initialize_state(cfg: Config, alpha_grad_fn, alpha_init_np: np.ndarray, beta_init_np: np.ndarray):
    import jax.numpy as jnp

    alpha = jnp.asarray(alpha_init_np, dtype=jnp.float64)
    beta = jnp.asarray(beta_init_np, dtype=jnp.float64)
    y = alpha_grad_fn(alpha, beta) if cfg.use_messenger else alpha_grad_fn(alpha)
    return {
        "step": 0,
        "alpha": alpha,
        "beta": beta,
        "y": y,
        "a_alpha": jnp.zeros_like(alpha),
        "b_alpha": jnp.zeros_like(alpha),
        "a_beta": jnp.zeros_like(beta),
        "b_beta": jnp.zeros_like(beta),
    }


def checkpoint_payload(
    iteration: int,
    metrics: dict[str, Any],
    alpha: np.ndarray,
    beta: np.ndarray,
    *,
    optimizer_state: dict[str, Any] | None = None,
    failed: bool = False,
    error_message: str | None = None,
) -> dict[str, Any]:
    payload = {
        "iteration": int(iteration),
        "latest_metrics": sanitize_jsonable(metrics),
        "alpha": encode_array(alpha),
        "beta": encode_array(beta),
        "failed": bool(failed),
        "error_message": error_message,
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = sanitize_jsonable(optimizer_state)
    return payload


def sparse_matrix_preview(matrix, max_elements: int = 200) -> str:
    rows_needed = max(1, math.ceil(max_elements / matrix.shape[1]))
    dense_head = np.asarray(matrix[:rows_needed, :].toarray(), dtype=np.float64)
    flat = dense_head.reshape(-1)[:max_elements]
    return (
        f"shape={matrix.shape}, showing first {len(flat)} flattened elements:\n"
        f"{format_array_preview(flat, max_elements=max_elements)}"
    )


def write_report(report_path: Path, result: dict[str, Any]) -> None:
    final = result["history"][-1]
    lines = [
        "# Partition Compare Optimization Report",
        "",
        "## Setup",
        f"- Case: `{result['case']}`",
        f"- Agent grid: `{result['problem']['n_rows']} x {result['problem']['n_cols']}`",
        f"- Local block qubits: `{result['problem']['local_qubits']}`",
        f"- Global qubits: `{result['problem']['global_qubits']}`",
        f"- Iterations: `{result['optimization']['iterations_requested']}`",
        f"- Learning rate: `{result['config']['learning_rate']}`",
        f"- Initialization seed: `{result['config']['init_seed']}`",
        f"- Initialization mode: `{result['config']['init_mode']}`",
        f"- Coupling `J`: `{result['problem']['j_coupling']}`",
        f"- Ansatz layers: `{result['config']['layers']}`",
        f"- Hadamard prefix: `{result['config']['hadamard_prefix']}`",
        f"- Messenger active: `{result['problem']['use_messenger']}`",
        f"- Row Laplacian: `{result['problem']['row_laplacian']}`",
        f"- Column mixing matrix: `{result['problem']['column_mix']}`",
        f"- Scaled spectrum check: `lambda_min={result['problem']['scaled_lambda_min']:.12g}`, `lambda_max={result['problem']['scaled_lambda_max']:.12g}`",
        f"- Row-block norms of `b_i`: `{result['problem']['b_row_norms']}`",
        f"- Agent-block norms of `b_ij`: `{result['problem']['b_agent_norms']}`",
        f"- Column split used for `b_ij`: `{result['problem']['b_column_split']}`",
        "",
        "## Final Metrics",
        f"- Global cost: `{final['global_cost']:.12g}`",
        f"- Global residual: `{final['global_residual_l2']:.12g}`",
        f"- Consensus error between row-neighbors: `{final['consensus_error_l2']:.12g}`",
        f"- Solution L2 error: `{final['solution_error_l2']:.12g}`",
        f"- Relative solution L2 error: `{final['relative_solution_error_l2']:.12g}`",
        f"- Elapsed time: `{result['optimization']['elapsed_s']:.6f} s`",
        "",
        "## Rescaled-x Diagnostic",
        f"- Best scalar to match true `x`: `{result['rescaled_diagnostics']['best_scale_to_true_real']:.12g}{result['rescaled_diagnostics']['best_scale_to_true_imag']:+.12g}j`",
        f"- Raw reconstructed `||x||_2`: `{result['rescaled_diagnostics']['raw_x_norm_l2']:.12g}`",
        f"- True `||x_true||_2`: `{result['rescaled_diagnostics']['true_x_norm_l2']:.12g}`",
        f"- Rescaled `||x||_2`: `{result['rescaled_diagnostics']['rescaled_x_norm_l2']:.12g}`",
        f"- Cosine similarity to true `x`: `{result['rescaled_diagnostics']['cosine_similarity_to_true']:.12g}`",
        f"- Rescaled relative solution error: `{result['rescaled_diagnostics']['rescaled_relative_solution_error_l2']:.12g}`",
        f"- Rescaled residual norm: `{result['rescaled_diagnostics']['rescaled_residual_norm_l2']:.12g}`",
        "",
        "## State Reconstruction Checks",
        f"- Final `|| |X_ij> ||_2`: `{result['final_diagnostics']['x_unit_state_norms']}`",
        f"- Final `||x_ij||_2`: `{result['final_diagnostics']['x_block_norms']}`",
        f"- Final column-block norms: `{result['final_diagnostics']['column_block_norms']}`",
        f"- Final row-copy norms: `{result['final_diagnostics']['row_copy_norms']}`",
        f"- Row-action residual norms: `{result['final_diagnostics']['row_action_residual_norms']}`",
        "",
        "## sigma and lambda",
        "### sigma",
        "```text",
        format_array_preview(np.asarray(result["final_diagnostics"]["sigma"], dtype=np.float64), max_elements=result["config"]["preview_elements"]),
        "```",
        "",
    ]
    if result["problem"]["use_messenger"]:
        lines.extend(
            [
                "### lambda",
                "```text",
                format_array_preview(
                    np.asarray(result["final_diagnostics"]["lambda"], dtype=np.float64),
                    max_elements=result["config"]["preview_elements"],
                ),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## b decomposition",
            "The stored vectors satisfy `b_i = sum_j b_ij` for each row.",
            "",
        ]
    )

    for row_index, preview in enumerate(result["final_diagnostics"]["b_row_previews"], start=1):
        lines.extend(
            [
                f"### b_{row_index}",
                "```text",
                preview,
                "```",
                "",
            ]
        )
    for row_index, row_previews in enumerate(result["final_diagnostics"]["b_agent_previews"], start=1):
        for col_index, preview in enumerate(row_previews, start=1):
            lines.extend(
                [
                    f"### b_{row_index}{col_index}",
                    "```text",
                    preview,
                    "```",
                    "",
                ]
            )

    lines.extend(
        [
            "## Reconstructed Row Actions",
        ]
    )
    for row_index, preview in enumerate(result["final_diagnostics"]["row_action_previews"], start=1):
        lines.extend(
            [
                f"### sum_j A_{row_index}j x_{row_index}j",
                "```text",
                preview,
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Communication Objects",
            "### Column mixing matrix",
            "```text",
            format_array_preview(
                np.asarray(result["problem"]["column_mix"], dtype=np.float64),
                max_elements=result["config"]["preview_elements"],
            ),
            "```",
            "",
            "### Row Laplacian",
            "```text",
            format_array_preview(
                np.asarray(result["problem"]["row_laplacian"], dtype=np.float64),
                max_elements=result["config"]["preview_elements"],
            ),
            "```",
            "",
            "## Final Trainable Parameters",
            "The first entry in each `alpha[i, j, :]` block is `sigma_ij`.",
            "",
            "### alpha",
            "```text",
            result["final_state"]["alpha_preview"],
            "```",
            "",
        ]
    )
    if result["problem"]["use_messenger"]:
        lines.extend(
            [
                "### beta",
                "```text",
                result["final_state"]["beta_preview"],
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Final Reconstructed x",
            "```text",
            result["final_state"]["x_estimate_preview"],
            "```",
            "",
            "## True Solution x",
            "```text",
            result["final_state"]["x_true_preview"],
            "```",
            "",
            "## True Matrix A",
            "```text",
            result["linear_system"]["A_sparse_preview"],
            "```",
            "",
            "## True Vector b",
            "```text",
            result["linear_system"]["b_preview"],
            "```",
            "",
            "## Artifacts",
            f"- JSON: `{result['artifacts']['json']}`",
            f"- Report: `{result['artifacts']['report']}`",
            f"- History: `{result['artifacts']['history']}`",
            f"- Checkpoint: `{result['artifacts']['checkpoint']}`",
            f"- Config: `{result['artifacts']['config']}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
