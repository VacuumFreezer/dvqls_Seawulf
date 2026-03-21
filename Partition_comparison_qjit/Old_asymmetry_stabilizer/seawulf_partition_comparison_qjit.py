from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pennylane as qml
from pennylane import numpy as pnp


def _identity_qjit(fn=None, *args, **kwargs):
    del args, kwargs
    if fn is None:
        return lambda f: f
    return fn


CATALYST_AVAILABLE = True
CATALYST_IMPORT_ERROR = None
try:
    import catalyst
except Exception as exc:  # pragma: no cover - env dependent fallback
    catalyst = None
    CATALYST_AVAILABLE = False
    CATALYST_IMPORT_ERROR = exc
    qml.qjit = _identity_qjit


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Partition_comparison_qjit.benchmark_13q_real_cluster_common import load_module_from_path
from common.DIGing_jax import (
    build_metropolis_matrix,
    consensus_mix_metropolis_jax,
    init_tracker_from_grad,
    update_gradient_tracker_metropolis_jax,
)
from common.params_io import flatten_params, rebuild_global_params, update_global_from_flat
from common.reporting import JsonlWriter, make_run_dir, setup_logger

import objective.builder_cluster_nodispatch as ib
from objective.circuits_cluster_nodispatch import (
    ANSATZ_BRICKWALL_RY_CZ,
    ANSATZ_CLUSTER_LOCAL_RY,
    ANSATZ_CLUSTER_RZ,
    ANSATZ_CLUSTER_RZ_LOCAL_RY,
    VALID_ANSATZ_KINDS,
    apply_selected_ansatz,
    describe_ansatz,
    normalize_ansatz_kind,
)

try:
    from scipy import sparse as sp
    from scipy.sparse.linalg import spsolve

    SCIPY_AVAILABLE = True
except Exception:
    sp = None
    spsolve = None
    SCIPY_AVAILABLE = False

from Partition_comparison_qjit.verify_partition_consistency import reconstruct_global_entries


def load_static_ops(spec: str):
    raw = str(spec).strip()
    if not raw:
        raise ValueError("--static_ops must be a non-empty module name or .py path.")

    candidate = Path(raw)
    if candidate.suffix == ".py" or "/" in raw or raw.startswith("."):
        resolved = (candidate if candidate.is_absolute() else (ROOT / candidate)).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Static ops file not found: {resolved}")
        return load_module_from_path(resolved)

    return importlib.import_module(raw)


def load_problem_system_and_wires(ops_module):
    if hasattr(ops_module, "SYSTEM"):
        system = ops_module.SYSTEM
    elif hasattr(ops_module, "SYSTEMS"):
        systems = getattr(ops_module, "SYSTEMS")
        if not systems:
            raise RuntimeError(f"{ops_module.__name__} exposes SYSTEMS but it is empty.")
        system = systems[next(iter(systems))]
    else:
        raise RuntimeError(f"{ops_module.__name__} does not expose SYSTEM or SYSTEMS.")

    if hasattr(system, "data_wires"):
        data_wires = list(system.data_wires)
    elif hasattr(ops_module, "DATA_WIRES"):
        data_wires = list(ops_module.DATA_WIRES)
    else:
        raise RuntimeError(f"{ops_module.__name__} does not expose DATA_WIRES.")

    return system, data_wires


def to_jax_flat(flat_list):
    return [jnp.asarray(x) for x in flat_list]


def _parse_local_ry_support_arg(raw_value) -> tuple[int, ...]:
    if raw_value is None:
        return ()
    text = str(raw_value).strip()
    if not text:
        return ()
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def resolve_local_ry_support(ops_module, raw_value) -> tuple[int, ...]:
    cli_value = _parse_local_ry_support_arg(raw_value)
    if cli_value:
        return cli_value

    if hasattr(ops_module, "LOCAL_RY_SUPPORT_WIRES"):
        return tuple(int(x) for x in getattr(ops_module, "LOCAL_RY_SUPPORT_WIRES"))

    metadata = getattr(getattr(ops_module, "SYSTEM", None), "metadata", {})
    if "local_ry_support_wires" in metadata:
        return tuple(int(x) for x in metadata["local_ry_support_wires"])

    return ()


def resolve_scaffold_edges(ops_module, n_qubits: int) -> tuple[tuple[int, int], ...]:
    if hasattr(ops_module, "CLUSTER_SCAFFOLD_EDGES_LOCAL"):
        return tuple((int(a), int(b)) for a, b in getattr(ops_module, "CLUSTER_SCAFFOLD_EDGES_LOCAL"))

    metadata = getattr(getattr(ops_module, "SYSTEM", None), "metadata", {})
    if "cluster_scaffold_edges_local" in metadata:
        return tuple((int(a), int(b)) for a, b in metadata["cluster_scaffold_edges_local"])

    return tuple((wire, wire + 1) for wire in range(max(0, int(n_qubits) - 1)))


def build_line_topology(n_agents: int) -> dict[int, list[int]]:
    topology = {}
    for node in range(int(n_agents)):
        neighbors = []
        if node > 0:
            neighbors.append(node - 1)
        if node + 1 < int(n_agents):
            neighbors.append(node + 1)
        topology[node] = neighbors
    return topology


INIT_MODE_UNIFORM_PM_PI = "uniform_pm_pi"
INIT_MODE_UNIFORM_0_PI = "uniform_0_pi"
INIT_MODE_METADATA_GAUSSIAN = "metadata_gaussian"
INIT_MODE_CONSTANT_CENTER_GAUSSIAN = "constant_center_gaussian"
VALID_INIT_MODES = (
    INIT_MODE_UNIFORM_PM_PI,
    INIT_MODE_UNIFORM_0_PI,
    INIT_MODE_METADATA_GAUSSIAN,
    INIT_MODE_CONSTANT_CENTER_GAUSSIAN,
)


def _build_metadata_angle_base(metadata, layers: int, n_qubits: int, agent_id: int) -> np.ndarray:
    init_angle_fill = float(metadata.get("init_angle_fill", 0.0))
    agent_init_overrides = metadata.get("agent_init_overrides", {})
    base = np.full((layers, n_qubits), init_angle_fill, dtype=np.float64)
    overrides = agent_init_overrides.get(str(agent_id), agent_init_overrides.get(agent_id, {}))
    for wire, value in dict(overrides).items():
        base[0, int(wire)] = float(value)
    return base


def initialize_cluster_params_jax(
    system,
    n_qubits: int,
    layers: int,
    seed: int = 0,
    *,
    init_mode: str = INIT_MODE_UNIFORM_PM_PI,
    init_angle_center: float = (np.pi / 2.0),
    init_angle_noise_std: float = 0.05,
    init_sigma_value: float | None = None,
    init_sigma_noise_std: float = 0.05,
    init_lambda_value: float | None = None,
    init_lambda_noise_std: float = 0.05,
):
    n_agents = int(system.n)
    key = jax.random.PRNGKey(seed)
    global_params = {"alpha": [], "beta": [], "sigma": [], "lambda": [], "b_norm": []}
    metadata = getattr(system, "metadata", {})
    sigma_target = float(metadata.get("init_sigma_target", 1.0 / np.sqrt(2.0)))
    init_mode = str(init_mode)
    if init_mode not in VALID_INIT_MODES:
        raise ValueError(f"Unknown init_mode `{init_mode}`. Expected one of {VALID_INIT_MODES}.")

    for sys_id in range(n_agents):
        local_b_norms = system.get_local_b_norms(sys_id)
        row_alpha, row_beta = [], []
        row_sigma, row_lam = [], []
        row_bnorms = []

        for agent_id in range(n_agents):
            key, sub_a = jax.random.split(key)
            key, sub_b = jax.random.split(key)
            key, sub_s = jax.random.split(key)
            key, sub_l = jax.random.split(key)

            if init_mode in (INIT_MODE_UNIFORM_PM_PI, INIT_MODE_UNIFORM_0_PI):
                minval = -jnp.pi if init_mode == INIT_MODE_UNIFORM_PM_PI else 0.0
                maxval = jnp.pi
                a = jax.random.uniform(
                    sub_a,
                    shape=(layers, n_qubits),
                    minval=minval,
                    maxval=maxval,
                    dtype=jnp.float64,
                )
                b = jax.random.uniform(
                    sub_b,
                    shape=(layers, n_qubits),
                    minval=minval,
                    maxval=maxval,
                    dtype=jnp.float64,
                )
            else:
                if init_mode == INIT_MODE_METADATA_GAUSSIAN:
                    base = _build_metadata_angle_base(metadata, layers=layers, n_qubits=n_qubits, agent_id=agent_id)
                else:
                    base = np.full((layers, n_qubits), float(init_angle_center), dtype=np.float64)
                a = jnp.asarray(base) + float(init_angle_noise_std) * jax.random.normal(
                    sub_a, shape=(layers, n_qubits), dtype=jnp.float64
                )
                b = jnp.asarray(base) + float(init_angle_noise_std) * jax.random.normal(
                    sub_b, shape=(layers, n_qubits), dtype=jnp.float64
                )

            sigma_base = sigma_target if init_sigma_value is None else float(init_sigma_value)
            lambda_base = 0.0 if init_lambda_value is None else float(init_lambda_value)
            s = jnp.asarray(
                sigma_base + float(init_sigma_noise_std) * jax.random.normal(sub_s, shape=(), dtype=jnp.float64),
                dtype=jnp.float64,
            )
            l = jnp.asarray(
                lambda_base + float(init_lambda_noise_std) * jax.random.normal(sub_l, shape=(), dtype=jnp.float64),
                dtype=jnp.float64,
            )

            row_alpha.append(a)
            row_beta.append(b)
            row_sigma.append(s)
            row_lam.append(l)
            row_bnorms.append(jax.lax.stop_gradient(jnp.asarray(float(local_b_norms[agent_id]), dtype=jnp.float64)))

        global_params["alpha"].append(row_alpha)
        global_params["beta"].append(row_beta)
        global_params["sigma"].append(row_sigma)
        global_params["lambda"].append(row_lam)
        global_params["b_norm"].append(row_bnorms)

    return global_params, key


def _count_local_qnodes(term_count: int, degree: int) -> int:
    m = int(degree) + 1
    return (term_count * term_count) + ((m + 1) * term_count) + m + (m * (m - 1) // 2)


def _bytes_human(nbytes: int) -> str:
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024.0 or unit == "PiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PiB"


def estimate_loss_memory_usage(system, row_topology, n_input_qubit: int):
    n_wires = int(n_input_qubit) + 1
    statevector_bytes = (1 << n_wires) * np.dtype(np.complex128).itemsize

    entry_details = []
    total_qnodes = 0
    for sys_id in range(int(system.n)):
        for agent_id in range(int(system.n)):
            term_count = len(system.gates_grid[sys_id][agent_id])
            degree = len(row_topology[agent_id])
            qnode_evals = _count_local_qnodes(term_count, degree)
            total_qnodes += qnode_evals
            entry_details.append(
                {
                    "sys_id": sys_id,
                    "agent_id": agent_id,
                    "L": term_count,
                    "degree": degree,
                    "qnode_evals": qnode_evals,
                }
            )

    return {
        "n_wires_per_hadamard_test": n_wires,
        "statevector_bytes": statevector_bytes,
        "statevector_human": _bytes_human(statevector_bytes),
        "peak_bytes_conservative": 2 * statevector_bytes,
        "peak_human_conservative": _bytes_human(2 * statevector_bytes),
        "sequential_bytes_per_loss_eval": total_qnodes * statevector_bytes,
        "sequential_human_per_loss_eval": _bytes_human(total_qnodes * statevector_bytes),
        "total_qnode_evals_per_loss": total_qnodes,
        "entry_details": entry_details,
    }


def _format_array_for_report(arr, *, precision: int = 6) -> str:
    arr_np = np.real_if_close(np.asarray(arr))
    return np.array2string(
        arr_np,
        precision=precision,
        suppress_small=False,
        separator=" ",
        max_line_width=120,
        threshold=np.inf,
    )


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _global_param_summary(global_params):
    sigma = np.asarray(_to_jsonable(global_params["sigma"]), dtype=float)
    lam = np.asarray(_to_jsonable(global_params["lambda"]), dtype=float)
    alpha = np.asarray(_to_jsonable(global_params["alpha"]), dtype=float)
    beta = np.asarray(_to_jsonable(global_params["beta"]), dtype=float)
    return {
        "sigma_min": float(np.min(sigma)),
        "sigma_max": float(np.max(sigma)),
        "sigma_mean": float(np.mean(sigma)),
        "sigma_l2": float(np.linalg.norm(sigma)),
        "lambda_min": float(np.min(lam)),
        "lambda_max": float(np.max(lam)),
        "lambda_mean": float(np.mean(lam)),
        "lambda_l2": float(np.linalg.norm(lam)),
        "alpha_l2": float(np.linalg.norm(alpha)),
        "beta_l2": float(np.linalg.norm(beta)),
    }


def _summarize_exception(exc: Exception, *, max_lines: int = 2, max_chars: int = 400) -> str:
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    if not lines:
        return f"{type(exc).__name__}"

    summary = " | ".join(lines[-max_lines:])
    if len(summary) > max_chars:
        summary = summary[-max_chars:]
    return f"{type(exc).__name__}: {summary}"


def build_state_getter(
    n_qubits: int,
    *,
    ansatz_kind: str,
    repeat_cz_each_layer: bool,
    local_ry_support,
    scaffold_edges,
):
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="jax")
    def get_local_state(weights):
        apply_selected_ansatz(
            weights,
            n_qubits,
            ansatz_kind=ansatz_kind,
            repeat_cz_each_layer=repeat_cz_each_layer,
            local_ry_support=local_ry_support,
            scaffold_edges=scaffold_edges,
        )
        return qml.state()

    return get_local_state


def recover_row_blocks(global_params, n_agents: int, get_local_state):
    row_blocks = []
    for row_id in range(int(n_agents)):
        row = []
        for col_id in range(int(n_agents)):
            alpha = global_params["alpha"][row_id][col_id]
            sigma = float(np.asarray(global_params["sigma"][row_id][col_id]))
            state = np.asarray(get_local_state(alpha))
            row.append(sigma * state)
        row_blocks.append(row)
    return row_blocks


def average_column_blocks(row_blocks):
    n_agents = len(row_blocks)
    mean_blocks = []
    for col_id in range(n_agents):
        stack = np.stack([row_blocks[row_id][col_id] for row_id in range(n_agents)], axis=0)
        mean_blocks.append(np.mean(stack, axis=0))
    return mean_blocks


def flatten_blocks(blocks):
    return np.concatenate([np.asarray(block) for block in blocks], axis=0)


def compute_consensus_error(row_blocks) -> float:
    n_agents = len(row_blocks)
    variances = []
    for col_id in range(n_agents):
        stack = np.stack([row_blocks[row_id][col_id] for row_id in range(n_agents)], axis=0)
        mean_vec = np.mean(stack, axis=0)
        diffs = stack - mean_vec
        sq_dists = np.sum(np.abs(diffs) ** 2, axis=1)
        variances.append(float(np.mean(sq_dists)))
    return float(np.mean(variances)) if variances else 0.0


def compute_global_residual_norm(system, mean_blocks, row_b_totals) -> float:
    residual_blocks = []
    for row_id in range(int(system.n)):
        row_action = np.zeros_like(mean_blocks[0], dtype=complex)
        for col_id in range(int(system.n)):
            row_action = row_action + system.apply_block_operator(row_id, col_id, mean_blocks[col_id])
        residual_blocks.append(row_action - row_b_totals[row_id])
    return float(np.linalg.norm(flatten_blocks(residual_blocks)))


def compute_l2_error(recovered_global_solution, true_solution) -> float:
    true_vec = np.asarray(true_solution).reshape(-1)
    diff = true_vec - np.asarray(recovered_global_solution).reshape(-1)
    return float(np.linalg.norm(diff) / np.linalg.norm(true_vec))


def compute_metric_bundle(global_params, system, get_local_state, row_b_totals, true_solution):
    row_blocks = recover_row_blocks(global_params, system.n, get_local_state)
    mean_blocks = average_column_blocks(row_blocks)
    recovered_solution = flatten_blocks(mean_blocks)
    return {
        "row_blocks": row_blocks,
        "mean_blocks": mean_blocks,
        "recovered_solution": recovered_solution,
        "residual_norm": compute_global_residual_norm(system, mean_blocks, row_b_totals),
        "l2_error": compute_l2_error(recovered_solution, true_solution),
        "consensus_error": compute_consensus_error(row_blocks),
    }


def compute_true_solution(system, global_b, logger):
    if SCIPY_AVAILABLE:
        entries = reconstruct_global_entries(system)
        dim = len(global_b)
        rows = []
        cols = []
        data = []
        for (row, col), value in entries.items():
            rows.append(int(row))
            cols.append(int(col))
            data.append(complex(value))

        matrix = sp.coo_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols))), shape=(dim, dim)).tocsr()
        logger.info("Computing true solution x* via scipy.sparse.linalg.spsolve on the reconstructed global matrix.")
        return np.asarray(spsolve(matrix, np.asarray(global_b, dtype=np.complex128)), dtype=np.complex128)

    logger.info(
        "SciPy sparse solve is unavailable in this environment; falling back to the analytic block-state helper "
        "for the true-solution reference."
    )
    return np.asarray(system.true_solution_vector())


def _write_final_params_json(path: Path, global_params):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            _to_jsonable(
                {
                    "alpha": global_params["alpha"],
                    "beta": global_params["beta"],
                    "sigma": global_params["sigma"],
                    "lambda": global_params["lambda"],
                    "b_norm": global_params["b_norm"],
                    "summary": _global_param_summary(global_params),
                }
            ),
            handle,
            ensure_ascii=False,
            indent=2,
        )


def _write_run_config(path: Path, payload: dict):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, ensure_ascii=False, indent=2)


def write_analysis_report(
    path: Path,
    *,
    args,
    ops_module_name: str,
    system,
    data_wires,
    mem_info,
    global_params,
    row_b_totals,
    true_solution,
    final_metrics,
):
    with path.open("w", encoding="utf-8") as out:
        metadata = {
            key: value
            for key, value in getattr(system, "metadata", {}).items()
            if key not in {"reference_row_gates", "exact_solution_gates_by_col"}
        }
        out.write("13-qubit real cluster partition-comparison run\n")
        out.write(f"static_ops: {ops_module_name}\n")
        out.write(f"system name: {getattr(system, 'name', 'unknown')}\n")
        out.write(f"n agents: {system.n}\n")
        out.write(f"local data qubits: {len(data_wires)}\n")
        out.write(f"ansatz: {args.ansatz} ({describe_ansatz(args.ansatz)})\n")
        out.write(f"layers: {int(args.layers)}\n")
        out.write(f"repeat_cz_each_layer: {bool(args.repeat_cz_each_layer)}\n")
        out.write(f"init_mode: {args.init_mode}\n")
        out.write(f"init_angle_center: {float(args.init_angle_center):.8f}\n")
        out.write(f"init_angle_noise_std: {float(args.init_angle_noise_std):.8f}\n")
        out.write(f"init_sigma_value: {args.init_sigma_value}\n")
        out.write(f"init_sigma_noise_std: {float(args.init_sigma_noise_std):.8f}\n")
        out.write(f"init_lambda_value: {args.init_lambda_value}\n")
        out.write(f"init_lambda_noise_std: {float(args.init_lambda_noise_std):.8f}\n")
        out.write(f"topology: {args.topology}\n")
        out.write(f"row/column graph: line\n")
        out.write(f"scaffold_edges: {tuple((int(a), int(b)) for a, b in final_metrics['scaffold_edges'])}\n")
        out.write(f"statevector per Hadamard test: {mem_info['statevector_human']}\n")
        out.write(f"conservative peak per Hadamard test: {mem_info['peak_human_conservative']}\n")
        out.write(f"qnode evals per loss evaluation: {mem_info['total_qnode_evals_per_loss']}\n")
        out.write(f"sequential statevector traffic per loss evaluation: {mem_info['sequential_human_per_loss_eval']}\n")
        if metadata:
            out.write(f"metadata: {metadata}\n")

        out.write(f"final parameter summary: {_global_param_summary(global_params)}\n")
        out.write(f"final global cost: {float(final_metrics['global_cost']):.8e}\n")
        out.write(f"final ||Ax-b||: {float(final_metrics['residual_norm']):.8e}\n")
        out.write(f"final relative L2 error: {float(final_metrics['l2_error']):.8e}\n")
        out.write(f"final consensus error: {float(final_metrics['consensus_error']):.8e}\n")

        out.write("\n" + "=" * 80 + "\n")
        out.write("TRAINABLE PARAMETERS\n")
        out.write("=" * 80 + "\n")
        for row_id in range(int(system.n)):
            for col_id in range(int(system.n)):
                out.write(f"\n[row={row_id}, col={col_id}]\n")
                out.write(f"alpha:\n{_format_array_for_report(global_params['alpha'][row_id][col_id])}\n")
                out.write(f"beta:\n{_format_array_for_report(global_params['beta'][row_id][col_id])}\n")
                out.write(f"sigma: {float(np.asarray(global_params['sigma'][row_id][col_id])):.8e}\n")
                out.write(f"lambda: {float(np.asarray(global_params['lambda'][row_id][col_id])):.8e}\n")

        out.write("\n" + "=" * 80 + "\n")
        out.write("RECONSTRUCTED x_ij AND ROW CHECKS\n")
        out.write("=" * 80 + "\n")
        for row_id in range(int(system.n)):
            out.write(f"\n>>> row {row_id} <<<\n")
            row_action = np.zeros_like(final_metrics["row_blocks"][row_id][0], dtype=complex)
            for col_id in range(int(system.n)):
                x_ij = final_metrics["row_blocks"][row_id][col_id]
                row_action = row_action + system.apply_block_operator(row_id, col_id, x_ij)
                out.write(f"[row={row_id}, col={col_id}] x_ij:\n{_format_array_for_report(x_ij)}\n")
            out.write(f"sum_j A_ij x_ij:\n{_format_array_for_report(row_action)}\n")
            out.write(f"b_i:\n{_format_array_for_report(row_b_totals[row_id])}\n")
            out.write(f"row residual norm: {float(np.linalg.norm(row_action - row_b_totals[row_id])):.8e}\n")

        out.write("\n" + "=" * 80 + "\n")
        out.write("GLOBAL RECOVERY\n")
        out.write("=" * 80 + "\n")
        out.write(f"consensus-averaged x:\n{_format_array_for_report(final_metrics['recovered_solution'])}\n")
        out.write(f"true solution x*:\n{_format_array_for_report(true_solution)}\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--static_ops", required=True, help="Module name or .py path for the partitioned static-ops file.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topology", type=str, default="line", choices=("line",))
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--decay", type=float, default=0.9999)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--ansatz", type=str, default=ANSATZ_BRICKWALL_RY_CZ, choices=VALID_ANSATZ_KINDS)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--repeat_cz_each_layer", action="store_true")
    ap.add_argument("--init_mode", type=str, default=INIT_MODE_UNIFORM_PM_PI, choices=VALID_INIT_MODES)
    ap.add_argument("--init_angle_center", type=float, default=(np.pi / 2.0))
    ap.add_argument("--init_angle_noise_std", type=float, default=0.05)
    ap.add_argument("--init_sigma_value", type=float, default=None)
    ap.add_argument("--init_sigma_noise_std", type=float, default=0.05)
    ap.add_argument("--init_lambda_value", type=float, default=None)
    ap.add_argument("--init_lambda_noise_std", type=float, default=0.05)
    ap.add_argument(
        "--local_ry_support",
        type=str,
        default="",
        help="Comma-separated local wire ids for ansatze that require a local-RY support set.",
    )
    args = ap.parse_args(argv)
    args.ansatz = normalize_ansatz_kind(args.ansatz)
    if args.layers < 1:
        raise ValueError("--layers must be at least 1.")

    ops = load_static_ops(args.static_ops)
    system, data_wires = load_problem_system_and_wires(ops)
    local_ry_support = resolve_local_ry_support(ops, args.local_ry_support)
    scaffold_edges = resolve_scaffold_edges(ops, len(data_wires))
    if args.ansatz in (ANSATZ_CLUSTER_RZ_LOCAL_RY, ANSATZ_CLUSTER_LOCAL_RY) and not local_ry_support:
        raise ValueError(
            f"Ansatz `{args.ansatz}` requires a non-empty local_ry_support wire set. "
            "Provide --local_ry_support or expose LOCAL_RY_SUPPORT_WIRES in the static_ops module."
        )

    pnp.random.seed(args.seed)
    np.random.seed(args.seed)
    jax.config.update("jax_enable_x64", True)

    paths = make_run_dir(args.out)
    logger = setup_logger(paths.report_txt)
    metrics = JsonlWriter(paths.metrics_jsonl)

    n_agents = int(system.n)
    n_qubits = len(data_wires)
    topology = build_line_topology(n_agents)
    spectrum = getattr(ops, "SPECTRUM_INFO", getattr(system, "metadata", {}).get("spectrum"))

    logger.info(f"System variant: {getattr(system, 'name', 'unknown')}")
    logger.info(f"static_ops: {args.static_ops}")
    logger.info(f"agents per row/column: {n_agents}")
    logger.info(f"local data qubits: {n_qubits}")
    logger.info(f"ansatz: {args.ansatz} ({describe_ansatz(args.ansatz)})")
    logger.info(f"layers: {int(args.layers)}")
    logger.info(f"repeat_cz_each_layer: {bool(args.repeat_cz_each_layer)}")
    logger.info(
        "init: "
        f"mode={args.init_mode}, "
        f"angle_center={float(args.init_angle_center):.8f}, "
        f"angle_noise_std={float(args.init_angle_noise_std):.8f}, "
        f"sigma_value={args.init_sigma_value}, "
        f"sigma_noise_std={float(args.init_sigma_noise_std):.8f}, "
        f"lambda_value={args.init_lambda_value}, "
        f"lambda_noise_std={float(args.init_lambda_noise_std):.8f}"
    )
    logger.info(f"local_ry_support: {tuple(int(x) for x in local_ry_support)}")
    logger.info(f"scaffold_edges: {tuple((int(a), int(b)) for a, b in scaffold_edges)}")
    logger.info(f"term counts by block: {[[len(cell) for cell in row] for row in system.gates_grid]}")
    logger.info(f"row graph: {topology}")
    logger.info(f"column graph: {topology}")
    if spectrum is not None:
        logger.info(
            "Analytic spectrum: "
            f"lambda_min={spectrum['lambda_min']:.8f}, "
            f"lambda_max={spectrum['lambda_max']:.8f}, "
            f"cond(A)={spectrum['condition_number']:.8f}"
        )
    if not CATALYST_AVAILABLE:
        logger.info(
            "Catalyst import failed in this environment; falling back to plain JAX without qjit. "
            f"Reason: {type(CATALYST_IMPORT_ERROR).__name__}: {CATALYST_IMPORT_ERROR}"
        )

    mem_info = estimate_loss_memory_usage(system, topology, n_qubits)
    logger.info(
        f"Per Hadamard-test statevector ({mem_info['n_wires_per_hadamard_test']} wires): {mem_info['statevector_human']}"
    )
    logger.info(f"Conservative peak per Hadamard test: {mem_info['peak_human_conservative']}")
    logger.info(f"QNode evaluations per compute_loss call: {mem_info['total_qnode_evals_per_loss']}")
    logger.info(
        f"Sequential statevector traffic per compute_loss call: {mem_info['sequential_human_per_loss_eval']}"
    )
    logger.info("These memory figures are for compute_loss only; compute_grad will be higher.")

    run_config = {
        "static_ops": args.static_ops,
        "out": str(paths.run_dir),
        "topology": args.topology,
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "lr": float(args.lr),
        "decay": float(args.decay),
        "log_every": int(args.log_every),
        "ansatz": args.ansatz,
        "layers": int(args.layers),
        "repeat_cz_each_layer": bool(args.repeat_cz_each_layer),
        "init_mode": args.init_mode,
        "init_angle_center": float(args.init_angle_center),
        "init_angle_noise_std": float(args.init_angle_noise_std),
        "init_sigma_value": None if args.init_sigma_value is None else float(args.init_sigma_value),
        "init_sigma_noise_std": float(args.init_sigma_noise_std),
        "init_lambda_value": None if args.init_lambda_value is None else float(args.init_lambda_value),
        "init_lambda_noise_std": float(args.init_lambda_noise_std),
        "local_ry_support": tuple(int(x) for x in local_ry_support),
        "scaffold_edges": tuple((int(a), int(b)) for a, b in scaffold_edges),
        "n_agents": n_agents,
        "n_qubits": n_qubits,
    }
    _write_run_config(paths.run_dir / "config_used.json", run_config)

    row_b_totals = [np.asarray(system.get_b_vectors(row_id)[0]) for row_id in range(n_agents)]
    global_b = flatten_blocks(row_b_totals)
    true_solution = compute_true_solution(system, global_b, logger)
    logger.info(f"Global right-hand side shape: {global_b.shape}")
    logger.info(f"True solution shape: {true_solution.shape}")
    global_params, _ = initialize_cluster_params_jax(
        system,
        n_qubits=n_qubits,
        layers=int(args.layers),
        seed=args.seed,
        init_mode=args.init_mode,
        init_angle_center=float(args.init_angle_center),
        init_angle_noise_std=float(args.init_angle_noise_std),
        init_sigma_value=args.init_sigma_value,
        init_sigma_noise_std=float(args.init_sigma_noise_std),
        init_lambda_value=args.init_lambda_value,
        init_lambda_noise_std=float(args.init_lambda_noise_std),
    )

    Wm = build_metropolis_matrix(topology, n=system.n, make_undirected=True)
    logger.info("Metropolis weight matrix Wm =\n" + str(Wm))

    logger.info("Starting prebuild_local_evals() for distributed Hadamard-test bundles.")
    t_prebuild = time.time()
    ib.prebuild_local_evals(
        system,
        topology,
        n_input_qubit=n_qubits,
        ansatz_kind=args.ansatz,
        repeat_cz_each_layer=args.repeat_cz_each_layer,
        local_ry_support=local_ry_support,
        scaffold_edges=scaffold_edges,
        diff_method="adjoint",
        interface="jax",
    )
    logger.info(f"Finished prebuild_local_evals() in {time.time() - t_prebuild:.2f} s.")

    def total_loss_fn_plain(args_flat):
        current_params = rebuild_global_params(args_flat, system.n, global_params["b_norm"])
        return ib.eval_total_loss_plain(current_params)

    if CATALYST_AVAILABLE:
        @qml.qjit
        def compute_grad(args_flat):
            def qjit_total_loss_fn(args_inner):
                current_params = rebuild_global_params(args_inner, system.n, global_params["b_norm"])
                return ib.eval_total_loss(current_params)

            return catalyst.grad(qjit_total_loss_fn, method="auto")(args_flat)

        @qml.qjit
        def compute_loss(args_flat):
            def qjit_total_loss_fn(args_inner):
                current_params = rebuild_global_params(args_inner, system.n, global_params["b_norm"])
                return ib.eval_total_loss(current_params)

            return qjit_total_loss_fn(args_flat)

        logger.info(
            "Optimization backend: qjit(catalyst.grad) + qjit(loss), matching "
            "optimization/seawulf_cat_line_tracking_nodispatch.py."
        )
    else:
        compute_loss_and_grad = jax.jit(jax.value_and_grad(total_loss_fn_plain))

        def compute_grad(args_flat):
            _, grad = compute_loss_and_grad(args_flat)
            return grad

        def compute_loss(args_flat):
            loss, _ = compute_loss_and_grad(args_flat)
            return loss

        logger.info(
            "Catalyst is unavailable in this environment; falling back to jax.jit(value_and_grad). "
            f"Reason: {_summarize_exception(CATALYST_IMPORT_ERROR)}"
        )

    get_local_state = build_state_getter(
        n_qubits=n_qubits,
        ansatz_kind=args.ansatz,
        repeat_cz_each_layer=args.repeat_cz_each_layer,
        local_ry_support=local_ry_support,
        scaffold_edges=scaffold_edges,
    )

    flat_params_init = to_jax_flat(flatten_params(global_params, keys=None))
    logger.info("Starting initial forward loss evaluation.")
    t_init_loss = time.time()
    current_loss = compute_loss(flat_params_init)
    logger.info(f"Finished initial forward loss evaluation in {time.time() - t_init_loss:.2f} s.")
    logger.info(f"[Init] Initial Loss = {float(current_loss):.5e}")

    lr_schedule = optax.exponential_decay(
        init_value=args.lr,
        transition_steps=1,
        decay_rate=args.decay,
        staircase=False,
    )
    opt_adam = optax.adam(lr_schedule)

    loss_history = []
    metric_epochs = []
    residual_history = []
    l2_history = []
    consensus_history = []

    logger.info("Starting initial gradient evaluation.")
    t_init_grad = time.time()
    try:
        grads_flat_init = compute_grad(flat_params_init)
    except Exception as exc:
        if not CATALYST_AVAILABLE:
            raise
        logger.info(
            "Catalyst gradient compilation failed for this benchmark; "
            "keeping qjit(loss) and falling back to jax.jit(value_and_grad) for gradients only. "
            f"Reason: {_summarize_exception(exc)}"
        )
        compute_loss_and_grad = jax.jit(jax.value_and_grad(total_loss_fn_plain))

        def compute_grad(args_flat):
            _, grad = compute_loss_and_grad(args_flat)
            return grad

        grads_flat_init = compute_grad(flat_params_init)
    logger.info(f"Finished initial gradient evaluation in {time.time() - t_init_grad:.2f} s.")
    grad_grid_init = rebuild_global_params(grads_flat_init, system.n, global_params["b_norm"])

    tracker_grid = init_tracker_from_grad(grad_grid_init)
    prev_grad_grid = copy.deepcopy(grad_grid_init)

    all_keys = ["alpha", "beta", "sigma", "lambda"]
    opt_adam_state = opt_adam.init(to_jax_flat(flatten_params(global_params, keys=all_keys)))
    loss_history.append(float(current_loss))

    t0 = time.time()
    last_log_time = t0
    last_metrics = None

    for epoch in range(1, args.epochs + 1):
        global_params = consensus_mix_metropolis_jax(global_params, W_np=Wm)

        flat_tracker = to_jax_flat(flatten_params(tracker_grid, keys=all_keys))
        flat_params = to_jax_flat(flatten_params(global_params, keys=all_keys))
        adam_updates, opt_adam_state = opt_adam.update(flat_tracker, opt_adam_state, params=flat_params)
        new_flat_params = optax.apply_updates(flat_params, adam_updates)
        update_global_from_flat(global_params, new_flat_params, keys=all_keys)

        flat_params_all = to_jax_flat(flatten_params(global_params, keys=None))
        grads_flat = compute_grad(flat_params_all)
        current_cost = compute_loss(flat_params_all)

        current_grad_grid = rebuild_global_params(grads_flat, system.n, global_params["b_norm"])
        tracker_grid = update_gradient_tracker_metropolis_jax(
            current_tracker=tracker_grid,
            current_grads=current_grad_grid,
            prev_grads=prev_grad_grid,
            W_np=Wm,
        )
        prev_grad_grid = copy.deepcopy(current_grad_grid)
        loss_history.append(float(current_cost))

        if (epoch % args.log_every) == 0 or epoch == 1:
            now = time.time()
            metric_bundle = compute_metric_bundle(
                global_params,
                system,
                get_local_state,
                row_b_totals=row_b_totals,
                true_solution=true_solution,
            )
            metric_epochs.append(epoch)
            residual_history.append(float(metric_bundle["residual_norm"]))
            l2_history.append(float(metric_bundle["l2_error"]))
            consensus_history.append(float(metric_bundle["consensus_error"]))
            last_metrics = metric_bundle
            log_interval_s = now - last_log_time
            wall_s = now - t0
            timestamp_utc = datetime.now(timezone.utc).isoformat()
            timestamp_local = datetime.now().astimezone().isoformat()

            current_lr = float(lr_schedule(epoch - 1))
            metrics.write(
                {
                    "epoch": epoch,
                    "global_cost": float(current_cost),
                    "loss": float(current_cost),
                    "residual_norm": float(metric_bundle["residual_norm"]),
                    "l2_error": float(metric_bundle["l2_error"]),
                    "consensus_error": float(metric_bundle["consensus_error"]),
                    "lr": current_lr,
                    "wall_s": wall_s,
                    "log_interval_s": log_interval_s,
                    "timestamp_utc": timestamp_utc,
                    "timestamp_local": timestamp_local,
                }
            )
            logger.info(
                f"[Epoch {epoch:04d}] Cost = {float(current_cost):.5e} | "
                f"||Ax-b|| = {float(metric_bundle['residual_norm']):.5e} | "
                f"L2 = {float(metric_bundle['l2_error']):.5e} | "
                f"Consensus = {float(metric_bundle['consensus_error']):.5e} | "
                f"wall_s = {wall_s:.2f} | "
                f"log_dt_s = {log_interval_s:.2f}"
            )
            last_log_time = now

    metrics.close()

    if last_metrics is None:
        last_metrics = compute_metric_bundle(
            global_params,
            system,
            get_local_state,
            row_b_totals=row_b_totals,
            true_solution=true_solution,
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    title = f"{getattr(system, 'name', 'partition')} | lr0={args.lr:g} | {args.ansatz}"

    xs_loss = np.arange(0, len(loss_history))
    plt.figure()
    plt.plot(xs_loss, loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Global Cost")
    plt.yscale("log")
    plt.grid(True)
    plt.title(title)
    plt.savefig(paths.fig_loss, dpi=200, bbox_inches="tight")
    plt.close()

    if metric_epochs:
        plt.figure()
        plt.plot(metric_epochs, l2_history)
        plt.xlabel("Epoch")
        plt.ylabel("Relative L2 Error")
        plt.yscale("log")
        plt.grid(True)
        plt.title(title)
        plt.savefig(paths.fig_diff, dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(metric_epochs, residual_history)
        plt.xlabel("Epoch")
        plt.ylabel("||Ax-b||")
        plt.yscale("log")
        plt.grid(True)
        plt.title(title)
        plt.savefig(paths.run_dir / "residual_norm.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(metric_epochs, consensus_history)
        plt.xlabel("Epoch")
        plt.ylabel("Consensus Error")
        plt.yscale("log")
        plt.grid(True)
        plt.title(title)
        plt.savefig(paths.run_dir / "consensus_error.png", dpi=200, bbox_inches="tight")
        plt.close()

    np.savez(
        paths.artifacts_npz,
        loss=np.array(loss_history),
        metric_epochs=np.array(metric_epochs),
        residual_norm=np.array(residual_history),
        l2_error=np.array(l2_history),
        consensus_error=np.array(consensus_history),
    )

    final_params_path = paths.run_dir / "final_params.json"
    _write_final_params_json(final_params_path, global_params)
    logger.info(f"Final parameters written to: {final_params_path}")

    analysis_path = paths.run_dir / "analysis.txt"
    write_analysis_report(
        analysis_path,
        args=args,
        ops_module_name=args.static_ops,
        system=system,
        data_wires=data_wires,
        mem_info=mem_info,
        global_params=global_params,
        row_b_totals=row_b_totals,
        true_solution=true_solution,
        final_metrics={
            **last_metrics,
            "global_cost": float(current_loss),
            "scaffold_edges": scaffold_edges,
        },
    )
    logger.info(f"Analysis report written to: {analysis_path}")
    logger.info(f"Finished. Outputs written to: {paths.run_dir}")


if __name__ == "__main__":
    main()
