from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pennylane as qml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _identity_qjit(fn=None, *args, **kwargs):
    del args, kwargs
    if fn is None:
        return lambda wrapped: wrapped
    return fn


CATALYST_AVAILABLE = True
CATALYST_IMPORT_ERROR = None
try:
    import catalyst
except Exception as exc:  # pragma: no cover - environment dependent
    catalyst = None
    CATALYST_AVAILABLE = False
    CATALYST_IMPORT_ERROR = exc
    qml.qjit = _identity_qjit


from Graph_comparison.Ising_5qubits_9benchmark.topology_registry_3x3 import (
    build_neighbor_map,
    first_agent_cost_formula,
    graph_label,
    laplacian_matrix_from_topology,
    metropolis_matrix_from_topology,
    resolve_benchmark_and_graphs,
)
from common.DIGing_jax import (
    consensus_mix_metropolis_jax,
    init_tracker_from_grad,
    update_gradient_tracker_metropolis_jax,
)
from common.params_io import flatten_params, rebuild_global_params, update_global_from_flat
from common.reporting import JsonlWriter, make_run_dir, setup_logger
import objective.builder_cluster_nodispatch as ib
from objective.circuits_cluster_nodispatch import (
    ANSATZ_PAPER_FIG3_RY_CZ,
    apply_selected_ansatz,
    describe_ansatz,
)


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


def _summarize_exception(exc: Exception, *, max_lines: int = 2, max_chars: int = 400) -> str:
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    if not lines:
        return type(exc).__name__
    summary = " | ".join(lines[-max_lines:])
    if len(summary) > max_chars:
        summary = summary[-max_chars:]
    return f"{type(exc).__name__}: {summary}"


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool from: {value}")


X_KEYS = ("alpha", "sigma")
Z_KEYS = ("beta", "lambda")

OPTIMIZATION_CASES = {
    "track_adamX_adamZ": {
        "label": "tracking+AdamOnX+AdamOnZ",
        "exchange": "tracking",
        "use_tracking": True,
        "x_optimizer": "adam",
        "z_optimizer": "adam",
    },
    "track_adamX_only": {
        "label": "tracking+AdamOnX",
        "exchange": "tracking",
        "use_tracking": True,
        "x_optimizer": "adam",
        "z_optimizer": "sgd",
    },
    "track_adamZ_only": {
        "label": "tracking+AdamOnZ",
        "exchange": "tracking",
        "use_tracking": True,
        "x_optimizer": "sgd",
        "z_optimizer": "adam",
    },
    "gradshare_adamX_adamZ": {
        "label": "gradient-sharing+AdamOnX+AdamOnZ",
        "exchange": "gradient-sharing",
        "use_tracking": False,
        "mix_gradients": True,
        "x_optimizer": "adam",
        "z_optimizer": "adam",
    },
    "consensus_adamX_adamZ": {
        "label": "consensus+AdamOnX+AdamOnZ",
        "exchange": "local-gradient",
        "use_tracking": False,
        "mix_gradients": False,
        "x_optimizer": "adam",
        "z_optimizer": "adam",
    },
}

OPTIMIZATION_CASE_ALIASES = {
    "trackadamxadamz": "track_adamX_adamZ",
    "trackingadamxadamz": "track_adamX_adamZ",
    "trackingadamonxadamonz": "track_adamX_adamZ",
    "trackadamxonly": "track_adamX_only",
    "trackingadamxonly": "track_adamX_only",
    "trackingadamonx": "track_adamX_only",
    "trackadamzonly": "track_adamZ_only",
    "trackingadamzonly": "track_adamZ_only",
    "trackingadamonz": "track_adamZ_only",
    "gradshareadamxadamz": "gradshare_adamX_adamZ",
    "gradientsharingadamxadamz": "gradshare_adamX_adamZ",
    "gradientsharingadamonxadamonz": "gradshare_adamX_adamZ",
    "notrackadamxadamz": "gradshare_adamX_adamZ",
    "consensusadamxadamz": "consensus_adamX_adamZ",
    "consensusadamonxadamonz": "consensus_adamX_adamZ",
}


def resolve_optimization_case(case_name: str):
    raw = str(case_name).strip()
    if not raw:
        raise ValueError("optimization case must be non-empty")
    canonical = "".join(ch for ch in raw.lower() if ch.isalnum())
    if canonical not in OPTIMIZATION_CASE_ALIASES:
        valid = ", ".join(sorted(OPTIMIZATION_CASES))
        raise ValueError(f"Unknown optimization case {case_name!r}. Expected one of: {valid}.")
    case_id = OPTIMIZATION_CASE_ALIASES[canonical]
    return case_id, OPTIMIZATION_CASES[case_id]


def make_optimizer(kind: str, lr_schedule):
    if kind == "adam":
        return optax.adam(lr_schedule)
    if kind == "sgd":
        return optax.sgd(lr_schedule)
    raise ValueError(f"Unsupported optimizer kind: {kind}")


def _default_scaffold_edges(n_qubits: int) -> tuple[tuple[int, int], ...]:
    return tuple((wire, wire + 1) for wire in range(max(0, int(n_qubits) - 1)))


def _split_alternating_edges(scaffold_edges) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    even_edges = []
    odd_edges = []
    for left, right in scaffold_edges:
        a, b = sorted((int(left), int(right)))
        if (b - a) != 1:
            continue
        if (a % 2) == 0:
            even_edges.append((a, b))
        else:
            odd_edges.append((a, b))
    return tuple(even_edges), tuple(odd_edges)


def consensus_mix_gradients_metropolis_jax(current_grads, W_np: np.ndarray):
    n_rows = len(current_grads["alpha"])
    n_cols = len(current_grads["alpha"][0])

    mixed_grads = {
        "alpha": [[None] * n_cols for _ in range(n_rows)],
        "sigma": [[None] * n_cols for _ in range(n_rows)],
        "beta": current_grads["beta"],
        "lambda": current_grads["lambda"],
        "b_norm": None,
    }

    for j in range(n_cols):
        for r in range(n_rows):
            idx = np.nonzero(W_np[r])[0]
            wts = jnp.asarray(W_np[r, idx], dtype=jnp.float64)

            grad_alpha_stack = jnp.stack([current_grads["alpha"][k][j] for k in idx], axis=0)
            grad_sigma_vec = jnp.stack([current_grads["sigma"][k][j] for k in idx], axis=0)

            mixed_grads["alpha"][r][j] = jnp.tensordot(wts, grad_alpha_stack, axes=(0, 0))
            mixed_grads["sigma"][r][j] = jnp.dot(wts, grad_sigma_vec)

    return mixed_grads


def to_jax_flat(flat_list):
    return [jnp.asarray(x) for x in flat_list]


def load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        key = "4x4" if "4x4" in systems else next(iter(systems))
        system = systems[key]
    else:
        raise RuntimeError(f"{ops_module.__name__} does not expose SYSTEM or SYSTEMS.")

    if hasattr(system, "data_wires"):
        data_wires = list(system.data_wires)
    elif hasattr(ops_module, "DATA_WIRES"):
        data_wires = list(ops_module.DATA_WIRES)
    else:
        raise RuntimeError(f"{ops_module.__name__} does not expose DATA_WIRES.")
    return system, data_wires


def initialize_paper_fig3_params_jax(system, n_qubits: int, layers: int, seed: int = 0):
    n_agents = int(system.n)
    key = jax.random.PRNGKey(seed)
    global_params = {"alpha": [], "beta": [], "sigma": [], "lambda": [], "b_norm": []}

    for sys_id in range(n_agents):
        _, *b_individuals = system.get_b_vectors(sys_id)
        row_alpha, row_beta = [], []
        row_sigma, row_lam = [], []
        row_bnorms = []

        for agent_id in range(n_agents):
            key, sub_a = jax.random.split(key)
            key, sub_b = jax.random.split(key)
            key, sub_s = jax.random.split(key)
            key, sub_l = jax.random.split(key)

            a = jax.random.uniform(
                sub_a,
                shape=(layers, 2, n_qubits),
                minval=-jnp.pi,
                maxval=jnp.pi,
                dtype=jnp.float64,
            )
            b = jax.random.uniform(
                sub_b,
                shape=(layers, 2, n_qubits),
                minval=-jnp.pi,
                maxval=jnp.pi,
                dtype=jnp.float64,
            )
            s = jax.random.uniform(sub_s, shape=(), minval=0.0, maxval=2.0, dtype=jnp.float64)
            l = jax.random.uniform(sub_l, shape=(), minval=0.0, maxval=2.0, dtype=jnp.float64)

            row_alpha.append(a)
            row_beta.append(b)
            row_sigma.append(s)
            row_lam.append(l)

            b_local = b_individuals[agent_id]
            local_norm_val = float(np.linalg.norm(b_local))
            row_bnorms.append(jax.lax.stop_gradient(jnp.asarray(local_norm_val, dtype=jnp.float64)))

        global_params["alpha"].append(row_alpha)
        global_params["beta"].append(row_beta)
        global_params["sigma"].append(row_sigma)
        global_params["lambda"].append(row_lam)
        global_params["b_norm"].append(row_bnorms)

    return global_params, key


def get_local_state_builder(n_qubits: int, scaffold_edges):
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="jax")
    def get_local_state(weights):
        apply_selected_ansatz(
            weights,
            n_qubits,
            ansatz_kind=ANSATZ_PAPER_FIG3_RY_CZ,
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
        row_action = np.zeros_like(mean_blocks[0], dtype=np.complex128)
        for col_id in range(int(system.n)):
            row_action = row_action + system.apply_block_operator(row_id, col_id, mean_blocks[col_id])
        residual_blocks.append(row_action - row_b_totals[row_id])
    return float(np.linalg.norm(flatten_blocks(residual_blocks)))


def compute_l2_error(recovered_global_solution, true_solution) -> float:
    true_vec = np.asarray(true_solution, dtype=np.complex128).reshape(-1)
    diff = true_vec - np.asarray(recovered_global_solution, dtype=np.complex128).reshape(-1)
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


def write_analysis_report(
    path: Path,
    *,
    args,
    optimization_case_id: str,
    optimization_case_config,
    benchmark,
    row_graph: str,
    column_graph: str,
    row_topology,
    column_topology,
    row_metropolis: np.ndarray,
    column_metropolis: np.ndarray,
    row_laplacian: np.ndarray,
    column_laplacian: np.ndarray,
    ops_module_name: str,
    system,
    data_wires,
    scaffold_edges,
    global_params,
    row_b_totals,
    true_solution,
    final_metrics,
    a_global,
    global_b,
):
    even_edges, odd_edges = _split_alternating_edges(scaffold_edges)
    with path.open("w", encoding="utf-8") as out:
        metadata = getattr(system, "metadata", {})
        out.write("5-local-qubit Ising necessity-comparison run\n")
        out.write(f"static_ops: {ops_module_name}\n")
        out.write(f"system name: {getattr(system, 'name', 'unknown')}\n")
        out.write(f"benchmark: {benchmark.benchmark_id if benchmark else 'custom'}\n")
        if benchmark is not None:
            out.write(f"benchmark name: {benchmark.name}\n")
            out.write(f"benchmark keywords: {', '.join(benchmark.keywords)}\n")
        out.write(f"n agents: {system.n}\n")
        out.write(f"local data qubits: {len(data_wires)}\n")
        out.write(f"global total qubits: {int(np.log2(a_global.shape[0]))}\n")
        out.write(f"ansatz: {ANSATZ_PAPER_FIG3_RY_CZ} ({describe_ansatz(ANSATZ_PAPER_FIG3_RY_CZ)})\n")
        out.write("ansatz block: RY(theta_1) -> CZ(even bonds) -> RY(theta_2) -> CZ(odd bonds)\n")
        out.write(f"layers: {int(args.layers)}\n")
        out.write(f"learning rate: {float(args.lr):.8f}\n")
        out.write(f"decay: {float(args.decay):.8f}\n")
        out.write(f"seed: {int(args.seed)}\n")
        out.write(f"optimization case: {optimization_case_id}\n")
        out.write(f"optimization label: {optimization_case_config['label']}\n")
        out.write(f"gradient exchange: {optimization_case_config['exchange']}\n")
        out.write(f"X-group optimizer (alpha/sigma): {optimization_case_config['x_optimizer']}\n")
        out.write(f"Z-group optimizer (beta/lambda): {optimization_case_config['z_optimizer']}\n")
        out.write(f"row graph: {row_graph} ({graph_label(row_graph)})\n")
        out.write(f"column graph: {column_graph} ({graph_label(column_graph)})\n")
        out.write(f"row topology: {row_topology}\n")
        out.write(f"column topology: {column_topology}\n")
        out.write(f"first-agent cost formula: {first_agent_cost_formula(row_graph)}\n")
        out.write(f"scaffold_edges: {tuple((int(a), int(b)) for a, b in scaffold_edges)}\n")
        out.write(f"even_bond_cz_edges: {tuple((int(a), int(b)) for a, b in even_edges)}\n")
        out.write(f"odd_bond_cz_edges: {tuple((int(a), int(b)) for a, b in odd_edges)}\n")
        out.write(f"row Metropolis matrix:\n{_format_array_for_report(row_metropolis)}\n")
        out.write(f"column Metropolis matrix:\n{_format_array_for_report(column_metropolis)}\n")
        out.write(f"row Laplacian:\n{_format_array_for_report(row_laplacian)}\n")
        out.write(f"column Laplacian:\n{_format_array_for_report(column_laplacian)}\n")
        if metadata:
            out.write(f"metadata: {_to_jsonable(metadata)}\n")
        out.write(f"condition number of A: {float(np.linalg.cond(a_global)):.8e}\n")
        out.write(f"global matrix shape: {a_global.shape}\n")
        out.write(f"global b shape: {global_b.shape}\n")
        out.write(f"final global cost: {float(final_metrics['global_cost']):.8e}\n")
        out.write(f"final ||Ax-b||: {float(final_metrics['residual_norm']):.8e}\n")
        out.write(f"final relative L2 error: {float(final_metrics['l2_error']):.8e}\n")
        out.write(f"final consensus error: {float(final_metrics['consensus_error']):.8e}\n")
        out.write(f"final recovered solution:\n{_format_array_for_report(final_metrics['recovered_solution'])}\n")
        out.write(f"true solution:\n{_format_array_for_report(true_solution)}\n")
        out.write(f"global b:\n{_format_array_for_report(global_b)}\n")
        out.write(f"final params summary: {_to_jsonable(global_params)}\n")
        out.write("\nROW RIGHT-HAND SIDES\n")
        for row_id, row_b in enumerate(row_b_totals):
            out.write(f"[row={row_id}] b_i:\n{_format_array_for_report(row_b)}\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--static_ops", required=True, help="Module name or .py path for the 4x4 static-ops file.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--benchmark", type=str, default="", help="Benchmark id, e.g. B1 ... B9.")
    ap.add_argument("--topology", type=str, default="", help="Use the same graph for rows and columns.")
    ap.add_argument("--row_graph", type=str, default="", help="Row graph override (P4, C4, K4).")
    ap.add_argument("--column_graph", type=str, default="", help="Column graph override (P4, C4, K4).")
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--decay", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=40)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--use_qjit", type=str, default="true")
    ap.add_argument("--optimization_case", type=str, default="track_adamX_adamZ")
    args = ap.parse_args(argv)

    if args.layers < 1:
        raise ValueError("--layers must be at least 1.")
    requested_qjit = _parse_optional_bool(args.use_qjit)
    if requested_qjit is None:
        requested_qjit = True
    optimization_case_id, optimization_case_config = resolve_optimization_case(args.optimization_case)
    args.optimization_case = optimization_case_id

    benchmark, row_graph, column_graph = resolve_benchmark_and_graphs(
        benchmark_id=args.benchmark,
        topology=args.topology,
        row_graph=args.row_graph,
        column_graph=args.column_graph,
    )

    ops = load_static_ops(args.static_ops)
    system, data_wires = load_problem_system_and_wires(ops)
    scaffold_edges = _default_scaffold_edges(len(data_wires))
    even_edges, odd_edges = _split_alternating_edges(scaffold_edges)

    np.random.seed(args.seed)
    jax.config.update("jax_enable_x64", True)

    row_topology = build_neighbor_map(row_graph, n_nodes=int(system.n))
    column_topology = build_neighbor_map(column_graph, n_nodes=int(system.n))
    row_metropolis = metropolis_matrix_from_topology(row_topology)
    column_metropolis = metropolis_matrix_from_topology(column_topology)
    row_laplacian = laplacian_matrix_from_topology(row_topology)
    column_laplacian = laplacian_matrix_from_topology(column_topology)

    paths = make_run_dir(args.out)
    logger = setup_logger(paths.report_txt)
    metrics = JsonlWriter(paths.metrics_jsonl)

    a_global = np.asarray(system.get_global_matrix(), dtype=np.complex128)
    global_b = np.asarray(system.get_global_b_vector(), dtype=np.complex128)
    true_solution = np.linalg.solve(a_global, global_b)
    row_b_totals = [np.asarray(system.get_b_vectors(row_id)[0], dtype=np.complex128) for row_id in range(int(system.n))]
    spectrum = getattr(ops, "SPECTRUM_INFO", getattr(system, "metadata", {}).get("spectrum"))

    logger.info(f"System variant: {getattr(system, 'name', 'unknown')}")
    logger.info(f"static_ops: {args.static_ops}")
    logger.info(f"benchmark: {benchmark.benchmark_id if benchmark else 'custom'}")
    if benchmark is not None:
        logger.info(f"benchmark name: {benchmark.name}")
        logger.info(f"benchmark keywords: {', '.join(benchmark.keywords)}")
    logger.info(f"row graph: {row_graph} ({graph_label(row_graph)})")
    logger.info(f"column graph: {column_graph} ({graph_label(column_graph)})")
    logger.info(f"row topology: {row_topology}")
    logger.info(f"column topology: {column_topology}")
    logger.info("row Metropolis matrix W_row =\n" + str(row_metropolis))
    logger.info("column Metropolis matrix W_col =\n" + str(column_metropolis))
    logger.info("row Laplacian L_row =\n" + str(row_laplacian))
    logger.info("column Laplacian L_col =\n" + str(column_laplacian))
    logger.info(f"first-agent cost formula: {first_agent_cost_formula(row_graph)}")
    logger.info(f"optimization case: {optimization_case_id} ({optimization_case_config['label']})")
    logger.info(f"gradient exchange: {optimization_case_config['exchange']}")
    logger.info("parameter groups: X=(alpha,sigma), Z=(beta,lambda)")
    logger.info(
        "optimizer assignment: "
        f"X={optimization_case_config['x_optimizer']} | "
        f"Z={optimization_case_config['z_optimizer']}"
    )
    logger.info(f"ansatz: {ANSATZ_PAPER_FIG3_RY_CZ} ({describe_ansatz(ANSATZ_PAPER_FIG3_RY_CZ)})")
    logger.info("ansatz block: RY(theta_1) -> CZ(even bonds) -> RY(theta_2) -> CZ(odd bonds)")
    logger.info(f"scaffold edges: {scaffold_edges}")
    logger.info(f"even-bond CZ edges: {even_edges}")
    logger.info(f"odd-bond CZ edges: {odd_edges}")
    logger.info(f"layers: {int(args.layers)}")
    logger.info(f"local data qubits: {len(data_wires)}")
    logger.info(f"global matrix shape: {a_global.shape}")
    logger.info(f"global b shape: {global_b.shape}")
    logger.info(f"condition number of A: {float(np.linalg.cond(a_global)):.8f}")
    if spectrum is not None:
        logger.info(
            "analytic spectrum: "
            f"lambda_min={float(spectrum['lambda_min']):.8f}, "
            f"lambda_max={float(spectrum['lambda_max']):.8f}, "
            f"cond(A)={float(spectrum['condition_number']):.8f}"
        )
    if not CATALYST_AVAILABLE:
        logger.info(
            "Catalyst import failed in this environment; falling back to plain JAX without qjit. "
            f"Reason: {_summarize_exception(CATALYST_IMPORT_ERROR)}"
        )
    qjit_enabled = bool(requested_qjit and CATALYST_AVAILABLE)
    if not requested_qjit:
        logger.info("qjit disabled by CLI; using plain JAX loss/grad path.")

    run_config = {
        "static_ops": args.static_ops,
        "out": str(paths.run_dir),
        "benchmark": None if benchmark is None else benchmark.benchmark_id,
        "row_graph": row_graph,
        "column_graph": column_graph,
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "lr": float(args.lr),
        "decay": float(args.decay),
        "log_every": int(args.log_every),
        "layers": int(args.layers),
        "optimization_case": optimization_case_id,
        "optimization_case_label": optimization_case_config["label"],
        "gradient_exchange": optimization_case_config["exchange"],
        "x_group_keys": list(X_KEYS),
        "z_group_keys": list(Z_KEYS),
        "x_optimizer": optimization_case_config["x_optimizer"],
        "z_optimizer": optimization_case_config["z_optimizer"],
        "requested_qjit": bool(requested_qjit),
        "qjit_enabled": bool(qjit_enabled),
        "ansatz_kind": ANSATZ_PAPER_FIG3_RY_CZ,
        "ansatz_description": describe_ansatz(ANSATZ_PAPER_FIG3_RY_CZ),
        "scaffold_edges": list(scaffold_edges),
        "even_bond_edges": list(even_edges),
        "odd_bond_edges": list(odd_edges),
        "row_topology": row_topology,
        "column_topology": column_topology,
        "row_metropolis": row_metropolis.tolist(),
        "column_metropolis": column_metropolis.tolist(),
        "row_laplacian": row_laplacian.tolist(),
        "column_laplacian": column_laplacian.tolist(),
        "first_agent_cost_formula": first_agent_cost_formula(row_graph),
    }
    with (paths.run_dir / "config_used.json").open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(run_config), handle, ensure_ascii=False, indent=2)

    global_params, _ = initialize_paper_fig3_params_jax(
        system,
        n_qubits=len(data_wires),
        layers=int(args.layers),
        seed=int(args.seed),
    )

    logger.info("Starting prebuild_local_evals() for distributed Hadamard-test bundles.")
    t_prebuild = time.time()
    ib.prebuild_local_evals(
        system,
        row_topology,
        n_input_qubit=len(data_wires),
        ansatz_kind=ANSATZ_PAPER_FIG3_RY_CZ,
        scaffold_edges=scaffold_edges,
        diff_method="adjoint",
        interface="jax",
    )
    logger.info(f"Finished prebuild_local_evals() in {time.time() - t_prebuild:.2f} s.")

    def total_loss_fn_plain(args_flat):
        current_params = rebuild_global_params(args_flat, system.n, global_params["b_norm"])
        return ib.eval_total_loss_plain(current_params)

    compute_loss_and_grad = None
    if qjit_enabled:
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

        logger.info("Optimization backend: qjit(catalyst.grad) + qjit(loss).")
    else:
        compute_loss_and_grad = jax.jit(jax.value_and_grad(total_loss_fn_plain))

        def compute_grad(args_flat):
            _, grad = compute_loss_and_grad(args_flat)
            return grad

        def compute_loss(args_flat):
            loss, _ = compute_loss_and_grad(args_flat)
            return loss

        logger.info("Optimization backend: jax.jit(value_and_grad).")

    def safe_compute_grad(args_flat):
        nonlocal compute_grad, compute_loss_and_grad
        try:
            return compute_grad(args_flat)
        except Exception as exc:
            if not qjit_enabled:
                raise
            logger.info(
                "Catalyst gradient compilation failed for this necessity sweep; "
                "keeping qjit(loss) and falling back to jax.jit(value_and_grad) for gradients only. "
                f"Reason: {_summarize_exception(exc)}"
            )
            compute_loss_and_grad = jax.jit(jax.value_and_grad(total_loss_fn_plain))

            def fallback_compute_grad(args_flat_inner):
                _, grad = compute_loss_and_grad(args_flat_inner)
                return grad

            compute_grad = fallback_compute_grad
            return compute_grad(args_flat)

    get_local_state = get_local_state_builder(len(data_wires), scaffold_edges)

    flat_params_init = to_jax_flat(flatten_params(global_params, keys=None))
    logger.info("Starting initial forward loss evaluation.")
    t_init_loss = time.time()
    current_loss = compute_loss(flat_params_init)
    logger.info(f"Finished initial forward loss evaluation in {time.time() - t_init_loss:.2f} s.")
    logger.info(f"[Init] Initial Loss = {float(current_loss):.5e}")

    lr_schedule = optax.exponential_decay(
        init_value=float(args.lr),
        transition_steps=1,
        decay_rate=float(args.decay),
        staircase=False,
    )
    opt_x = make_optimizer(optimization_case_config["x_optimizer"], lr_schedule)
    opt_z = make_optimizer(optimization_case_config["z_optimizer"], lr_schedule)

    loss_history = [float(current_loss)]
    metric_epochs = []
    residual_history = []
    l2_history = []
    consensus_history = []
    final_global_cost = float(current_loss)

    tracker_grid = None
    prev_grad_grid = None
    if optimization_case_config["use_tracking"]:
        logger.info("Starting initial gradient evaluation for tracker initialization.")
        t_init_grad = time.time()
        grads_flat_init = safe_compute_grad(flat_params_init)
        logger.info(f"Finished initial gradient evaluation in {time.time() - t_init_grad:.2f} s.")

        grad_grid_init = rebuild_global_params(grads_flat_init, system.n, global_params["b_norm"])
        tracker_grid = init_tracker_from_grad(grad_grid_init)
        prev_grad_grid = copy.deepcopy(grad_grid_init)
    else:
        logger.info("Gradient-sharing case selected; tracker initialization skipped.")

    opt_x_state = opt_x.init(to_jax_flat(flatten_params(global_params, keys=X_KEYS)))
    opt_z_state = opt_z.init(to_jax_flat(flatten_params(global_params, keys=Z_KEYS)))

    def apply_group_updates(update_grid):
        nonlocal opt_x_state, opt_z_state

        flat_updates_x = to_jax_flat(flatten_params(update_grid, keys=X_KEYS))
        flat_params_x = to_jax_flat(flatten_params(global_params, keys=X_KEYS))
        x_updates, opt_x_state = opt_x.update(flat_updates_x, opt_x_state, params=flat_params_x)
        new_flat_params_x = optax.apply_updates(flat_params_x, x_updates)
        update_global_from_flat(global_params, new_flat_params_x, keys=X_KEYS)

        flat_updates_z = to_jax_flat(flatten_params(update_grid, keys=Z_KEYS))
        flat_params_z = to_jax_flat(flatten_params(global_params, keys=Z_KEYS))
        z_updates, opt_z_state = opt_z.update(flat_updates_z, opt_z_state, params=flat_params_z)
        new_flat_params_z = optax.apply_updates(flat_params_z, z_updates)
        update_global_from_flat(global_params, new_flat_params_z, keys=Z_KEYS)

    t0 = time.time()
    last_log_time = t0
    last_metrics = None

    for epoch in range(1, args.epochs + 1):
        global_params = consensus_mix_metropolis_jax(global_params, W_np=column_metropolis)

        if optimization_case_config["use_tracking"]:
            apply_group_updates(tracker_grid)

            flat_params_all = to_jax_flat(flatten_params(global_params, keys=None))
            grads_flat = safe_compute_grad(flat_params_all)
            current_cost = compute_loss(flat_params_all)

            current_grad_grid = rebuild_global_params(grads_flat, system.n, global_params["b_norm"])
            tracker_grid = update_gradient_tracker_metropolis_jax(
                current_tracker=tracker_grid,
                current_grads=current_grad_grid,
                prev_grads=prev_grad_grid,
                W_np=column_metropolis,
            )
            prev_grad_grid = copy.deepcopy(current_grad_grid)
        else:
            flat_params_mixed = to_jax_flat(flatten_params(global_params, keys=None))
            grads_flat = safe_compute_grad(flat_params_mixed)
            current_grad_grid = rebuild_global_params(grads_flat, system.n, global_params["b_norm"])
            if optimization_case_config.get("mix_gradients", False):
                update_grid = consensus_mix_gradients_metropolis_jax(
                    current_grad_grid,
                    W_np=column_metropolis,
                )
            else:
                update_grid = current_grad_grid
            apply_group_updates(update_grid)

            flat_params_all = to_jax_flat(flatten_params(global_params, keys=None))
            current_cost = compute_loss(flat_params_all)

        final_global_cost = float(current_cost)
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
            current_lr = float(lr_schedule(epoch - 1))
            timestamp_utc = datetime.now(timezone.utc).isoformat()
            timestamp_local = datetime.now().astimezone().isoformat()

            metrics.write(
                {
                    "epoch": epoch,
                    "benchmark": None if benchmark is None else benchmark.benchmark_id,
                    "row_graph": row_graph,
                    "column_graph": column_graph,
                    "optimization_case": optimization_case_id,
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
                f"[Epoch {epoch:05d}] Cost = {float(current_cost):.5e} | "
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

    title = (
        f"{benchmark.benchmark_id if benchmark else 'custom'} | case={optimization_case_id} | "
        f"row={row_graph} | col={column_graph} | "
        f"lr0={args.lr:g} | {ANSATZ_PAPER_FIG3_RY_CZ}"
    )

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

    with (paths.run_dir / "final_params.json").open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(global_params), handle, ensure_ascii=False, indent=2)

    analysis_path = paths.run_dir / "analysis.txt"
    write_analysis_report(
        analysis_path,
        args=args,
        optimization_case_id=optimization_case_id,
        optimization_case_config=optimization_case_config,
        benchmark=benchmark,
        row_graph=row_graph,
        column_graph=column_graph,
        row_topology=row_topology,
        column_topology=column_topology,
        row_metropolis=row_metropolis,
        column_metropolis=column_metropolis,
        row_laplacian=row_laplacian,
        column_laplacian=column_laplacian,
        ops_module_name=args.static_ops,
        system=system,
        data_wires=data_wires,
        scaffold_edges=scaffold_edges,
        global_params=global_params,
        row_b_totals=row_b_totals,
        true_solution=true_solution,
        final_metrics={
            **last_metrics,
            "global_cost": final_global_cost,
        },
        a_global=a_global,
        global_b=global_b,
    )
    logger.info(f"Analysis report written to: {analysis_path}")
    logger.info(f"Finished. Outputs written to: {paths.run_dir}")


if __name__ == "__main__":
    main()
