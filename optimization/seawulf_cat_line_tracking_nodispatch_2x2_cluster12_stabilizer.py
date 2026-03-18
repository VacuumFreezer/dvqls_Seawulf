import argparse
import copy
import importlib
import sys
import time
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
    ANSATZ_CLUSTER_RZ_LOCAL_RY,
    ANSATZ_CLUSTER_RZ,
    VALID_ANSATZ_KINDS,
    apply_selected_ansatz,
    describe_ansatz,
    normalize_ansatz_kind,
)


TOPOLOGY_LINE_2 = {0: [1], 1: [0]}


def load_static_ops(module_name: str):
    return importlib.import_module(module_name)


def load_problem_system_and_wires(ops_module):
    if hasattr(ops_module, "SYSTEMS") and "2x2" in getattr(ops_module, "SYSTEMS"):
        system = ops_module.SYSTEMS["2x2"]
    elif hasattr(ops_module, "SYSTEM"):
        system = ops_module.SYSTEM
    else:
        raise RuntimeError(f"{ops_module.__name__} does not expose SYSTEM or SYSTEMS['2x2'].")

    if hasattr(ops_module, "DATA_WIRES_BY_SYSTEM") and "2x2" in getattr(ops_module, "DATA_WIRES_BY_SYSTEM"):
        data_wires = list(ops_module.DATA_WIRES_BY_SYSTEM["2x2"])
    elif hasattr(system, "data_wires"):
        data_wires = list(system.data_wires)
    elif hasattr(ops_module, "DATA_WIRES"):
        data_wires = list(ops_module.DATA_WIRES)
    else:
        raise RuntimeError(f"{ops_module.__name__} does not expose data wires for the 2x2 system.")

    if int(system.n) != 2:
        raise RuntimeError(f"This script expects a 2-agent system, got system.n={system.n}.")

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

    return tuple((left, left + 1) for left in range(1, int(n_qubits) - 1))


def _cluster_scaffold_ansatz(
    weights,
    n_qubits: int,
    *,
    ansatz_kind: str = ANSATZ_CLUSTER_RZ,
    repeat_cz_each_layer: bool = False,
    local_ry_support=(),
    scaffold_edges=None,
):
    apply_selected_ansatz(
        weights,
        n_qubits,
        ansatz_kind=ansatz_kind,
        repeat_cz_each_layer=repeat_cz_each_layer,
        local_ry_support=local_ry_support,
        scaffold_edges=scaffold_edges,
    )


def initialize_cluster_params_jax(system, n_qubits: int, layers: int, seed: int = 0):
    n_agents = int(system.n)
    key = jax.random.PRNGKey(seed)
    global_params = {"alpha": [], "beta": [], "sigma": [], "lambda": [], "b_norm": []}
    metadata = getattr(system, "metadata", {})
    sigma_target = float(metadata.get("init_sigma_target", 1.0 / np.sqrt(2.0)))
    init_angle_fill = float(metadata.get("init_angle_fill", 0.0))
    agent_init_overrides = metadata.get("agent_init_overrides", {})

    for sys_id in range(n_agents):
        if hasattr(system, "get_local_b_norms"):
            local_b_norms = system.get_local_b_norms(sys_id)
        else:
            _, *b_individuals = system.get_b_vectors(sys_id)
            local_b_norms = [float(np.linalg.norm(b_local)) for b_local in b_individuals]

        row_alpha, row_beta = [], []
        row_sigma, row_lam = [], []
        row_bnorms = []

        for agent_id in range(n_agents):
            base = np.full((layers, n_qubits), init_angle_fill, dtype=np.float64)
            if not agent_init_overrides:
                if agent_id == 1:
                    base[0, 0] = np.pi
                    if n_qubits > 1:
                        base[0, 1] = np.pi
            else:
                overrides = agent_init_overrides.get(str(agent_id), agent_init_overrides.get(agent_id, {}))
                for wire, value in dict(overrides).items():
                    base[0, int(wire)] = float(value)

            key, sub_a = jax.random.split(key)
            key, sub_b = jax.random.split(key)
            key, sub_s = jax.random.split(key)
            key, sub_l = jax.random.split(key)

            a = jnp.asarray(base) + 0.05 * jax.random.normal(sub_a, shape=(layers, n_qubits), dtype=jnp.float64)
            b = jnp.asarray(base) + 0.05 * jax.random.normal(sub_b, shape=(layers, n_qubits), dtype=jnp.float64)
            s = jnp.asarray(
                sigma_target + 0.05 * jax.random.normal(sub_s, shape=(), dtype=jnp.float64),
                dtype=jnp.float64,
            )
            l = jnp.asarray(0.05 * jax.random.normal(sub_l, shape=(), dtype=jnp.float64), dtype=jnp.float64)

            row_alpha.append(a)
            row_beta.append(b)
            row_sigma.append(s)
            row_lam.append(l)
            row_bnorms.append(
                jax.lax.stop_gradient(jnp.asarray(float(local_b_norms[agent_id]), dtype=jnp.float64))
            )

        global_params["alpha"].append(row_alpha)
        global_params["beta"].append(row_beta)
        global_params["sigma"].append(row_sigma)
        global_params["lambda"].append(row_lam)
        global_params["b_norm"].append(row_bnorms)

    return global_params, key


def recover_global_solution(
    global_params,
    n_qubits,
    *,
    ansatz_kind: str = ANSATZ_CLUSTER_RZ,
    repeat_cz_each_layer: bool = False,
    local_ry_support=(),
    scaffold_edges=None,
):
    n_agents = GLOBAL_AGENTS_N
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="jax")
    def get_local_state_segment(weights):
        _cluster_scaffold_ansatz(
            weights,
            n_qubits,
            ansatz_kind=ansatz_kind,
            repeat_cz_each_layer=repeat_cz_each_layer,
            local_ry_support=local_ry_support,
            scaffold_edges=scaffold_edges,
        )
        return qml.state()

    row_estimates = []
    for i in range(n_agents):
        segs = []
        for j in range(n_agents):
            alpha = global_params["alpha"][i][j]
            sigma = global_params["sigma"][i][j]
            state = get_local_state_segment(alpha)
            segs.append(sigma * state)
        row_estimates.append(np.concatenate(segs, axis=0))

    return np.mean(np.stack(row_estimates), axis=0)


def cal_sol_diff(global_params, true_sol, n_qubits, recover_sol=None):
    if true_sol is None:
        return np.nan
    if recover_sol is None:
        recover_sol = recover_global_solution(global_params, n_qubits)

    true_sol_vec = np.asarray(true_sol).reshape(-1)
    diff = true_sol_vec - recover_sol
    return np.linalg.norm(diff) / np.linalg.norm(true_sol_vec)


def cal_ax_minus_b_norm(global_params, A_global, global_b, n_qubits, recover_sol=None):
    if A_global is None or global_b is None:
        return np.nan
    if recover_sol is None:
        recover_sol = recover_global_solution(global_params, n_qubits)

    b_vec = np.asarray(global_b).reshape(-1)
    residual = A_global @ recover_sol - b_vec
    return np.linalg.norm(residual)


def _count_local_qnodes(L: int, degree: int) -> int:
    m = int(degree) + 1
    return (L * L) + ((m + 1) * L) + m + (m * (m - 1) // 2)


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
            L = len(system.gates_grid[sys_id][agent_id])
            degree = len(row_topology[agent_id])
            qnode_evals = _count_local_qnodes(L, degree)
            total_qnodes += qnode_evals
            entry_details.append(
                {
                    "sys_id": sys_id,
                    "agent_id": agent_id,
                    "L": L,
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


def _fmt_metric(x):
    if x is None:
        return "n/a"
    if isinstance(x, float) and np.isnan(x):
        return "n/a"
    return f"{float(x):.5e}"


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


def _summarize_exception(exc: Exception, *, max_lines: int = 2, max_chars: int = 400) -> str:
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    if not lines:
        return f"{type(exc).__name__}"

    summary = " | ".join(lines[-max_lines:])
    if len(summary) > max_chars:
        summary = summary[-max_chars:]
    return f"{type(exc).__name__}: {summary}"


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
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


def _write_analysis(
    path: Path,
    ops_module,
    system,
    data_wires,
    mem_info,
    ops_module_name: str,
    global_params,
    *,
    ansatz_kind: str,
    repeat_cz_each_layer: bool,
    local_ry_support,
    scaffold_edges,
    dense_metrics_active: bool,
    A_global,
    global_b,
    true_sol,
):
    with open(path, "w", encoding="utf-8") as f:
        metadata = getattr(system, "metadata", {})
        n_total_qubits = metadata.get("n_total_qubits", len(data_wires) + 1)
        f.write(f"{n_total_qubits}-qubit cluster-stabilizer 2x2 distributed run\n")
        f.write(f"static_ops module: {ops_module_name}\n")
        f.write(f"system name: {getattr(system, 'name', 'unknown')}\n")
        f.write(f"n agents: {system.n}\n")
        f.write(f"local data qubits: {len(data_wires)}\n")
        f.write(f"ansatz: {ansatz_kind} ({describe_ansatz(ansatz_kind)})\n")
        f.write(f"repeat_cz_each_layer: {bool(repeat_cz_each_layer)}\n")
        f.write(f"local_ry_support: {tuple(int(x) for x in local_ry_support)}\n")
        f.write(f"scaffold_edges: {tuple((int(a), int(b)) for a, b in scaffold_edges)}\n")
        f.write(f"dense validation available: {getattr(system, 'supports_dense_validation', True)}\n")
        f.write(f"statevector per Hadamard test: {mem_info['statevector_human']}\n")
        f.write(f"conservative peak per Hadamard test: {mem_info['peak_human_conservative']}\n")
        f.write(f"qnode evals per loss evaluation: {mem_info['total_qnode_evals_per_loss']}\n")
        f.write(f"sequential statevector traffic per loss evaluation: {mem_info['sequential_human_per_loss_eval']}\n")
        for item in mem_info["entry_details"]:
            f.write(
                f"entry(sys={item['sys_id']}, agent={item['agent_id']}): "
                f"L={item['L']}, degree={item['degree']}, qnodes={item['qnode_evals']}\n"
            )
        if metadata:
            f.write(f"metadata: {metadata}\n")

        summary = _global_param_summary(global_params)
        f.write(f"final parameter summary: {summary}\n")

        if dense_metrics_active and A_global is not None and global_b is not None and true_sol is not None:
            recover_sol = recover_global_solution(
                global_params,
                n_qubits=len(data_wires),
                ansatz_kind=ansatz_kind,
                repeat_cz_each_layer=repeat_cz_each_layer,
                local_ry_support=local_ry_support,
                scaffold_edges=scaffold_edges,
            )
            sol_diff = cal_sol_diff(
                global_params,
                true_sol=true_sol,
                n_qubits=len(data_wires),
                recover_sol=recover_sol,
            )
            ax_minus_b = cal_ax_minus_b_norm(
                global_params,
                A_global=A_global,
                global_b=global_b,
                n_qubits=len(data_wires),
                recover_sol=recover_sol,
            )
            f.write(f"final recovered solution norm: {float(np.linalg.norm(recover_sol)):.8e}\n")
            f.write(f"final true solution norm: {float(np.linalg.norm(np.asarray(true_sol).reshape(-1))):.8e}\n")
            f.write(f"final sol_diff: {float(sol_diff):.8e}\n")
            f.write(f"final ||Ax-b||: {float(ax_minus_b):.8e}\n")

            n_agents = int(system.n)
            dim = 2 ** len(data_wires)
            dev = qml.device("lightning.qubit", wires=len(data_wires))

            @qml.qnode(dev, interface="jax")
            def get_state(weights):
                _cluster_scaffold_ansatz(
                    weights,
                    len(data_wires),
                    ansatz_kind=ansatz_kind,
                    repeat_cz_each_layer=repeat_cz_each_layer,
                    local_ry_support=local_ry_support,
                    scaffold_edges=scaffold_edges,
                )
                return qml.state()

            f.write("\n" + "=" * 60 + "\n")
            f.write(f"      SOLUTION RECOVERY & VERIFICATION (N={n_agents})\n")
            f.write("=" * 60 + "\n")

            for sys_id in range(n_agents):
                true_b_sum = system.get_b_vectors(sys_id)[0]
                row_recovered_Ax = np.zeros(dim, dtype=complex)
                f.write(f"\n>>> SYSTEM {sys_id} (Row {sys_id}) <<<\n")
                for agent_id in range(n_agents):
                    alpha = global_params["alpha"][sys_id][agent_id]
                    beta = global_params["beta"][sys_id][agent_id]
                    sigma = float(np.asarray(global_params["sigma"][sys_id][agent_id]))
                    state_x = np.array(get_state(alpha))
                    r0, r1 = sys_id * dim, (sys_id + 1) * dim
                    c0, c1 = agent_id * dim, (agent_id + 1) * dim
                    row_recovered_Ax += sigma * (A_global[r0:r1, c0:c1] @ state_x)
                    f.write(f"  [Agent {agent_id}]\n")
                    f.write(f"alpha: {_format_array_for_report(alpha)}\n")
                    f.write(f"beta: {_format_array_for_report(beta)}\n")
                    f.write(f"    |x>: {_format_array_for_report(sigma * state_x)}\n")
                row_residual = float(np.linalg.norm(row_recovered_Ax - true_b_sum))
                rel_err = row_residual / float(np.linalg.norm(true_b_sum))
                f.write("-" * 40 + "\n")
                f.write(f"  Recovered b_total (Sum sigma Ax): {_format_array_for_report(np.round(row_recovered_Ax, 4), precision=4)}\n")
                f.write(f"  True b_total               : {_format_array_for_report(np.round(true_b_sum, 4), precision=4)}\n")
                f.write(f"  > L2 Difference: {row_residual:.5e}\n")
                f.write(f"  > Relative Err : {rel_err:.2%}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("      GLOBAL RECOVERY\n")
            f.write("=" * 60 + "\n")
            f.write(f"final recovered x: {_format_array_for_report(recover_sol)}\n")
            f.write(f"true solution x : {_format_array_for_report(np.asarray(true_sol).reshape(-1))}\n")
            f.write(f"true b          : {_format_array_for_report(np.asarray(global_b).reshape(-1))}\n")
        else:
            f.write(
                "dense post-analysis was skipped because dense validation is disabled for this run or unavailable "
                "for this system.\n"
            )

        if hasattr(ops_module, "write_structured_analysis"):
            ops_module.write_structured_analysis(
                out=f,
                global_params=global_params,
                ansatz_kind=ansatz_kind,
                repeat_cz_each_layer=repeat_cz_each_layer,
                local_ry_support=tuple(int(x) for x in local_ry_support),
                scaffold_edges=tuple((int(a), int(b)) for a, b in scaffold_edges),
            )


def _write_final_params_json(path: Path, global_params):
    with open(path, "w", encoding="utf-8") as f:
        json_ready = _to_jsonable(
            {
                "alpha": global_params["alpha"],
                "beta": global_params["beta"],
                "sigma": global_params["sigma"],
                "lambda": global_params["lambda"],
                "b_norm": global_params["b_norm"],
                "summary": _global_param_summary(global_params),
            }
        )
        import json

        json.dump(json_ready, f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--static_ops", required=True, help="e.g. problems.static_ops_2x2_cluster12_stabilizer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topology", type=str, default="line")
    ap.add_argument("--system_id", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--decay", type=float, default=0.9999)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--ansatz", type=str, default=ANSATZ_CLUSTER_RZ, choices=VALID_ANSATZ_KINDS)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--repeat_cz_each_layer", action="store_true")
    ap.add_argument(
        "--local_ry_support",
        type=str,
        default="",
        help="Comma-separated local wire ids for the extra local RY correction ansatz.",
    )
    ap.add_argument(
        "--enable_dense_metrics",
        action="store_true",
        help="Force-enable sol_diff and ||Ax-b|| using dense state recovery when the system supports it.",
    )
    ap.add_argument(
        "--disable_dense_metrics",
        action="store_true",
        help="Force-disable sol_diff and ||Ax-b|| even when the system supports dense validation.",
    )
    args = ap.parse_args(argv)
    args.ansatz = normalize_ansatz_kind(args.ansatz)
    if args.layers < 1:
        raise ValueError("--layers must be at least 1.")

    ops = load_static_ops(args.static_ops)
    SYSTEM, DATA_WIRES = load_problem_system_and_wires(ops)
    local_ry_support = resolve_local_ry_support(ops, args.local_ry_support)
    scaffold_edges = resolve_scaffold_edges(ops, len(DATA_WIRES))
    if args.ansatz in (ANSATZ_CLUSTER_RZ_LOCAL_RY, ANSATZ_CLUSTER_LOCAL_RY) and not local_ry_support:
        raise ValueError(
            f"Ansatz `{args.ansatz}` requires support wires. "
            "Provide --local_ry_support or define LOCAL_RY_SUPPORT_WIRES in the static_ops module."
        )

    global GLOBAL_AGENTS_N
    GLOBAL_AGENTS_N = int(SYSTEM.n)

    pnp.random.seed(args.seed)
    np.random.seed(args.seed)
    jax.config.update("jax_enable_x64", True)

    paths = make_run_dir(args.out)
    logger = setup_logger(paths.report_txt)
    metrics = JsonlWriter(paths.metrics_jsonl)

    n_qubits = len(DATA_WIRES)
    layers = int(args.layers)

    spectrum = None
    if hasattr(ops, "SPECTRUM_INFO"):
        spectrum = getattr(ops, "SPECTRUM_INFO")
    elif hasattr(SYSTEM, "metadata") and "spectrum" in SYSTEM.metadata:
        spectrum = SYSTEM.metadata["spectrum"]

    logger.info(f"System variant: {getattr(SYSTEM, 'name', '2x2')}")
    logger.info(f"static_ops module: {args.static_ops}")
    logger.info(f"local data qubits per agent: {n_qubits}")
    logger.info(f"ansatz: {args.ansatz} ({describe_ansatz(args.ansatz)})")
    logger.info(f"layers: {layers}")
    logger.info(f"repeat_cz_each_layer: {bool(args.repeat_cz_each_layer)}")
    logger.info(f"local_ry_support: {tuple(int(x) for x in local_ry_support)}")
    logger.info(f"scaffold_edges: {tuple((int(a), int(b)) for a, b in scaffold_edges)}")
    logger.info(f"term counts by block: {[[len(cell) for cell in row] for row in SYSTEM.gates_grid]}")
    if not CATALYST_AVAILABLE:
        logger.info(
            "Catalyst import failed in this environment; falling back to plain JAX without qjit. "
            f"Reason: {type(CATALYST_IMPORT_ERROR).__name__}: {CATALYST_IMPORT_ERROR}"
        )
    if spectrum is not None:
        logger.info(
            "Analytic spectrum: "
            f"lambda_min={spectrum['lambda_min']:.8f}, "
            f"lambda_max={spectrum['lambda_max']:.8f}, "
            f"cond(A)={spectrum['condition_number']:.8f}"
        )

    mem_info = estimate_loss_memory_usage(SYSTEM, TOPOLOGY_LINE_2, n_qubits)
    logger.info(
        f"Per Hadamard-test statevector ({mem_info['n_wires_per_hadamard_test']} wires): {mem_info['statevector_human']}"
    )
    logger.info(f"Conservative peak per Hadamard test: {mem_info['peak_human_conservative']}")
    logger.info(f"QNode evaluations per compute_loss call: {mem_info['total_qnode_evals_per_loss']}")
    logger.info(
        f"Sequential statevector traffic per compute_loss call: {mem_info['sequential_human_per_loss_eval']}"
    )
    logger.info("These memory figures are for compute_loss only; compute_grad will be higher.")

    if args.enable_dense_metrics and args.disable_dense_metrics:
        raise ValueError("Use at most one of --enable_dense_metrics or --disable_dense_metrics.")

    dense_validation_supported = bool(getattr(SYSTEM, "supports_dense_validation", True))
    if args.disable_dense_metrics:
        dense_metrics_active = False
    elif args.enable_dense_metrics:
        dense_metrics_active = dense_validation_supported
    else:
        dense_metrics_active = dense_validation_supported

    if args.enable_dense_metrics and not dense_metrics_active:
        logger.info(
            "Dense validation metrics were requested but are unavailable for this system. sol_diff and ||Ax-b|| will be logged as n/a."
        )
    elif args.disable_dense_metrics:
        logger.info(
            "Dense validation metrics were explicitly disabled for this run. sol_diff and ||Ax-b|| will be logged as n/a."
        )
    elif dense_metrics_active:
        logger.info(
            "Dense validation metrics are enabled for this run because the loaded system supports dense validation."
        )
    else:
        logger.info(
            "Dense validation metrics are unavailable for this system. sol_diff and ||Ax-b|| will be logged as n/a."
        )

    A_global = None
    global_b = None
    true_sol = None
    if dense_metrics_active:
        A_global = SYSTEM.get_global_matrix()
        global_b = SYSTEM.get_global_b_vector()
        true_sol = np.linalg.solve(A_global, global_b.T)
        logger.info(f"Dense validation matrix shape: {A_global.shape}")

    GLOBAL_PARAMS, _ = initialize_cluster_params_jax(
        SYSTEM, n_qubits=n_qubits, layers=layers, seed=args.seed
    )

    Wm = build_metropolis_matrix(TOPOLOGY_LINE_2, n=SYSTEM.n, make_undirected=True)
    logger.info("Metropolis weight matrix Wm =\n" + str(Wm))

    logger.info("Starting prebuild_local_evals() for distributed Hadamard-test bundles.")
    t_prebuild = time.time()
    ib.prebuild_local_evals(
        SYSTEM,
        TOPOLOGY_LINE_2,
        n_input_qubit=n_qubits,
        ansatz_kind=args.ansatz,
        repeat_cz_each_layer=args.repeat_cz_each_layer,
        local_ry_support=local_ry_support,
        scaffold_edges=scaffold_edges,
        diff_method="adjoint",
        interface="jax",
    )
    logger.info(f"Finished prebuild_local_evals() in {time.time() - t_prebuild:.2f} s.")

    def total_loss_fn(args_flat):
        current_params = rebuild_global_params(args_flat, SYSTEM.n, GLOBAL_PARAMS["b_norm"])
        return ib.eval_total_loss(current_params)

    def compute_grad_jax(args_flat):
        return jax.grad(total_loss_fn)(args_flat)

    def compute_loss_plain(args_flat):
        return total_loss_fn(args_flat)

    if CATALYST_AVAILABLE:
        @qml.qjit
        def compute_grad_catalyst(args_flat):
            return catalyst.grad(total_loss_fn, method="auto")(args_flat)

        @qml.qjit
        def compute_loss_qjit(args_flat):
            return total_loss_fn(args_flat)

        use_catalyst_grad = True

        def compute_grad(args_flat):
            nonlocal use_catalyst_grad

            if use_catalyst_grad:
                try:
                    return compute_grad_catalyst(args_flat)
                except Exception as exc:
                    use_catalyst_grad = False
                    logger.info(
                        "Catalyst gradient compilation failed for this workload; "
                        "falling back to jax.grad while keeping the qjit'd forward loss. "
                        f"Summary: {_summarize_exception(exc)}"
                    )

            return compute_grad_jax(args_flat)

        def compute_loss(args_flat):
            return compute_loss_qjit(args_flat)
    else:
        def compute_grad(args_flat):
            return compute_grad_jax(args_flat)

        def compute_loss(args_flat):
            return compute_loss_plain(args_flat)

    logger.info("Starting initial forward loss evaluation.")
    t_init_loss = time.time()
    current_loss = compute_loss(to_jax_flat(flatten_params(GLOBAL_PARAMS)))
    logger.info(f"Finished initial forward loss evaluation in {time.time() - t_init_loss:.2f} s.")
    logger.info(f"[Init] Initial Loss = {float(current_loss):.5e}")

    lr_schedule = optax.exponential_decay(
        init_value=args.lr,
        transition_steps=1,
        decay_rate=args.decay,
        staircase=False,
    )
    opt_adam = optax.adam(lr_schedule)

    loss_history = [float(current_loss)]
    metric_epochs = []
    diff_history = []
    ax_minus_b_history = []

    flat_params_init = to_jax_flat(flatten_params(GLOBAL_PARAMS, keys=None))
    logger.info("Starting initial gradient evaluation.")
    t_init_grad = time.time()
    grads_flat_init = compute_grad(flat_params_init)
    logger.info(f"Finished initial gradient evaluation in {time.time() - t_init_grad:.2f} s.")
    grad_grid_init = rebuild_global_params(grads_flat_init, SYSTEM.n, GLOBAL_PARAMS["b_norm"])

    tracker_grid = init_tracker_from_grad(grad_grid_init)
    prev_grad_grid = copy.deepcopy(grad_grid_init)

    all_keys = ["alpha", "beta", "sigma", "lambda"]
    opt_adam_state = opt_adam.init(to_jax_flat(flatten_params(GLOBAL_PARAMS, keys=all_keys)))

    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        GLOBAL_PARAMS = consensus_mix_metropolis_jax(GLOBAL_PARAMS, W_np=Wm)

        flat_tracker = to_jax_flat(flatten_params(tracker_grid, keys=all_keys))
        flat_params = to_jax_flat(flatten_params(GLOBAL_PARAMS, keys=all_keys))
        adam_updates, opt_adam_state = opt_adam.update(flat_tracker, opt_adam_state, params=flat_params)
        new_flat_params = optax.apply_updates(flat_params, adam_updates)
        update_global_from_flat(GLOBAL_PARAMS, new_flat_params, keys=all_keys)

        flat_params_all = to_jax_flat(flatten_params(GLOBAL_PARAMS, keys=None))
        grads_flat = compute_grad(flat_params_all)
        current_cost = compute_loss(flat_params_all)

        current_grad_grid = rebuild_global_params(grads_flat, SYSTEM.n, GLOBAL_PARAMS["b_norm"])
        tracker_grid = update_gradient_tracker_metropolis_jax(
            current_tracker=tracker_grid,
            current_grads=current_grad_grid,
            prev_grads=prev_grad_grid,
            W_np=Wm,
        )
        prev_grad_grid = copy.deepcopy(current_grad_grid)

        loss_history.append(float(current_cost))

        if (ep % args.log_every) == 0 or ep == 1:
            sol_diff = None
            ax_minus_b = None
            if dense_metrics_active:
                recover_sol = recover_global_solution(
                    GLOBAL_PARAMS,
                    n_qubits=n_qubits,
                    ansatz_kind=args.ansatz,
                    repeat_cz_each_layer=args.repeat_cz_each_layer,
                    local_ry_support=local_ry_support,
                    scaffold_edges=scaffold_edges,
                )
                sol_diff = float(
                    cal_sol_diff(
                        GLOBAL_PARAMS,
                        true_sol=true_sol,
                        n_qubits=n_qubits,
                        recover_sol=recover_sol,
                    )
                )
                ax_minus_b = float(
                    cal_ax_minus_b_norm(
                        GLOBAL_PARAMS,
                        A_global=A_global,
                        global_b=global_b,
                        n_qubits=n_qubits,
                        recover_sol=recover_sol,
                    )
                )
                metric_epochs.append(ep)
                diff_history.append(sol_diff)
                ax_minus_b_history.append(ax_minus_b)

            metrics.write(
                {
                    "epoch": ep,
                    "loss": float(current_cost),
                    "sol_diff": sol_diff,
                    "ax_minus_b_norm": ax_minus_b,
                    "lr_g": float(args.lr),
                    "lr_a": float(args.lr),
                    "wall_s": time.time() - t0,
                }
            )
            logger.info(
                f"[Epoch {ep:04d}] Total Loss = {float(current_cost):.5e} | "
                f"Sol Diff = {_fmt_metric(sol_diff)} | ||Ax-b|| = {_fmt_metric(ax_minus_b)}"
            )

    metrics.close()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    title = f"{Path(__file__).stem} | lr0={args.lr:g}"
    if spectrum is not None:
        title = f"kappa≈{int(np.rint(spectrum['condition_number']))} | {title}"

    xs_loss = np.arange(0, len(loss_history))
    plt.figure()
    plt.plot(xs_loss, loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.yscale("log")
    plt.grid(True)
    plt.title(title)
    plt.savefig(paths.fig_loss, dpi=200, bbox_inches="tight")
    plt.close()

    if diff_history:
        plt.figure()
        plt.plot(metric_epochs, diff_history)
        plt.xlabel("Epoch")
        plt.ylabel("L2 error")
        plt.yscale("log")
        plt.grid(True)
        plt.title(title)
        plt.savefig(paths.fig_diff, dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(metric_epochs, ax_minus_b_history)
        plt.xlabel("Epoch")
        plt.ylabel("||Ax-b||")
        plt.yscale("log")
        plt.grid(True)
        plt.title(title)
        plt.savefig(paths.run_dir / "ax_minus_b_norm.png", dpi=200, bbox_inches="tight")
        plt.close()
    else:
        logger.info("Dense validation metrics were not computed; sol_diff.png and ax_minus_b_norm.png were not generated.")

    np.savez(
        paths.artifacts_npz,
        loss=np.array(loss_history),
        metric_epochs=np.array(metric_epochs),
        sol_diff=np.array(diff_history),
        ax_minus_b_norm=np.array(ax_minus_b_history),
    )

    final_params_path = paths.run_dir / "final_params.json"
    _write_final_params_json(final_params_path, GLOBAL_PARAMS)
    logger.info(f"Final parameters written to: {final_params_path}")

    analysis_path = paths.run_dir / "analysis.txt"
    _write_analysis(
        analysis_path,
        ops,
        SYSTEM,
        DATA_WIRES,
        mem_info,
        args.static_ops,
        GLOBAL_PARAMS,
        ansatz_kind=args.ansatz,
        repeat_cz_each_layer=args.repeat_cz_each_layer,
        local_ry_support=local_ry_support,
        scaffold_edges=scaffold_edges,
        dense_metrics_active=dense_metrics_active,
        A_global=A_global,
        global_b=global_b,
        true_sol=true_sol,
    )
    logger.info(f"Post-analysis written to: {analysis_path}")
    logger.info("Finished. Outputs written to: " + str(paths.run_dir))


if __name__ == "__main__":
    main()
