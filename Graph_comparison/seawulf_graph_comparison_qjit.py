from __future__ import annotations

import argparse
import copy
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Graph_comparison.topology_registry import (
    first_agent_cost_formula,
    graph_label,
    laplacian_matrix_from_topology,
    metropolis_matrix_from_topology,
    resolve_benchmark_and_graphs,
    build_neighbor_map,
)


def _identity_qjit(fn=None, *args, **kwargs):
    del args, kwargs
    if fn is None:
        return lambda f: f
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


from Partition_comparison_qjit.Old_asymmetry_stabilizer.seawulf_partition_comparison_qjit import (
    ANSATZ_BRICKWALL_RY_CZ,
    VALID_ANSATZ_KINDS,
    _format_array_for_report,
    _global_param_summary,
    _summarize_exception,
    _to_jsonable,
    _write_final_params_json,
    build_state_getter,
    compute_metric_bundle,
    compute_true_solution,
    estimate_loss_memory_usage,
    initialize_cluster_params_jax,
    load_problem_system_and_wires,
    load_static_ops,
    resolve_local_ry_support,
    resolve_scaffold_edges,
)
from common.DIGing_jax import (
    consensus_mix_metropolis_jax,
    init_tracker_from_grad,
    update_gradient_tracker_metropolis_jax,
)
from common.params_io import flatten_params, rebuild_global_params, update_global_from_flat
from common.reporting import JsonlWriter, make_run_dir, setup_logger
import objective.builder_cluster_nodispatch as ib
from objective.circuits_cluster_nodispatch import describe_ansatz, normalize_ansatz_kind


def to_jax_flat(flat_list):
    import jax.numpy as jnp

    return [jnp.asarray(x) for x in flat_list]


def _write_run_config(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, ensure_ascii=False, indent=2)


GROUP_KEYS = ("alpha", "beta", "sigma", "lambda")


def compute_group_norm_summary(group_grid, *, prefix: str) -> dict[str, object]:
    summary: dict[str, object] = {}
    total_sq = 0.0
    max_abs = 0.0

    for key in GROUP_KEYS:
        key_sq = 0.0
        key_agent_norms = []
        for row in group_grid[key]:
            row_agent_norms = []
            for leaf in row:
                arr = np.real_if_close(np.asarray(leaf, dtype=np.float64))
                row_agent_norms.append(float(np.linalg.norm(arr.reshape(-1))))
                key_sq += float(np.sum(np.square(arr)))
                if arr.size:
                    max_abs = max(max_abs, float(np.max(np.abs(arr))))
            key_agent_norms.append(row_agent_norms)
        summary[f"{prefix}_norm_{key}"] = float(np.sqrt(key_sq))
        summary[f"{prefix}_norm_{key}_by_agent"] = key_agent_norms
        total_sq += key_sq

    summary[f"{prefix}_norm_total"] = float(np.sqrt(total_sq))
    summary[f"{prefix}_max_abs"] = float(max_abs)
    return summary


def compute_gradient_norm_summary(grad_grid) -> dict[str, object]:
    return compute_group_norm_summary(grad_grid, prefix="grad")


def normalize_group_list(text: str) -> tuple[str, ...]:
    raw = [part.strip().lower() for part in str(text).split(",") if part.strip()]
    if not raw:
        return ()
    invalid = [part for part in raw if part not in GROUP_KEYS]
    if invalid:
        valid = ", ".join(GROUP_KEYS)
        raise ValueError(f"Unsupported parameter group(s) in freeze list: {invalid}. Expected subset of: {valid}")
    return tuple(dict.fromkeys(raw))


def mask_flat_group_entries(flat_list, freeze_groups: tuple[str, ...]) -> list[object]:
    if not freeze_groups:
        return list(flat_list)

    freeze_set = set(freeze_groups)
    masked = []
    n_groups = len(GROUP_KEYS)
    for idx, leaf in enumerate(flat_list):
        key = GROUP_KEYS[idx % n_groups]
        masked.append(jnp.zeros_like(leaf) if key in freeze_set else leaf)
    return masked


def subtract_flat_lists(lhs, rhs) -> list[object]:
    return [jnp.asarray(a) - jnp.asarray(b) for a, b in zip(lhs, rhs)]


def restore_frozen_groups(current_params, frozen_reference, freeze_groups: tuple[str, ...]) -> None:
    if not freeze_groups:
        return
    for key in freeze_groups:
        for row_id in range(len(current_params[key])):
            for col_id in range(len(current_params[key][row_id])):
                current_params[key][row_id][col_id] = frozen_reference[key][row_id][col_id]


def compute_zero_block_mask(system) -> list[list[bool]]:
    coeff_grid = getattr(system, "coeffs", getattr(system, "coeffs_grid", None))
    if coeff_grid is None:
        raise AttributeError("System does not expose `coeffs` or `coeffs_grid` for zero-block detection.")

    n_agents = int(system.n)
    zero_block_mask: list[list[bool]] = []
    for row_id in range(n_agents):
        mask_row = []
        for col_id in range(n_agents):
            coeffs = np.asarray(coeff_grid[row_id][col_id], dtype=np.float64)
            mask_row.append(bool(not np.any(np.abs(coeffs) > 1e-12)))
        zero_block_mask.append(mask_row)
    return zero_block_mask


def build_optimizer_labels(
    n_agents: int,
    *,
    relay_only_zero_blocks: bool,
    zero_block_mask: list[list[bool]],
) -> list[str]:
    labels: list[str] = []
    for row_id in range(n_agents):
        for col_id in range(n_agents):
            relay_here = bool(relay_only_zero_blocks and zero_block_mask[row_id][col_id])
            labels.append("relay" if relay_here else "alpha")
            labels.append("beta")
            labels.append("relay" if relay_here else "sigma")
            labels.append("lambda")
    return labels


def write_analysis_report(
    path: Path,
    *,
    args,
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
    mem_info,
    global_params,
    row_b_totals,
    true_solution,
    final_metrics,
) -> None:
    with path.open("w", encoding="utf-8") as out:
        metadata = {
            key: value
            for key, value in getattr(system, "metadata", {}).items()
            if key not in {"reference_row_gates", "exact_solution_gates_by_col"}
        }
        out.write("13-qubit graph-comparison run\n")
        out.write(f"static_ops: {ops_module_name}\n")
        out.write(f"system name: {getattr(system, 'name', 'unknown')}\n")
        out.write(f"benchmark: {benchmark.benchmark_id if benchmark else 'custom'}\n")
        if benchmark is not None:
            out.write(f"benchmark name: {benchmark.name}\n")
            out.write(f"benchmark keywords: {', '.join(benchmark.keywords)}\n")
        out.write(f"n agents: {system.n}\n")
        out.write(f"local data qubits: {len(data_wires)}\n")
        out.write(f"ansatz: {args.ansatz} ({describe_ansatz(args.ansatz)})\n")
        out.write(f"layers: {int(args.layers)}\n")
        out.write(f"repeat_cz_each_layer: {bool(args.repeat_cz_each_layer)}\n")
        out.write(f"freeze groups: {normalize_group_list(args.freeze_groups)}\n")
        out.write(f"freeze window: [{int(args.freeze_from_epoch)}, {int(args.freeze_until_epoch)}]\n")
        out.write(f"row graph: {row_graph} ({graph_label(row_graph)})\n")
        out.write(f"column graph: {column_graph} ({graph_label(column_graph)})\n")
        out.write(f"row topology: {row_topology}\n")
        out.write(f"column topology: {column_topology}\n")
        out.write(f"first-agent cost formula: {first_agent_cost_formula(row_graph)}\n")
        out.write(f"row Metropolis matrix:\n{_format_array_for_report(row_metropolis)}\n")
        out.write(f"column Metropolis matrix:\n{_format_array_for_report(column_metropolis)}\n")
        out.write(f"row Laplacian:\n{_format_array_for_report(row_laplacian)}\n")
        out.write(f"column Laplacian:\n{_format_array_for_report(column_laplacian)}\n")
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
        out.write("GLOBAL RECOVERY\n")
        out.write("=" * 80 + "\n")
        out.write(f"consensus-averaged x:\n{_format_array_for_report(final_metrics['recovered_solution'])}\n")
        out.write(f"true solution x*:\n{_format_array_for_report(true_solution)}\n")

        out.write("\n" + "=" * 80 + "\n")
        out.write("ROW RIGHT-HAND SIDES\n")
        out.write("=" * 80 + "\n")
        for row_id, row_b in enumerate(row_b_totals):
            out.write(f"[row={row_id}] b_i:\n{_format_array_for_report(row_b)}\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--static_ops", required=True, help="Module name or .py path for the 4x4 static-ops file.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--benchmark", type=str, default="", help="Benchmark id, e.g. B1 ... B6.")
    ap.add_argument("--topology", type=str, default="", help="Use the same graph for rows and columns.")
    ap.add_argument("--row_graph", type=str, default="", help="Row graph override (P4, K4, S4, C4).")
    ap.add_argument("--column_graph", type=str, default="", help="Column graph override (P4, K4, S4, C4).")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lr_alpha", type=float, default=None)
    ap.add_argument("--lr_beta", type=float, default=None)
    ap.add_argument("--lr_sigma", type=float, default=None)
    ap.add_argument("--lr_lambda", type=float, default=None)
    ap.add_argument("--decay", type=float, default=0.9999)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--ansatz", type=str, default=ANSATZ_BRICKWALL_RY_CZ, choices=VALID_ANSATZ_KINDS)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--repeat_cz_each_layer", action="store_true")
    ap.add_argument("--init_mode", type=str, default="uniform_pm_pi")
    ap.add_argument("--init_angle_center", type=float, default=(np.pi / 2.0))
    ap.add_argument("--init_angle_noise_std", type=float, default=0.05)
    ap.add_argument("--init_sigma_value", type=float, default=None)
    ap.add_argument("--init_sigma_noise_std", type=float, default=0.0)
    ap.add_argument("--init_lambda_value", type=float, default=None)
    ap.add_argument("--init_lambda_noise_std", type=float, default=0.0)
    ap.add_argument(
        "--relay_only_zero_blocks",
        action="store_true",
        help="For zero-coefficient blocks, let alpha/sigma evolve only through consensus and skip local optimizer steps.",
    )
    ap.add_argument(
        "--freeze_groups",
        type=str,
        default="",
        help="Comma-separated parameter groups to freeze during a selected epoch window, e.g. alpha,sigma.",
    )
    ap.add_argument("--freeze_from_epoch", type=int, default=0)
    ap.add_argument("--freeze_until_epoch", type=int, default=0)
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
    freeze_groups = normalize_group_list(args.freeze_groups)
    if (args.freeze_from_epoch or args.freeze_until_epoch) and not freeze_groups:
        raise ValueError("A non-empty --freeze_groups list is required when specifying a freeze epoch window.")
    if args.freeze_until_epoch and args.freeze_from_epoch and args.freeze_until_epoch < args.freeze_from_epoch:
        raise ValueError("--freeze_until_epoch must be >= --freeze_from_epoch.")

    benchmark, row_graph, column_graph = resolve_benchmark_and_graphs(
        benchmark_id=args.benchmark,
        topology=args.topology,
        row_graph=args.row_graph,
        column_graph=args.column_graph,
    )

    ops = load_static_ops(args.static_ops)
    system, data_wires = load_problem_system_and_wires(ops)
    local_ry_support = resolve_local_ry_support(ops, args.local_ry_support)
    scaffold_edges = resolve_scaffold_edges(ops, len(data_wires))

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

    n_agents = int(system.n)
    n_qubits = len(data_wires)
    spectrum = getattr(ops, "SPECTRUM_INFO", getattr(system, "metadata", {}).get("spectrum"))
    zero_block_mask = compute_zero_block_mask(system)
    zero_block_cells = [(row_id, col_id) for row_id in range(n_agents) for col_id in range(n_agents) if zero_block_mask[row_id][col_id]]
    lr_by_group = {
        "alpha": float(args.lr if args.lr_alpha is None else args.lr_alpha),
        "beta": float(args.lr if args.lr_beta is None else args.lr_beta),
        "sigma": float(args.lr if args.lr_sigma is None else args.lr_sigma),
        "lambda": float(args.lr if args.lr_lambda is None else args.lr_lambda),
    }

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
    logger.info(f"zero blocks: {zero_block_cells}")
    logger.info(
        "learning rates by group: "
        f"alpha={lr_by_group['alpha']:.8f}, "
        f"beta={lr_by_group['beta']:.8f}, "
        f"sigma={lr_by_group['sigma']:.8f}, "
        f"lambda={lr_by_group['lambda']:.8f}"
    )
    logger.info(f"relay_only_zero_blocks: {bool(args.relay_only_zero_blocks)}")
    logger.info(
        "freeze window: "
        f"groups={freeze_groups if freeze_groups else 'none'}, "
        f"from={int(args.freeze_from_epoch)}, "
        f"until={int(args.freeze_until_epoch)}"
    )
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

    mem_info = estimate_loss_memory_usage(system, row_topology, n_qubits)
    logger.info(
        f"Per Hadamard-test statevector ({mem_info['n_wires_per_hadamard_test']} wires): {mem_info['statevector_human']}"
    )
    logger.info(f"Conservative peak per Hadamard test: {mem_info['peak_human_conservative']}")
    logger.info(f"QNode evaluations per compute_loss call: {mem_info['total_qnode_evals_per_loss']}")
    logger.info(f"Sequential statevector traffic per compute_loss call: {mem_info['sequential_human_per_loss_eval']}")

    run_config = {
        "static_ops": args.static_ops,
        "out": str(paths.run_dir),
        "benchmark": None if benchmark is None else benchmark.benchmark_id,
        "row_graph": row_graph,
        "column_graph": column_graph,
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "lr": float(args.lr),
        "lr_alpha": float(lr_by_group["alpha"]),
        "lr_beta": float(lr_by_group["beta"]),
        "lr_sigma": float(lr_by_group["sigma"]),
        "lr_lambda": float(lr_by_group["lambda"]),
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
        "row_topology": row_topology,
        "column_topology": column_topology,
        "row_metropolis": row_metropolis.tolist(),
        "column_metropolis": column_metropolis.tolist(),
        "row_laplacian": row_laplacian.tolist(),
        "column_laplacian": column_laplacian.tolist(),
        "first_agent_cost_formula": first_agent_cost_formula(row_graph),
        "zero_block_mask": zero_block_mask,
        "relay_only_zero_blocks": bool(args.relay_only_zero_blocks),
        "freeze_groups": list(freeze_groups),
        "freeze_from_epoch": int(args.freeze_from_epoch),
        "freeze_until_epoch": int(args.freeze_until_epoch),
    }
    _write_run_config(paths.run_dir / "config_used.json", run_config)

    row_b_totals = [np.asarray(system.get_b_vectors(row_id)[0]) for row_id in range(n_agents)]
    global_b = np.concatenate(row_b_totals, axis=0)
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

    logger.info("Starting prebuild_local_evals() for distributed Hadamard-test bundles.")
    t_prebuild = time.time()
    ib.prebuild_local_evals(
        system,
        row_topology,
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

        logger.info("Optimization backend: qjit(catalyst.grad) + qjit(loss).")
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

    lr_schedule_by_group = {
        key: optax.exponential_decay(
            init_value=value,
            transition_steps=1,
            decay_rate=args.decay,
            staircase=False,
        )
        for key, value in lr_by_group.items()
    }
    optimizer_labels = build_optimizer_labels(
        n_agents,
        relay_only_zero_blocks=bool(args.relay_only_zero_blocks),
        zero_block_mask=zero_block_mask,
    )
    opt_adam = optax.multi_transform(
        {
            "alpha": optax.adam(lr_schedule_by_group["alpha"]),
            "beta": optax.adam(lr_schedule_by_group["beta"]),
            "sigma": optax.adam(lr_schedule_by_group["sigma"]),
            "lambda": optax.adam(lr_schedule_by_group["lambda"]),
            "relay": optax.set_to_zero(),
        },
        optimizer_labels,
    )

    loss_history = [float(current_loss)]
    metric_epochs = []
    residual_history = []
    l2_history = []
    consensus_history = []
    grad_norm_total_history = []
    grad_norm_alpha_history = []
    grad_norm_beta_history = []
    grad_norm_sigma_history = []
    grad_norm_lambda_history = []
    tracker_norm_total_history = []
    tracker_norm_alpha_history = []
    tracker_norm_beta_history = []
    tracker_norm_sigma_history = []
    tracker_norm_lambda_history = []
    update_norm_total_history = []
    update_norm_alpha_history = []
    update_norm_beta_history = []
    update_norm_sigma_history = []
    update_norm_lambda_history = []
    param_change_norm_total_history = []
    param_change_norm_alpha_history = []
    param_change_norm_beta_history = []
    param_change_norm_sigma_history = []
    param_change_norm_lambda_history = []
    grad_norm_alpha_by_agent_history = []
    grad_norm_beta_by_agent_history = []
    grad_norm_sigma_by_agent_history = []
    grad_norm_lambda_by_agent_history = []
    tracker_norm_alpha_by_agent_history = []
    tracker_norm_beta_by_agent_history = []
    tracker_norm_sigma_by_agent_history = []
    tracker_norm_lambda_by_agent_history = []
    update_norm_alpha_by_agent_history = []
    update_norm_beta_by_agent_history = []
    update_norm_sigma_by_agent_history = []
    update_norm_lambda_by_agent_history = []
    param_change_norm_alpha_by_agent_history = []
    param_change_norm_beta_by_agent_history = []
    param_change_norm_sigma_by_agent_history = []
    param_change_norm_lambda_by_agent_history = []
    final_global_cost = float(current_loss)

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
    init_grad_summary = compute_gradient_norm_summary(grad_grid_init)
    logger.info(
        "[Init] Gradient norms = "
        f"total {init_grad_summary['grad_norm_total']:.5e} | "
        f"alpha {init_grad_summary['grad_norm_alpha']:.5e} | "
        f"beta {init_grad_summary['grad_norm_beta']:.5e} | "
        f"sigma {init_grad_summary['grad_norm_sigma']:.5e} | "
        f"lambda {init_grad_summary['grad_norm_lambda']:.5e}"
    )
    tracker_grid = init_tracker_from_grad(grad_grid_init)
    prev_grad_grid = copy.deepcopy(grad_grid_init)
    init_tracker_summary = compute_group_norm_summary(tracker_grid, prefix="tracker")
    logger.info(
        "[Init] Tracker norms = "
        f"total {init_tracker_summary['tracker_norm_total']:.5e} | "
        f"alpha {init_tracker_summary['tracker_norm_alpha']:.5e} | "
        f"beta {init_tracker_summary['tracker_norm_beta']:.5e} | "
        f"sigma {init_tracker_summary['tracker_norm_sigma']:.5e} | "
        f"lambda {init_tracker_summary['tracker_norm_lambda']:.5e}"
    )

    all_keys = ["alpha", "beta", "sigma", "lambda"]
    opt_adam_state = opt_adam.init(to_jax_flat(flatten_params(global_params, keys=all_keys)))

    t0 = time.time()
    last_log_time = t0
    last_metrics = None

    for epoch in range(1, args.epochs + 1):
        epoch_start_params = copy.deepcopy(global_params)
        flat_params_epoch_start = to_jax_flat(flatten_params(epoch_start_params, keys=all_keys))
        freeze_active = bool(
            freeze_groups
            and args.freeze_from_epoch <= epoch
            and (args.freeze_until_epoch <= 0 or epoch <= args.freeze_until_epoch)
        )

        global_params = consensus_mix_metropolis_jax(global_params, W_np=column_metropolis)
        if freeze_active:
            restore_frozen_groups(global_params, epoch_start_params, freeze_groups)

        tracker_summary = compute_group_norm_summary(tracker_grid, prefix="tracker")
        flat_tracker = to_jax_flat(flatten_params(tracker_grid, keys=all_keys))
        if freeze_active:
            flat_tracker = mask_flat_group_entries(flat_tracker, freeze_groups)
        flat_params = to_jax_flat(flatten_params(global_params, keys=all_keys))
        adam_updates, opt_adam_state = opt_adam.update(flat_tracker, opt_adam_state, params=flat_params)
        update_summary = compute_group_norm_summary(
            rebuild_global_params(adam_updates, system.n, global_params["b_norm"]),
            prefix="update",
        )
        new_flat_params = optax.apply_updates(flat_params, adam_updates)
        param_change_summary = compute_group_norm_summary(
            rebuild_global_params(
                subtract_flat_lists(new_flat_params, flat_params_epoch_start),
                system.n,
                global_params["b_norm"],
            ),
            prefix="param_change",
        )
        update_global_from_flat(global_params, new_flat_params, keys=all_keys)

        flat_params_all = to_jax_flat(flatten_params(global_params, keys=None))
        grads_flat = compute_grad(flat_params_all)
        current_cost = compute_loss(flat_params_all)
        final_global_cost = float(current_cost)

        current_grad_grid = rebuild_global_params(grads_flat, system.n, global_params["b_norm"])
        grad_summary = compute_gradient_norm_summary(current_grad_grid)
        tracker_grid = update_gradient_tracker_metropolis_jax(
            current_tracker=tracker_grid,
            current_grads=current_grad_grid,
            prev_grads=prev_grad_grid,
            W_np=column_metropolis,
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
            grad_norm_total_history.append(float(grad_summary["grad_norm_total"]))
            grad_norm_alpha_history.append(float(grad_summary["grad_norm_alpha"]))
            grad_norm_beta_history.append(float(grad_summary["grad_norm_beta"]))
            grad_norm_sigma_history.append(float(grad_summary["grad_norm_sigma"]))
            grad_norm_lambda_history.append(float(grad_summary["grad_norm_lambda"]))
            tracker_norm_total_history.append(float(tracker_summary["tracker_norm_total"]))
            tracker_norm_alpha_history.append(float(tracker_summary["tracker_norm_alpha"]))
            tracker_norm_beta_history.append(float(tracker_summary["tracker_norm_beta"]))
            tracker_norm_sigma_history.append(float(tracker_summary["tracker_norm_sigma"]))
            tracker_norm_lambda_history.append(float(tracker_summary["tracker_norm_lambda"]))
            update_norm_total_history.append(float(update_summary["update_norm_total"]))
            update_norm_alpha_history.append(float(update_summary["update_norm_alpha"]))
            update_norm_beta_history.append(float(update_summary["update_norm_beta"]))
            update_norm_sigma_history.append(float(update_summary["update_norm_sigma"]))
            update_norm_lambda_history.append(float(update_summary["update_norm_lambda"]))
            param_change_norm_total_history.append(float(param_change_summary["param_change_norm_total"]))
            param_change_norm_alpha_history.append(float(param_change_summary["param_change_norm_alpha"]))
            param_change_norm_beta_history.append(float(param_change_summary["param_change_norm_beta"]))
            param_change_norm_sigma_history.append(float(param_change_summary["param_change_norm_sigma"]))
            param_change_norm_lambda_history.append(float(param_change_summary["param_change_norm_lambda"]))
            grad_norm_alpha_by_agent_history.append(np.asarray(grad_summary["grad_norm_alpha_by_agent"], dtype=float))
            grad_norm_beta_by_agent_history.append(np.asarray(grad_summary["grad_norm_beta_by_agent"], dtype=float))
            grad_norm_sigma_by_agent_history.append(np.asarray(grad_summary["grad_norm_sigma_by_agent"], dtype=float))
            grad_norm_lambda_by_agent_history.append(np.asarray(grad_summary["grad_norm_lambda_by_agent"], dtype=float))
            tracker_norm_alpha_by_agent_history.append(np.asarray(tracker_summary["tracker_norm_alpha_by_agent"], dtype=float))
            tracker_norm_beta_by_agent_history.append(np.asarray(tracker_summary["tracker_norm_beta_by_agent"], dtype=float))
            tracker_norm_sigma_by_agent_history.append(np.asarray(tracker_summary["tracker_norm_sigma_by_agent"], dtype=float))
            tracker_norm_lambda_by_agent_history.append(np.asarray(tracker_summary["tracker_norm_lambda_by_agent"], dtype=float))
            update_norm_alpha_by_agent_history.append(np.asarray(update_summary["update_norm_alpha_by_agent"], dtype=float))
            update_norm_beta_by_agent_history.append(np.asarray(update_summary["update_norm_beta_by_agent"], dtype=float))
            update_norm_sigma_by_agent_history.append(np.asarray(update_summary["update_norm_sigma_by_agent"], dtype=float))
            update_norm_lambda_by_agent_history.append(np.asarray(update_summary["update_norm_lambda_by_agent"], dtype=float))
            param_change_norm_alpha_by_agent_history.append(np.asarray(param_change_summary["param_change_norm_alpha_by_agent"], dtype=float))
            param_change_norm_beta_by_agent_history.append(np.asarray(param_change_summary["param_change_norm_beta_by_agent"], dtype=float))
            param_change_norm_sigma_by_agent_history.append(np.asarray(param_change_summary["param_change_norm_sigma_by_agent"], dtype=float))
            param_change_norm_lambda_by_agent_history.append(np.asarray(param_change_summary["param_change_norm_lambda_by_agent"], dtype=float))
            last_metrics = metric_bundle
            log_interval_s = now - last_log_time
            wall_s = now - t0
            timestamp_utc = datetime.now(timezone.utc).isoformat()
            timestamp_local = datetime.now().astimezone().isoformat()

            current_lr_by_group = {
                key: float(schedule(epoch - 1))
                for key, schedule in lr_schedule_by_group.items()
            }
            metrics.write(
                {
                    "epoch": epoch,
                    "benchmark": None if benchmark is None else benchmark.benchmark_id,
                    "row_graph": row_graph,
                    "column_graph": column_graph,
                    "global_cost": float(current_cost),
                    "loss": float(current_cost),
                    "residual_norm": float(metric_bundle["residual_norm"]),
                    "l2_error": float(metric_bundle["l2_error"]),
                    "consensus_error": float(metric_bundle["consensus_error"]),
                    "grad_norm_total": float(grad_summary["grad_norm_total"]),
                    "grad_norm_alpha": float(grad_summary["grad_norm_alpha"]),
                    "grad_norm_beta": float(grad_summary["grad_norm_beta"]),
                    "grad_norm_sigma": float(grad_summary["grad_norm_sigma"]),
                    "grad_norm_lambda": float(grad_summary["grad_norm_lambda"]),
                    "grad_norm_alpha_by_agent": grad_summary["grad_norm_alpha_by_agent"],
                    "grad_norm_beta_by_agent": grad_summary["grad_norm_beta_by_agent"],
                    "grad_norm_sigma_by_agent": grad_summary["grad_norm_sigma_by_agent"],
                    "grad_norm_lambda_by_agent": grad_summary["grad_norm_lambda_by_agent"],
                    "grad_max_abs": float(grad_summary["grad_max_abs"]),
                    "tracker_norm_total": float(tracker_summary["tracker_norm_total"]),
                    "tracker_norm_alpha": float(tracker_summary["tracker_norm_alpha"]),
                    "tracker_norm_beta": float(tracker_summary["tracker_norm_beta"]),
                    "tracker_norm_sigma": float(tracker_summary["tracker_norm_sigma"]),
                    "tracker_norm_lambda": float(tracker_summary["tracker_norm_lambda"]),
                    "tracker_norm_alpha_by_agent": tracker_summary["tracker_norm_alpha_by_agent"],
                    "tracker_norm_beta_by_agent": tracker_summary["tracker_norm_beta_by_agent"],
                    "tracker_norm_sigma_by_agent": tracker_summary["tracker_norm_sigma_by_agent"],
                    "tracker_norm_lambda_by_agent": tracker_summary["tracker_norm_lambda_by_agent"],
                    "update_norm_total": float(update_summary["update_norm_total"]),
                    "update_norm_alpha": float(update_summary["update_norm_alpha"]),
                    "update_norm_beta": float(update_summary["update_norm_beta"]),
                    "update_norm_sigma": float(update_summary["update_norm_sigma"]),
                    "update_norm_lambda": float(update_summary["update_norm_lambda"]),
                    "update_norm_alpha_by_agent": update_summary["update_norm_alpha_by_agent"],
                    "update_norm_beta_by_agent": update_summary["update_norm_beta_by_agent"],
                    "update_norm_sigma_by_agent": update_summary["update_norm_sigma_by_agent"],
                    "update_norm_lambda_by_agent": update_summary["update_norm_lambda_by_agent"],
                    "param_change_norm_total": float(param_change_summary["param_change_norm_total"]),
                    "param_change_norm_alpha": float(param_change_summary["param_change_norm_alpha"]),
                    "param_change_norm_beta": float(param_change_summary["param_change_norm_beta"]),
                    "param_change_norm_sigma": float(param_change_summary["param_change_norm_sigma"]),
                    "param_change_norm_lambda": float(param_change_summary["param_change_norm_lambda"]),
                    "param_change_norm_alpha_by_agent": param_change_summary["param_change_norm_alpha_by_agent"],
                    "param_change_norm_beta_by_agent": param_change_summary["param_change_norm_beta_by_agent"],
                    "param_change_norm_sigma_by_agent": param_change_summary["param_change_norm_sigma_by_agent"],
                    "param_change_norm_lambda_by_agent": param_change_summary["param_change_norm_lambda_by_agent"],
                    "lr": float(current_lr_by_group["alpha"]),
                    "lr_alpha": float(current_lr_by_group["alpha"]),
                    "lr_beta": float(current_lr_by_group["beta"]),
                    "lr_sigma": float(current_lr_by_group["sigma"]),
                    "lr_lambda": float(current_lr_by_group["lambda"]),
                    "relay_only_zero_blocks": bool(args.relay_only_zero_blocks),
                    "freeze_groups": list(freeze_groups),
                    "freeze_from_epoch": int(args.freeze_from_epoch),
                    "freeze_until_epoch": int(args.freeze_until_epoch),
                    "freeze_active": freeze_active,
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
                f"Grad = {float(grad_summary['grad_norm_total']):.5e} | "
                f"Update = {float(update_summary['update_norm_total']):.5e} | "
                f"dTheta = {float(param_change_summary['param_change_norm_total']):.5e} | "
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
        f"{benchmark.benchmark_id if benchmark else 'custom'} | row={row_graph} | col={column_graph} | "
        f"lr0={args.lr:g} | {args.ansatz}"
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

        plt.figure()
        plt.plot(metric_epochs, grad_norm_total_history)
        plt.xlabel("Epoch")
        plt.ylabel("Gradient Norm")
        plt.yscale("log")
        plt.grid(True)
        plt.title(title)
        plt.savefig(paths.run_dir / "grad_norm_total.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(metric_epochs, grad_norm_alpha_history, label="alpha")
        plt.plot(metric_epochs, grad_norm_beta_history, label="beta")
        plt.plot(metric_epochs, grad_norm_sigma_history, label="sigma")
        plt.plot(metric_epochs, grad_norm_lambda_history, label="lambda")
        plt.xlabel("Epoch")
        plt.ylabel("Gradient Norm by Group")
        plt.yscale("log")
        plt.grid(True)
        plt.legend()
        plt.title(title)
        plt.savefig(paths.run_dir / "grad_norm_components.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(metric_epochs, tracker_norm_total_history)
        plt.xlabel("Epoch")
        plt.ylabel("Tracker Norm")
        plt.yscale("log")
        plt.grid(True)
        plt.title(title)
        plt.savefig(paths.run_dir / "tracker_norm_total.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(metric_epochs, tracker_norm_alpha_history, label="alpha")
        plt.plot(metric_epochs, tracker_norm_beta_history, label="beta")
        plt.plot(metric_epochs, tracker_norm_sigma_history, label="sigma")
        plt.plot(metric_epochs, tracker_norm_lambda_history, label="lambda")
        plt.xlabel("Epoch")
        plt.ylabel("Tracker Norm by Group")
        plt.yscale("log")
        plt.grid(True)
        plt.legend()
        plt.title(title)
        plt.savefig(paths.run_dir / "tracker_norm_components.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(metric_epochs, update_norm_total_history)
        plt.xlabel("Epoch")
        plt.ylabel("Optimizer Update Norm")
        plt.yscale("log")
        plt.grid(True)
        plt.title(title)
        plt.savefig(paths.run_dir / "update_norm_total.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(metric_epochs, update_norm_alpha_history, label="alpha")
        plt.plot(metric_epochs, update_norm_beta_history, label="beta")
        plt.plot(metric_epochs, update_norm_sigma_history, label="sigma")
        plt.plot(metric_epochs, update_norm_lambda_history, label="lambda")
        plt.xlabel("Epoch")
        plt.ylabel("Optimizer Update Norm by Group")
        plt.yscale("log")
        plt.grid(True)
        plt.legend()
        plt.title(title)
        plt.savefig(paths.run_dir / "update_norm_components.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(metric_epochs, param_change_norm_total_history)
        plt.xlabel("Epoch")
        plt.ylabel("Parameter Change Norm")
        plt.yscale("log")
        plt.grid(True)
        plt.title(title)
        plt.savefig(paths.run_dir / "param_change_norm_total.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(metric_epochs, param_change_norm_alpha_history, label="alpha")
        plt.plot(metric_epochs, param_change_norm_beta_history, label="beta")
        plt.plot(metric_epochs, param_change_norm_sigma_history, label="sigma")
        plt.plot(metric_epochs, param_change_norm_lambda_history, label="lambda")
        plt.xlabel("Epoch")
        plt.ylabel("Parameter Change Norm by Group")
        plt.yscale("log")
        plt.grid(True)
        plt.legend()
        plt.title(title)
        plt.savefig(paths.run_dir / "param_change_norm_components.png", dpi=200, bbox_inches="tight")
        plt.close()

    np.savez(
        paths.artifacts_npz,
        loss=np.array(loss_history),
        metric_epochs=np.array(metric_epochs),
        residual_norm=np.array(residual_history),
        l2_error=np.array(l2_history),
        consensus_error=np.array(consensus_history),
        grad_norm_total=np.array(grad_norm_total_history),
        grad_norm_alpha=np.array(grad_norm_alpha_history),
        grad_norm_beta=np.array(grad_norm_beta_history),
        grad_norm_sigma=np.array(grad_norm_sigma_history),
        grad_norm_lambda=np.array(grad_norm_lambda_history),
        tracker_norm_total=np.array(tracker_norm_total_history),
        tracker_norm_alpha=np.array(tracker_norm_alpha_history),
        tracker_norm_beta=np.array(tracker_norm_beta_history),
        tracker_norm_sigma=np.array(tracker_norm_sigma_history),
        tracker_norm_lambda=np.array(tracker_norm_lambda_history),
        update_norm_total=np.array(update_norm_total_history),
        update_norm_alpha=np.array(update_norm_alpha_history),
        update_norm_beta=np.array(update_norm_beta_history),
        update_norm_sigma=np.array(update_norm_sigma_history),
        update_norm_lambda=np.array(update_norm_lambda_history),
        param_change_norm_total=np.array(param_change_norm_total_history),
        param_change_norm_alpha=np.array(param_change_norm_alpha_history),
        param_change_norm_beta=np.array(param_change_norm_beta_history),
        param_change_norm_sigma=np.array(param_change_norm_sigma_history),
        param_change_norm_lambda=np.array(param_change_norm_lambda_history),
        grad_norm_alpha_by_agent=np.array(grad_norm_alpha_by_agent_history),
        grad_norm_beta_by_agent=np.array(grad_norm_beta_by_agent_history),
        grad_norm_sigma_by_agent=np.array(grad_norm_sigma_by_agent_history),
        grad_norm_lambda_by_agent=np.array(grad_norm_lambda_by_agent_history),
        tracker_norm_alpha_by_agent=np.array(tracker_norm_alpha_by_agent_history),
        tracker_norm_beta_by_agent=np.array(tracker_norm_beta_by_agent_history),
        tracker_norm_sigma_by_agent=np.array(tracker_norm_sigma_by_agent_history),
        tracker_norm_lambda_by_agent=np.array(tracker_norm_lambda_by_agent_history),
        update_norm_alpha_by_agent=np.array(update_norm_alpha_by_agent_history),
        update_norm_beta_by_agent=np.array(update_norm_beta_by_agent_history),
        update_norm_sigma_by_agent=np.array(update_norm_sigma_by_agent_history),
        update_norm_lambda_by_agent=np.array(update_norm_lambda_by_agent_history),
        param_change_norm_alpha_by_agent=np.array(param_change_norm_alpha_by_agent_history),
        param_change_norm_beta_by_agent=np.array(param_change_norm_beta_by_agent_history),
        param_change_norm_sigma_by_agent=np.array(param_change_norm_sigma_by_agent_history),
        param_change_norm_lambda_by_agent=np.array(param_change_norm_lambda_by_agent_history),
    )

    final_params_path = paths.run_dir / "final_params.json"
    _write_final_params_json(final_params_path, global_params)
    logger.info(f"Final parameters written to: {final_params_path}")

    analysis_path = paths.run_dir / "analysis.txt"
    write_analysis_report(
        analysis_path,
        args=args,
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
        mem_info=mem_info,
        global_params=global_params,
        row_b_totals=row_b_totals,
        true_solution=true_solution,
        final_metrics={
            **last_metrics,
            "global_cost": final_global_cost,
            "scaffold_edges": scaffold_edges,
        },
    )
    logger.info(f"Analysis report written to: {analysis_path}")
    logger.info(f"Finished. Outputs written to: {paths.run_dir}")


if __name__ == "__main__":
    main()
