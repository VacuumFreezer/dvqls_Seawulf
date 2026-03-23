from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import numpy as np
import optax
import pennylane as qml
from pennylane import numpy as pnp

from common.DIGing_jax import (
    build_metropolis_matrix,
    consensus_mix_metropolis_jax,
    init_tracker_from_grad,
    update_gradient_tracker_metropolis_jax,
)
from common.params_io import flatten_params, rebuild_global_params, update_global_from_flat
from common.reporting import JsonlWriter, make_run_dir, setup_logger
import objective.builder_cluster_nodispatch as ib

from Partition_comparison_qjit.Old_asymmetry_stabilizer import seawulf_partition_comparison_qjit as base


VARIANT_NO_TRACKING = "no_tracking_single_consensus"
VARIANT_SLOW_CONS_TRACK = "slow_consensus_tracking"


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static_ops", required=True, help="Module name or .py path for the partitioned static-ops file.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topology", type=str, default="line", choices=("line",))
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--decay", type=float, default=0.9999)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--ansatz", type=str, default=base.ANSATZ_BRICKWALL_RY_CZ, choices=base.VALID_ANSATZ_KINDS)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--repeat_cz_each_layer", action="store_true")
    ap.add_argument("--init_mode", type=str, default=base.INIT_MODE_UNIFORM_PM_PI, choices=base.VALID_INIT_MODES)
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
    return ap


def run_variant(mode: str, argv=None) -> None:
    if mode not in (VARIANT_NO_TRACKING, VARIANT_SLOW_CONS_TRACK):
        raise ValueError(f"Unsupported optimizer mode {mode!r}.")

    ap = _build_parser()
    args = ap.parse_args(argv)
    args.ansatz = base.normalize_ansatz_kind(args.ansatz)
    if args.layers < 1:
        raise ValueError("--layers must be at least 1.")

    ops = base.load_static_ops(args.static_ops)
    system, data_wires = base.load_problem_system_and_wires(ops)
    local_ry_support = base.resolve_local_ry_support(ops, args.local_ry_support)
    scaffold_edges = base.resolve_scaffold_edges(ops, len(data_wires))
    if args.ansatz in (base.ANSATZ_CLUSTER_RZ_LOCAL_RY, base.ANSATZ_CLUSTER_LOCAL_RY) and not local_ry_support:
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
    topology = base.build_line_topology(n_agents)
    spectrum = getattr(ops, "SPECTRUM_INFO", getattr(system, "metadata", {}).get("spectrum"))

    logger.info(f"System variant: {getattr(system, 'name', 'unknown')}")
    logger.info(f"static_ops: {args.static_ops}")
    logger.info(f"optimizer_variant: {mode}")
    logger.info(f"agents per row/column: {n_agents}")
    logger.info(f"local data qubits: {n_qubits}")
    logger.info(f"ansatz: {args.ansatz} ({base.describe_ansatz(args.ansatz)})")
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
    if not base.CATALYST_AVAILABLE:
        logger.info(
            "Catalyst import failed in this environment; falling back to plain JAX without qjit. "
            f"Reason: {type(base.CATALYST_IMPORT_ERROR).__name__}: {base.CATALYST_IMPORT_ERROR}"
        )

    mem_info = base.estimate_loss_memory_usage(system, topology, n_qubits)
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
        "optimizer_variant": mode,
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
    base._write_run_config(paths.run_dir / "config_used.json", run_config)

    row_b_totals = [np.asarray(system.get_b_vectors(row_id)[0]) for row_id in range(n_agents)]
    global_b = base.flatten_blocks(row_b_totals)
    true_solution = base.compute_true_solution(system, global_b, logger)
    logger.info(f"Global right-hand side shape: {global_b.shape}")
    logger.info(f"True solution shape: {true_solution.shape}")
    global_params, _ = base.initialize_cluster_params_jax(
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

    if base.CATALYST_AVAILABLE:
        @qml.qjit
        def compute_grad(args_flat):
            def qjit_total_loss_fn(args_inner):
                current_params = rebuild_global_params(args_inner, system.n, global_params["b_norm"])
                return ib.eval_total_loss(current_params)

            return base.catalyst.grad(qjit_total_loss_fn, method="auto")(args_flat)

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
            f"Reason: {base._summarize_exception(base.CATALYST_IMPORT_ERROR)}"
        )

    get_local_state = base.build_state_getter(
        n_qubits=n_qubits,
        ansatz_kind=args.ansatz,
        repeat_cz_each_layer=args.repeat_cz_each_layer,
        local_ry_support=local_ry_support,
        scaffold_edges=scaffold_edges,
    )

    flat_params_init = base.to_jax_flat(flatten_params(global_params, keys=None))
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

    loss_history = [float(current_loss)]
    metric_epochs = []
    residual_history = []
    l2_history = []
    consensus_history = []

    logger.info("Starting initial gradient evaluation.")
    t_init_grad = time.time()
    try:
        grads_flat_init = compute_grad(flat_params_init)
    except Exception as exc:
        if not base.CATALYST_AVAILABLE:
            raise
        logger.info(
            "Catalyst gradient compilation failed for this benchmark; "
            "keeping qjit(loss) and falling back to jax.jit(value_and_grad) for gradients only. "
            f"Reason: {base._summarize_exception(exc)}"
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
    opt_adam_state = opt_adam.init(base.to_jax_flat(flatten_params(global_params, keys=all_keys)))

    t0 = time.time()
    last_log_time = t0
    last_metrics = None

    for epoch in range(1, args.epochs + 1):
        if mode == VARIANT_NO_TRACKING:
            global_params = consensus_mix_metropolis_jax(global_params, W_np=Wm)

            flat_params_pre = base.to_jax_flat(flatten_params(global_params, keys=None))
            grads_flat = compute_grad(flat_params_pre)
            grad_grid = rebuild_global_params(grads_flat, system.n, global_params["b_norm"])

            flat_grads = base.to_jax_flat(flatten_params(grad_grid, keys=all_keys))
            flat_params = base.to_jax_flat(flatten_params(global_params, keys=all_keys))
            adam_updates, opt_adam_state = opt_adam.update(flat_grads, opt_adam_state, params=flat_params)
            new_flat_params = optax.apply_updates(flat_params, adam_updates)
            update_global_from_flat(global_params, new_flat_params, keys=all_keys)

            flat_params_all = base.to_jax_flat(flatten_params(global_params, keys=None))
            current_cost = compute_loss(flat_params_all)

        else:
            if (epoch % 2) == 0:
                global_params = consensus_mix_metropolis_jax(global_params, W_np=Wm)

            flat_tracker = base.to_jax_flat(flatten_params(tracker_grid, keys=all_keys))
            flat_params = base.to_jax_flat(flatten_params(global_params, keys=all_keys))
            adam_updates, opt_adam_state = opt_adam.update(flat_tracker, opt_adam_state, params=flat_params)
            new_flat_params = optax.apply_updates(flat_params, adam_updates)
            update_global_from_flat(global_params, new_flat_params, keys=all_keys)

            flat_params_all = base.to_jax_flat(flatten_params(global_params, keys=None))
            grads_flat = compute_grad(flat_params_all)
            current_cost = compute_loss(flat_params_all)
            current_grad_grid = rebuild_global_params(grads_flat, system.n, global_params["b_norm"])

            if (epoch % 2) == 0:
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
            metric_bundle = base.compute_metric_bundle(
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
        last_metrics = base.compute_metric_bundle(
            global_params,
            system,
            get_local_state,
            row_b_totals=row_b_totals,
            true_solution=true_solution,
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    title = f"{getattr(system, 'name', 'partition')} | {mode} | lr0={args.lr:g} | {args.ansatz}"

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
        plt.ylabel(r"$||Ax-b||$")
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

    final_metrics = {
        **last_metrics,
        "global_cost": float(loss_history[-1]),
        "scaffold_edges": scaffold_edges,
    }
    base._write_final_params_json(paths.run_dir / "final_params.json", global_params)
    logger.info(f"Final parameters written to: {paths.run_dir / 'final_params.json'}")

    analysis_path = paths.run_dir / "analysis.txt"
    base.write_analysis_report(
        analysis_path,
        args=args,
        ops_module_name=args.static_ops,
        system=system,
        data_wires=data_wires,
        mem_info=mem_info,
        global_params=global_params,
        row_b_totals=row_b_totals,
        true_solution=true_solution,
        final_metrics=final_metrics,
    )
    logger.info(f"Analysis report written to: {analysis_path}")
    logger.info(f"Finished. Outputs written to: {paths.run_dir}")
