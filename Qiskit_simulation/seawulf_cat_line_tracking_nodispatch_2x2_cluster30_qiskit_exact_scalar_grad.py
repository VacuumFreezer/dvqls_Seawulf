#!/usr/bin/env python3
"""Qiskit/Aer MPS optimizer with SPSA on alpha/beta and exact sigma/lambda gradients."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from common.reporting import JsonlWriter, make_run_dir, setup_logger
from Qiskit_simulation.builder_cat_nodispatch_qiskit import DistributedCostBuilderQiskit
from Qiskit_simulation.seawulf_cat_line_tracking_nodispatch_2x2_cluster30_qiskit import (
    AdamOptimizer,
    TOPOLOGY_LINE_2,
    TRAINABLE_KEYS,
    build_metropolis_matrix,
    clone_params,
    consensus_mix,
    flatten_params,
    init_tracker_from_grad,
    initialize_global_params,
    load_problem_system,
    load_static_ops,
    maybe_plot,
    structured_from_flat,
    unflatten_params,
    update_gradient_tracker,
    write_final_params,
)


ANGLE_KEYS = ("alpha", "beta")


def estimate_spsa_gradient_subset(
    builder: DistributedCostBuilderQiskit,
    params_template: Mapping[str, np.ndarray],
    flat_params: np.ndarray,
    *,
    keys: Iterable[str],
    rng: np.random.Generator,
    c: float,
) -> tuple[np.ndarray, float, float]:
    keys = tuple(keys)
    delta = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=flat_params.shape).astype(np.float32)
    theta_plus = flat_params + np.float32(c) * delta
    theta_minus = flat_params - np.float32(c) * delta
    params_plus = unflatten_params(theta_plus, params_template, keys=keys)
    params_minus = unflatten_params(theta_minus, params_template, keys=keys)
    loss_plus = builder.evaluate_total_loss(params_plus)
    loss_minus = builder.evaluate_total_loss(params_minus)
    grad = ((loss_plus - loss_minus) / (2.0 * float(c))) * delta
    return grad.astype(np.float32), float(loss_plus), float(loss_minus)


def compose_mixed_gradients(
    reference: Mapping[str, np.ndarray],
    angle_grad_struct: Mapping[str, np.ndarray],
    sigma_grad: np.ndarray,
    lambda_grad: np.ndarray,
) -> dict[str, np.ndarray]:
    out = clone_params(reference)
    out["alpha"] = np.asarray(angle_grad_struct["alpha"], dtype=np.float32)
    out["beta"] = np.asarray(angle_grad_struct["beta"], dtype=np.float32)
    out["sigma"] = np.asarray(sigma_grad, dtype=np.float32)
    out["lambda"] = np.asarray(lambda_grad, dtype=np.float32)
    out["b_norm"] = np.asarray(reference["b_norm"], dtype=np.float32)
    return out


def write_analysis(path: Path, args, system, builder: DistributedCostBuilderQiskit, w_mat: np.ndarray, params=None, diagnostics=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Qiskit/Aer MPS distributed 2x2 cluster30 run\n")
        f.write(f"static_ops module: {args.static_ops}\n")
        f.write(f"system name: {system.name}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"log_every: {args.log_every}\n")
        f.write(f"layers: {args.layers}\n")
        f.write(f"repeat_cz_each_layer: {bool(args.repeat_cz_each_layer)}\n")
        f.write(f"ansatz_kind: {builder.ansatz_kind}\n")
        f.write(f"scaffold_edges: {builder.scaffold_edges}\n")
        f.write("gradient_method_alpha_beta: spsa\n")
        f.write("gradient_method_sigma_lambda: exact_from_terms\n")
        f.write(f"learning rate: {args.lr}\n")
        f.write(f"spsa_c: {args.spsa_c}\n")
        f.write(f"precision: {builder.precision}\n")
        f.write(f"max_bond_dim: {builder.max_bond_dim}\n")
        f.write(f"num_threads: {builder.num_threads}\n")
        f.write(f"metropolis matrix:\n{w_mat}\n")
        f.write(f"block term counts: {system.metadata.get('block_term_counts')}\n")
        f.write(f"spectrum: {system.metadata.get('spectrum')}\n")
        f.write(
            "residual is computed from overlap formulas on the averaged column states x0=(v00+v10)/2 and x1=(v01+v11)/2,\n"
        )
        f.write(
            "without building dense 2^30 vectors. variance is the mean of the two column-wise pair variances.\n"
        )
        if params is not None:
            from Qiskit_simulation.seawulf_cat_line_tracking_nodispatch_2x2_cluster30_qiskit import _param_summary

            f.write(f"final parameter summary: {_param_summary(params)}\n")
        if diagnostics is not None:
            f.write(f"final diagnostics: {diagnostics}\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static_ops", default="Qiskit_simulation.static_ops_2x2_cluster30_qiskit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--repeat_cz_each_layer", action="store_true")
    ap.add_argument("--spsa_c", type=float, default=0.05)
    ap.add_argument("--max_bond_dim", type=int, default=8)
    ap.add_argument("--num_threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    return ap.parse_args()


def main():
    args = parse_args()
    paths = make_run_dir(args.out)
    logger = setup_logger(paths.report_txt)
    metrics = JsonlWriter(paths.metrics_jsonl)

    ops = load_static_ops(args.static_ops)
    system = load_problem_system(ops)
    n_qubits = int(system.n_data_qubits)
    if args.layers < 1:
        raise ValueError("--layers must be at least 1.")
    rng = np.random.default_rng(args.seed)

    builder = DistributedCostBuilderQiskit(
        system,
        TOPOLOGY_LINE_2,
        ansatz_layers=args.layers,
        repeat_cz_each_layer=args.repeat_cz_each_layer,
        max_bond_dim=args.max_bond_dim,
        num_threads=args.num_threads,
        precision="single",
        seed=args.seed,
    )
    w_mat = build_metropolis_matrix(TOPOLOGY_LINE_2, n=system.n)

    params = initialize_global_params(system, n_qubits=n_qubits, layers=args.layers, seed=args.seed)
    logger.info(f"System variant: {system.name}")
    logger.info(f"static_ops module: {args.static_ops}")
    logger.info(f"local data qubits per agent: {n_qubits}")
    logger.info(
        f"epochs={args.epochs}, log_every={args.log_every}, lr={args.lr}, spsa_c={args.spsa_c}, "
        f"layers={args.layers}, repeat_cz_each_layer={bool(args.repeat_cz_each_layer)}"
    )
    logger.info(f"ansatz={builder.ansatz_kind}, scaffold_edges={builder.scaffold_edges}")
    logger.info("gradient methods: alpha/beta=SPSA, sigma/lambda=exact_from_terms")
    logger.info(f"Aer method=matrix_product_state, precision=single, max_bond_dim={args.max_bond_dim}")
    logger.info("Metropolis weight matrix Wm =\n" + str(w_mat))
    spectrum = system.metadata.get("spectrum")
    if spectrum:
        logger.info(
            "Analytic spectrum: "
            f"lambda_min={spectrum['lambda_min']:.8f}, "
            f"lambda_max={spectrum['lambda_max']:.8f}, "
            f"cond(A)={spectrum['condition_number']:.8f}"
        )

    write_analysis(paths.run_dir / "analysis.txt", args, system, builder, w_mat)

    current_loss, sigma_grad_exact, lambda_grad_exact = builder.evaluate_total_loss_and_exact_sigma_lambda_grads(params)
    logger.info(f"[Init] Initial Loss = {current_loss:.6e}")
    init_diag = builder.compute_diagnostics(params)
    logger.info(
        f"[Init] Residual = {init_diag['residual_norm']:.6e} | Variance(mean) = {init_diag['variance_mean']:.6e}"
    )
    metrics.write(
        {
            "epoch": 0,
            "loss": float(current_loss),
            "residual_norm": float(init_diag["residual_norm"]),
            "variance_mean": float(init_diag["variance_mean"]),
            "variance_col0": float(init_diag["variance_col0"]),
            "variance_col1": float(init_diag["variance_col1"]),
            "wall_s": 0.0,
        }
    )

    flat_angles = flatten_params(params, keys=ANGLE_KEYS)
    grad_angles_flat_init, _, _ = estimate_spsa_gradient_subset(
        builder,
        params,
        flat_angles,
        keys=ANGLE_KEYS,
        rng=rng,
        c=args.spsa_c,
    )
    grad_angles_struct_init = unflatten_params(grad_angles_flat_init, params, keys=ANGLE_KEYS)
    grad_grid = compose_mixed_gradients(params, grad_angles_struct_init, sigma_grad_exact, lambda_grad_exact)
    tracker_grid = init_tracker_from_grad(grad_grid)
    prev_grad_grid = clone_params(grad_grid)

    adam = AdamOptimizer(lr=args.lr)
    adam_state = adam.init(flatten_params(params, keys=TRAINABLE_KEYS).size)

    log_epochs = [0]
    loss_history = [float(current_loss)]
    residual_history = [float(init_diag["residual_norm"])]
    variance_history = [float(init_diag["variance_mean"])]

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        params = consensus_mix(params, w_mat)

        tracker_flat = flatten_params(tracker_grid, keys=TRAINABLE_KEYS)
        params_flat = flatten_params(params, keys=TRAINABLE_KEYS)
        new_flat, adam_state = adam.update(tracker_flat, params_flat, adam_state)
        params = structured_from_flat(new_flat, params)

        current_loss, sigma_grad_exact, lambda_grad_exact = builder.evaluate_total_loss_and_exact_sigma_lambda_grads(params)
        flat_angles = flatten_params(params, keys=ANGLE_KEYS)
        grad_angles_flat, loss_plus, loss_minus = estimate_spsa_gradient_subset(
            builder,
            params,
            flat_angles,
            keys=ANGLE_KEYS,
            rng=rng,
            c=args.spsa_c,
        )
        grad_angles_struct = unflatten_params(grad_angles_flat, params, keys=ANGLE_KEYS)
        grad_grid = compose_mixed_gradients(params, grad_angles_struct, sigma_grad_exact, lambda_grad_exact)
        tracker_grid = update_gradient_tracker(tracker_grid, grad_grid, prev_grad_grid, w_mat)
        prev_grad_grid = clone_params(grad_grid)

        if ep % args.log_every == 0:
            diag = builder.compute_diagnostics(params)
            wall_s = time.time() - t0
            metrics.write(
                {
                    "epoch": ep,
                    "loss": float(current_loss),
                    "residual_norm": float(diag["residual_norm"]),
                    "variance_mean": float(diag["variance_mean"]),
                    "variance_col0": float(diag["variance_col0"]),
                    "variance_col1": float(diag["variance_col1"]),
                    "loss_plus": float(loss_plus),
                    "loss_minus": float(loss_minus),
                    "wall_s": float(wall_s),
                }
            )
            logger.info(
                f"[Epoch {ep:05d}] Loss = {current_loss:.6e} | Residual = {diag['residual_norm']:.6e} | "
                f"Variance(mean) = {diag['variance_mean']:.6e}"
            )
            log_epochs.append(ep)
            loss_history.append(float(current_loss))
            residual_history.append(float(diag["residual_norm"]))
            variance_history.append(float(diag["variance_mean"]))

    metrics.close()
    maybe_plot(paths, loss_history, log_epochs, residual_history, variance_history)
    final_diag = builder.compute_diagnostics(params)
    final_params_path = paths.run_dir / "final_params.json"
    write_final_params(final_params_path, params)
    write_analysis(paths.run_dir / "analysis.txt", args, system, builder, w_mat, params=params, diagnostics=final_diag)
    np.savez(
        paths.artifacts_npz,
        epochs=np.asarray(log_epochs, dtype=np.int32),
        loss=np.asarray(loss_history, dtype=np.float64),
        residual_norm=np.asarray(residual_history, dtype=np.float64),
        variance_mean=np.asarray(variance_history, dtype=np.float64),
    )
    logger.info(f"Final parameters written to: {final_params_path}")
    logger.info("Finished. Outputs written to: " + str(paths.run_dir))


if __name__ == "__main__":
    main()
