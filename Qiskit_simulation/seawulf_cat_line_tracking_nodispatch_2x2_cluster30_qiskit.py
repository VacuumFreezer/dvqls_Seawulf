#!/usr/bin/env python3
"""Qiskit/Aer MPS distributed optimizer for the 2x2 cluster30 problem."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.reporting import JsonlWriter, make_run_dir, setup_logger
from Qiskit_simulation.builder_cat_nodispatch_qiskit import DistributedCostBuilderQiskit


TOPOLOGY_LINE_2 = {0: [1], 1: [0]}
TRAINABLE_KEYS = ("alpha", "beta", "sigma", "lambda")


@dataclass
class AdamState:
    m: np.ndarray
    v: np.ndarray
    step: int = 0


class AdamOptimizer:
    def __init__(self, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

    def init(self, size: int) -> AdamState:
        return AdamState(m=np.zeros(size, dtype=np.float32), v=np.zeros(size, dtype=np.float32), step=0)

    def update(self, grad: np.ndarray, params: np.ndarray, state: AdamState) -> tuple[np.ndarray, AdamState]:
        grad = np.asarray(grad, dtype=np.float32)
        params = np.asarray(params, dtype=np.float32)
        step = state.step + 1
        m = self.beta1 * state.m + (1.0 - self.beta1) * grad
        v = self.beta2 * state.v + (1.0 - self.beta2) * (grad * grad)
        mhat = m / (1.0 - self.beta1**step)
        vhat = v / (1.0 - self.beta2**step)
        new_params = params - self.lr * mhat / (np.sqrt(vhat) + self.eps)
        return new_params.astype(np.float32), AdamState(m=m, v=v, step=step)


def load_static_ops(module_name: str):
    return importlib.import_module(module_name)


def load_problem_system(ops_module):
    if hasattr(ops_module, "SYSTEMS") and "2x2" in getattr(ops_module, "SYSTEMS"):
        system = ops_module.SYSTEMS["2x2"]
    elif hasattr(ops_module, "SYSTEM"):
        system = ops_module.SYSTEM
    else:
        raise RuntimeError(f"{ops_module.__name__} does not expose SYSTEM or SYSTEMS['2x2']")
    if int(system.n) != 2:
        raise RuntimeError(f"This optimizer expects a 2-agent system, got {system.n}")
    return system


def build_metropolis_matrix(neighbor_map: Mapping[int, Sequence[int]], n: int) -> np.ndarray:
    adj = [set() for _ in range(n)]
    for i, nbrs in neighbor_map.items():
        for k in nbrs:
            if i == k:
                continue
            adj[int(i)].add(int(k))
            adj[int(k)].add(int(i))
    deg = [len(adj[i]) + 1 for i in range(n)]
    w = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        if not adj[i]:
            w[i, i] = 1.0
            continue
        acc = 0.0
        for k in adj[i]:
            val = 1.0 / (1.0 + max(deg[i], deg[k]))
            w[i, k] = val
            acc += val
        w[i, i] = 1.0 - acc
    return w.astype(np.float32)


def initialize_global_params(system, n_qubits: int, seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(-math.pi, math.pi, size=(system.n, system.n, n_qubits)).astype(np.float32)
    beta = rng.uniform(-math.pi, math.pi, size=(system.n, system.n, n_qubits)).astype(np.float32)
    sigma = rng.uniform(0.0, 2.0, size=(system.n, system.n)).astype(np.float32)
    lamb = rng.uniform(0.0, 2.0, size=(system.n, system.n)).astype(np.float32)
    b_norm = np.zeros((system.n, system.n), dtype=np.float32)
    for sys_id in range(system.n):
        b_norm[sys_id, :] = np.asarray(system.get_local_b_norms(sys_id), dtype=np.float32)
    return {
        "alpha": alpha,
        "beta": beta,
        "sigma": sigma,
        "lambda": lamb,
        "b_norm": b_norm,
    }


def clone_params(params: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {key: np.array(val, copy=True) for key, val in params.items()}


def flatten_params(params: Mapping[str, np.ndarray], keys: Iterable[str] = TRAINABLE_KEYS) -> np.ndarray:
    parts: List[np.ndarray] = []
    for i in range(params["alpha"].shape[0]):
        for j in range(params["alpha"].shape[1]):
            if "alpha" in keys:
                parts.append(np.asarray(params["alpha"][i, j], dtype=np.float32).reshape(-1))
            if "beta" in keys:
                parts.append(np.asarray(params["beta"][i, j], dtype=np.float32).reshape(-1))
            if "sigma" in keys:
                parts.append(np.asarray([params["sigma"][i, j]], dtype=np.float32))
            if "lambda" in keys:
                parts.append(np.asarray([params["lambda"][i, j]], dtype=np.float32))
    if not parts:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


def unflatten_params(flat: np.ndarray, template: Mapping[str, np.ndarray], keys: Iterable[str] = TRAINABLE_KEYS) -> Dict[str, np.ndarray]:
    keys = tuple(keys)
    flat = np.asarray(flat, dtype=np.float32).reshape(-1)
    out = clone_params(template)
    idx = 0
    nq = template["alpha"].shape[-1]
    for i in range(template["alpha"].shape[0]):
        for j in range(template["alpha"].shape[1]):
            if "alpha" in keys:
                out["alpha"][i, j] = flat[idx : idx + nq]
                idx += nq
            if "beta" in keys:
                out["beta"][i, j] = flat[idx : idx + nq]
                idx += nq
            if "sigma" in keys:
                out["sigma"][i, j] = flat[idx]
                idx += 1
            if "lambda" in keys:
                out["lambda"][i, j] = flat[idx]
                idx += 1
    if idx != len(flat):
        raise ValueError(f"Unused flat parameters: consumed {idx}, length {len(flat)}")
    return out


def structured_from_flat(flat: np.ndarray, reference: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return unflatten_params(flat, reference, keys=TRAINABLE_KEYS)


def consensus_mix(params: Mapping[str, np.ndarray], w_mat: np.ndarray) -> Dict[str, np.ndarray]:
    mixed = clone_params(params)
    for col in range(params["alpha"].shape[1]):
        for row in range(params["alpha"].shape[0]):
            weights = w_mat[row].reshape(-1, 1)
            mixed["alpha"][row, col] = np.sum(weights * params["alpha"][:, col, :], axis=0, dtype=np.float32)
            mixed["sigma"][row, col] = np.sum(w_mat[row] * params["sigma"][:, col], dtype=np.float32)
    return mixed


def subtract_params(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {
        "alpha": a["alpha"] - b["alpha"],
        "beta": a["beta"] - b["beta"],
        "sigma": a["sigma"] - b["sigma"],
        "lambda": a["lambda"] - b["lambda"],
        "b_norm": np.array(a["b_norm"], copy=True),
    }
    return out


def init_tracker_from_grad(grads: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return clone_params(grads)


def update_gradient_tracker(
    current_tracker: Mapping[str, np.ndarray],
    current_grads: Mapping[str, np.ndarray],
    prev_grads: Mapping[str, np.ndarray],
    w_mat: np.ndarray,
) -> Dict[str, np.ndarray]:
    new_tracker = {
        "alpha": np.zeros_like(current_tracker["alpha"]),
        "beta": np.array(current_grads["beta"], copy=True),
        "sigma": np.zeros_like(current_tracker["sigma"]),
        "lambda": np.array(current_grads["lambda"], copy=True),
        "b_norm": np.array(current_grads["b_norm"], copy=True),
    }
    diff_alpha = current_grads["alpha"] - prev_grads["alpha"]
    diff_sigma = current_grads["sigma"] - prev_grads["sigma"]
    for col in range(current_grads["alpha"].shape[1]):
        for row in range(current_grads["alpha"].shape[0]):
            weights = w_mat[row].reshape(-1, 1)
            new_tracker["alpha"][row, col] = (
                np.sum(weights * current_tracker["alpha"][:, col, :], axis=0, dtype=np.float32)
                + diff_alpha[row, col]
            )
            new_tracker["sigma"][row, col] = (
                np.sum(w_mat[row] * current_tracker["sigma"][:, col], dtype=np.float32)
                + diff_sigma[row, col]
            )
    return new_tracker


def estimate_spsa_gradient(
    builder: DistributedCostBuilderQiskit,
    params_template: Mapping[str, np.ndarray],
    flat_params: np.ndarray,
    rng: np.random.Generator,
    c: float,
) -> tuple[np.ndarray, float, float]:
    delta = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=flat_params.shape).astype(np.float32)
    theta_plus = flat_params + np.float32(c) * delta
    theta_minus = flat_params - np.float32(c) * delta
    loss_plus = builder.evaluate_total_loss(structured_from_flat(theta_plus, params_template))
    loss_minus = builder.evaluate_total_loss(structured_from_flat(theta_minus, params_template))
    grad = ((loss_plus - loss_minus) / (2.0 * float(c))) * delta
    return grad.astype(np.float32), float(loss_plus), float(loss_minus)


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _param_summary(params: Mapping[str, np.ndarray]) -> Dict[str, float]:
    return {
        "sigma_min": float(np.min(params["sigma"])),
        "sigma_max": float(np.max(params["sigma"])),
        "sigma_mean": float(np.mean(params["sigma"])),
        "sigma_l2": float(np.linalg.norm(params["sigma"])),
        "lambda_min": float(np.min(params["lambda"])),
        "lambda_max": float(np.max(params["lambda"])),
        "lambda_mean": float(np.mean(params["lambda"])),
        "lambda_l2": float(np.linalg.norm(params["lambda"])),
        "alpha_l2": float(np.linalg.norm(params["alpha"])),
        "beta_l2": float(np.linalg.norm(params["beta"])),
    }


def write_analysis(path: Path, args, system, builder: DistributedCostBuilderQiskit, w_mat: np.ndarray, params=None, diagnostics=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Qiskit/Aer MPS distributed 2x2 cluster30 run\n")
        f.write(f"static_ops module: {args.static_ops}\n")
        f.write(f"system name: {system.name}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"log_every: {args.log_every}\n")
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
            f.write(f"final parameter summary: {_param_summary(params)}\n")
        if diagnostics is not None:
            f.write(f"final diagnostics: {diagnostics}\n")


def write_final_params(path: Path, params: Mapping[str, np.ndarray]):
    payload = _to_jsonable(
        {
            "alpha": params["alpha"],
            "beta": params["beta"],
            "sigma": params["sigma"],
            "lambda": params["lambda"],
            "b_norm": params["b_norm"],
            "summary": _param_summary(params),
        }
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def maybe_plot(paths, loss_history, epochs_logged, residual_history, variance_history):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    if loss_history:
        plt.figure()
        plt.plot(np.arange(len(loss_history)), loss_history)
        plt.xlabel("Log Index")
        plt.ylabel("Cost")
        plt.yscale("log")
        plt.grid(True)
        plt.savefig(paths.fig_loss, dpi=200, bbox_inches="tight")
        plt.close()

    if residual_history:
        plt.figure()
        plt.plot(epochs_logged, residual_history)
        plt.xlabel("Epoch")
        plt.ylabel("||Ax-b||")
        plt.yscale("log")
        plt.grid(True)
        plt.savefig(paths.run_dir / "residual_norm.png", dpi=200, bbox_inches="tight")
        plt.close()

    if variance_history:
        plt.figure()
        plt.plot(epochs_logged, variance_history)
        plt.xlabel("Epoch")
        plt.ylabel("Consensus variance")
        plt.yscale("log")
        plt.grid(True)
        plt.savefig(paths.run_dir / "variance.png", dpi=200, bbox_inches="tight")
        plt.close()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static_ops", default="Qiskit_simulation.static_ops_2x2_cluster30_qiskit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--log_every", type=int, default=50)
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
    rng = np.random.default_rng(args.seed)

    builder = DistributedCostBuilderQiskit(
        system,
        TOPOLOGY_LINE_2,
        max_bond_dim=args.max_bond_dim,
        num_threads=args.num_threads,
        precision="single",
        seed=args.seed,
    )
    w_mat = build_metropolis_matrix(TOPOLOGY_LINE_2, n=system.n)

    params = initialize_global_params(system, n_qubits=n_qubits, seed=args.seed)
    logger.info(f"System variant: {system.name}")
    logger.info(f"static_ops module: {args.static_ops}")
    logger.info(f"local data qubits per agent: {n_qubits}")
    logger.info(f"epochs={args.epochs}, log_every={args.log_every}, lr={args.lr}, spsa_c={args.spsa_c}")
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

    current_loss = builder.evaluate_total_loss(params)
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

    flat_all = flatten_params(params, keys=TRAINABLE_KEYS)
    grad_flat_init, _, _ = estimate_spsa_gradient(builder, params, flat_all, rng, args.spsa_c)
    grad_grid = structured_from_flat(grad_flat_init, params)
    tracker_grid = init_tracker_from_grad(grad_grid)
    prev_grad_grid = clone_params(grad_grid)

    adam = AdamOptimizer(lr=args.lr)
    adam_state = adam.init(flat_all.size)

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

        flat_all = flatten_params(params, keys=TRAINABLE_KEYS)
        grad_flat, loss_plus, loss_minus = estimate_spsa_gradient(builder, params, flat_all, rng, args.spsa_c)
        grad_grid = structured_from_flat(grad_flat, params)
        tracker_grid = update_gradient_tracker(tracker_grid, grad_grid, prev_grad_grid, w_mat)
        prev_grad_grid = clone_params(grad_grid)

        if ep % args.log_every == 0:
            current_loss = builder.evaluate_total_loss(params)
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
