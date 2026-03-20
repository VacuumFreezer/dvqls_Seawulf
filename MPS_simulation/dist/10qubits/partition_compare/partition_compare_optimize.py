#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from partition_compare_common import (  # noqa: E402
    DEFAULT_PARAM_PATH,
    JsonlWriter,
    atomic_write_json,
    build_partitioned_problem_numpy,
    build_final_diagnostics,
    checkpoint_payload,
    compute_metrics,
    compute_rescaling_diagnostics,
    distributed_iteration,
    dump_yaml_config,
    encode_array,
    ensure_output_dirs,
    format_array_preview,
    global_cost_jax,
    initialize_state,
    make_config,
    make_initial_parameters,
    resolve_output_paths,
    sanitize_jsonable,
    sparse_matrix_preview,
    to_jax_problem,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the generalized 13-qubit partition comparison optimizer for "
            "8x8, 4x4, 2x2, or 1x1 agent layouts."
        )
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_PARAM_PATH))
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--report-every", type=int, default=None)
    parser.add_argument(
        "--init-mode",
        type=str,
        choices=("normal_angles", "uniform_block_norm_j0", "normal_block_norm_j0", "random_uniform", "structured_linspace"),
        default=None,
    )
    parser.add_argument("--init-seed", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument("--out-report", type=str, default=None)
    parser.add_argument("--out-history", type=str, default=None)
    parser.add_argument("--out-checkpoint", type=str, default=None)
    parser.add_argument("--out-config", type=str, default=None)
    return parser.parse_args()


def scalar_or_none(value: Any) -> float | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.size != 1:
        raise ValueError(f"Expected scalar-compatible value, got shape {array.shape}.")
    return float(array.reshape(()))


def history_entry(iteration: int, metrics: dict[str, Any], diagnostics: dict[str, Any] | None) -> dict[str, Any]:
    entry = {
        "iteration": int(iteration),
        "global_cost": float(metrics["global_cost"]),
        "global_residual_l2": float(metrics["global_residual_l2"]),
        "consensus_error_l2": float(metrics["consensus_error_l2"]),
        "solution_error_l2": float(metrics["solution_error_l2"]),
        "relative_solution_error_l2": float(metrics["relative_solution_error_l2"]),
    }
    if diagnostics is not None:
        entry.update(
            {
                "tracked_cost": scalar_or_none(diagnostics.get("current_cost")),
                "alpha_grad_l2": scalar_or_none(diagnostics.get("alpha_grad_l2")),
                "beta_grad_l2": scalar_or_none(diagnostics.get("beta_grad_l2")),
                "alpha_step_l2": scalar_or_none(diagnostics.get("alpha_step_l2")),
                "beta_step_l2": scalar_or_none(diagnostics.get("beta_step_l2")),
            }
        )
    return entry


def print_progress(entry: dict[str, Any], *, force: bool = False) -> None:
    iteration = int(entry["iteration"])
    if not force and iteration == 0:
        return
    print(
        "Iteration "
        f"{iteration:4d} | cost={entry['global_cost']:.12g} "
        f"| residual={entry['global_residual_l2']:.12g} "
        f"| consensus={entry['consensus_error_l2']:.12g} "
        f"| error={entry['solution_error_l2']:.12g}",
        flush=True,
    )


def snapshot_optimizer_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": int(state["step"]),
        "y": encode_array(np.asarray(state["y"], dtype=np.float64)),
        "a_alpha": encode_array(np.asarray(state["a_alpha"], dtype=np.float64)),
        "b_alpha": encode_array(np.asarray(state["b_alpha"], dtype=np.float64)),
        "a_beta": encode_array(np.asarray(state["a_beta"], dtype=np.float64)),
        "b_beta": encode_array(np.asarray(state["b_beta"], dtype=np.float64)),
    }


def should_record(iteration: int, total_iterations: int, report_every: int) -> bool:
    if iteration == 0:
        return True
    if iteration == total_iterations:
        return True
    return iteration % report_every == 0


def main() -> None:
    args = parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)

    cfg = make_config(args)
    paths = resolve_output_paths(cfg)
    ensure_output_dirs(paths)
    dump_yaml_config(paths["config"], sanitize_jsonable(asdict(cfg)))

    print(f"Preparing partition-compare case `{cfg.case_name}`.", flush=True)
    problem_np = build_partitioned_problem_numpy(cfg)
    problem_jax = to_jax_problem(problem_np)

    alpha_init_np, beta_init_np = make_initial_parameters(cfg)
    if cfg.use_messenger:
        cost_fn = lambda alpha, beta: global_cost_jax(alpha, beta, cfg, problem_jax)
        full_grad_fn = jax.value_and_grad(cost_fn, argnums=(0, 1))
        alpha_grad_fn = jax.grad(cost_fn, argnums=0)
    else:
        cost_fn = lambda alpha: global_cost_jax(alpha, None, cfg, problem_jax)
        full_grad_fn = jax.value_and_grad(cost_fn)
        alpha_grad_fn = jax.grad(cost_fn)

    writer = JsonlWriter(paths["history"])
    history: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    state = initialize_state(cfg, alpha_grad_fn, alpha_init_np, beta_init_np)
    last_alpha_np = np.asarray(alpha_init_np, dtype=np.float64)
    last_beta_np = np.asarray(beta_init_np, dtype=np.float64)
    last_metrics_full: dict[str, Any] | None = None

    try:
        if cfg.use_messenger:
            initial_cost, (initial_alpha_grad, initial_beta_grad) = full_grad_fn(state["alpha"], state["beta"])
        else:
            initial_cost, initial_alpha_grad = full_grad_fn(state["alpha"])
            initial_beta_grad = None
        initial_metrics = compute_metrics(last_alpha_np, last_beta_np, cfg, problem_np)
        initial_entry = history_entry(
            0,
            initial_metrics,
            {
                "current_cost": initial_cost,
                "alpha_grad_l2": np.linalg.norm(np.asarray(initial_alpha_grad, dtype=np.float64)),
                "beta_grad_l2": None
                if not cfg.use_messenger
                else np.linalg.norm(np.asarray(initial_beta_grad, dtype=np.float64)),
                "alpha_step_l2": 0.0,
                "beta_step_l2": None if not cfg.use_messenger else 0.0,
            },
        )
        history.append(initial_entry)
        writer.write(initial_entry)
        atomic_write_json(
            paths["checkpoint"],
            checkpoint_payload(
                0,
                initial_entry,
                last_alpha_np,
                last_beta_np,
                optimizer_state=snapshot_optimizer_state(state),
            ),
        )
        last_metrics_full = initial_metrics
        print_progress(initial_entry, force=True)

        for _ in range(cfg.iterations):
            state, diagnostics = distributed_iteration(state, cfg, problem_jax, full_grad_fn, alpha_grad_fn)
            iteration = int(state["step"])
            last_alpha_np = np.asarray(state["alpha"], dtype=np.float64)
            last_beta_np = np.asarray(state["beta"], dtype=np.float64)
            checkpoint_metrics = history[-1] if history else {"status": "running"}
            if should_record(iteration, cfg.iterations, cfg.report_every):
                last_metrics_full = compute_metrics(last_alpha_np, last_beta_np, cfg, problem_np)
                entry = history_entry(iteration, last_metrics_full, diagnostics)
                history.append(entry)
                writer.write(entry)
                checkpoint_metrics = entry
                print_progress(entry, force=True)
            atomic_write_json(
                paths["checkpoint"],
                checkpoint_payload(
                    iteration,
                    checkpoint_metrics,
                    last_alpha_np,
                    last_beta_np,
                    optimizer_state=snapshot_optimizer_state(state),
                ),
            )

    except Exception as exc:
        atomic_write_json(
            paths["checkpoint"],
            checkpoint_payload(
                int(history[-1]["iteration"]) if history else int(state.get("step", 0)),
                history[-1] if history else {"status": "failed"},
                np.asarray(state["alpha"], dtype=np.float64),
                np.asarray(state["beta"], dtype=np.float64),
                optimizer_state=snapshot_optimizer_state(state),
                failed=True,
                error_message=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise
    finally:
        writer.close()

    if last_metrics_full is None or int(history[-1]["iteration"]) != cfg.iterations:
        last_metrics_full = compute_metrics(last_alpha_np, last_beta_np, cfg, problem_np)

    elapsed_s = time.perf_counter() - start_time
    final_diagnostics = build_final_diagnostics(last_alpha_np, last_beta_np, cfg, problem_np)
    rescaled_diagnostics = compute_rescaling_diagnostics(
        np.asarray(last_metrics_full["x_estimate"], dtype=np.float64),
        np.asarray(problem_np["x_true"], dtype=np.float64),
        problem_np["a_sparse"],
        np.asarray(problem_np["b_dense"], dtype=np.float64),
    )
    b_row_norms = [float(np.linalg.norm(np.asarray(row, dtype=np.float64))) for row in problem_np["b_rows"]]
    b_agent_norms = [
        [float(np.linalg.norm(np.asarray(vec, dtype=np.float64))) for vec in row_parts]
        for row_parts in problem_np["b_vectors"]
    ]
    b_column_split = (
        np.full((cfg.n_rows, cfg.n_cols), 1.0 / cfg.n_cols, dtype=np.float64).tolist()
        if cfg.n_cols > 0
        else None
    )
    result = {
        "case": cfg.case_name,
        "config": sanitize_jsonable(asdict(cfg)),
        "problem": {
            "global_qubits": cfg.global_qubits,
            "local_qubits": cfg.local_qubits,
            "n_rows": cfg.n_rows,
            "n_cols": cfg.n_cols,
            "global_dim": cfg.global_dim,
            "local_dim": cfg.local_dim,
            "j_coupling": cfg.j_coupling,
            "kappa": cfg.kappa,
            "eta": float(problem_np["eta"]),
            "zeta": float(problem_np["zeta"]),
            "lambda_min": float(problem_np["lambda_min"]),
            "lambda_max": float(problem_np["lambda_max"]),
            "scaled_lambda_min": float(problem_np["scaled_lambda_min"]),
            "scaled_lambda_max": float(problem_np["scaled_lambda_max"]),
            "use_messenger": cfg.use_messenger,
            "use_consensus": cfg.use_consensus,
            "column_mix": np.asarray(problem_np["column_mix"], dtype=np.float64).tolist(),
            "row_laplacian": np.asarray(problem_np["row_laplacian"], dtype=np.float64).tolist(),
            "b_row_norms": b_row_norms,
            "b_agent_norms": b_agent_norms,
            "b_column_split": b_column_split,
        },
        "optimization": {
            "iterations_requested": cfg.iterations,
            "iterations_completed": int(history[-1]["iteration"]),
            "elapsed_s": float(elapsed_s),
        },
        "final_state": {
            "alpha": encode_array(last_alpha_np),
            "beta": encode_array(last_beta_np),
            "alpha_preview": format_array_preview(last_alpha_np, max_elements=cfg.preview_elements),
            "beta_preview": format_array_preview(last_beta_np, max_elements=cfg.preview_elements),
            "x_estimate": encode_array(np.asarray(last_metrics_full["x_estimate"], dtype=np.float64)),
            "x_true": encode_array(np.asarray(problem_np["x_true"], dtype=np.float64)),
            "x_estimate_preview": format_array_preview(
                np.asarray(last_metrics_full["x_estimate"], dtype=np.float64),
                max_elements=cfg.preview_elements,
            ),
            "x_true_preview": format_array_preview(
                np.asarray(problem_np["x_true"], dtype=np.float64),
                max_elements=cfg.preview_elements,
            ),
        },
        "linear_system": {
            "A_sparse_preview": sparse_matrix_preview(problem_np["a_sparse"], max_elements=cfg.preview_elements),
            "b_preview": format_array_preview(
                np.asarray(problem_np["b_dense"], dtype=np.float64),
                max_elements=cfg.preview_elements,
            ),
        },
        "rescaled_diagnostics": rescaled_diagnostics,
        "final_diagnostics": final_diagnostics,
        "history": history,
        "artifacts": {
            "json": str(paths["json"].resolve()),
            "report": str(paths["report"].resolve()),
            "history": str(paths["history"].resolve()),
            "checkpoint": str(paths["checkpoint"].resolve()),
            "config": str(paths["config"].resolve()),
        },
    }

    atomic_write_json(paths["json"], result)
    write_report(paths["report"], result)
    print(f"Wrote JSON report to {paths['json']}", flush=True)
    print(f"Wrote markdown report to {paths['report']}", flush=True)


if __name__ == "__main__":
    main()
