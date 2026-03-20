#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from formal_single_agent_common import (  # noqa: E402
    DEFAULT_PARAM_PATH,
    JsonlWriter,
    atomic_write_json,
    build_final_diagnostics,
    build_problem_numpy,
    checkpoint_payload,
    compute_metrics,
    compute_rescaling_diagnostics,
    dump_yaml_config,
    encode_array,
    ensure_output_dirs,
    format_array_preview,
    global_cost_jax,
    make_config,
    make_initial_parameters,
    resolve_output_paths,
    sanitize_jsonable,
    sparse_matrix_preview,
    to_jax_params,
    to_jax_problem,
    to_numpy_params,
    wrap_angles_jax,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the formal-style single-agent MPS comparison benchmark."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_PARAM_PATH))
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--report-every", type=int, default=None)
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


def vector_norm(value: Any) -> float | None:
    if value is None:
        return None
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))


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
                "sigma_grad_l2": scalar_or_none(diagnostics.get("sigma_grad_l2")),
                "alpha_step_l2": scalar_or_none(diagnostics.get("alpha_step_l2")),
                "sigma_step_l2": scalar_or_none(diagnostics.get("sigma_step_l2")),
                "beta_grad_l2": scalar_or_none(diagnostics.get("beta_grad_l2")),
                "lambda_grad_l2": scalar_or_none(diagnostics.get("lambda_grad_l2")),
                "lr": scalar_or_none(diagnostics.get("lr")),
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
        f"| error={entry['solution_error_l2']:.12g} "
        f"| sigma_grad={entry.get('sigma_grad_l2', float('nan')):.12g}",
        flush=True,
    )


def should_record(iteration: int, total_iterations: int, report_every: int) -> bool:
    if iteration == 0:
        return True
    if iteration == total_iterations:
        return True
    return iteration % report_every == 0


def init_optimizer_state(params):
    import jax.numpy as jnp

    return {
        "step": 0,
        "m_alpha": jnp.zeros_like(params["alpha"]),
        "v_alpha": jnp.zeros_like(params["alpha"]),
        "m_sigma": jnp.zeros_like(params["sigma"]),
        "v_sigma": jnp.zeros_like(params["sigma"]),
    }


def snapshot_optimizer_state(opt_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": int(opt_state["step"]),
        "m_alpha": encode_array(np.asarray(opt_state["m_alpha"], dtype=np.float64)),
        "v_alpha": encode_array(np.asarray(opt_state["v_alpha"], dtype=np.float64)),
        "m_sigma": float(np.asarray(opt_state["m_sigma"], dtype=np.float64)),
        "v_sigma": float(np.asarray(opt_state["v_sigma"], dtype=np.float64)),
    }


def adam_like_step(params, grads, opt_state, cfg):
    import jax.numpy as jnp

    step = int(opt_state["step"]) + 1
    lr = cfg.learning_rate * (cfg.decay ** (step - 1))

    g_alpha = grads["alpha"]
    g_sigma = grads["sigma"]

    m_alpha = cfg.adam_beta1 * opt_state["m_alpha"] + (1.0 - cfg.adam_beta1) * g_alpha
    v_alpha = cfg.adam_beta2 * opt_state["v_alpha"] + (1.0 - cfg.adam_beta2) * (g_alpha * g_alpha)
    m_sigma = cfg.adam_beta1 * opt_state["m_sigma"] + (1.0 - cfg.adam_beta1) * g_sigma
    v_sigma = cfg.adam_beta2 * opt_state["v_sigma"] + (1.0 - cfg.adam_beta2) * (g_sigma * g_sigma)

    mhat_alpha = m_alpha / (1.0 - cfg.adam_beta1**step)
    vhat_alpha = v_alpha / (1.0 - cfg.adam_beta2**step)
    mhat_sigma = m_sigma / (1.0 - cfg.adam_beta1**step)
    vhat_sigma = v_sigma / (1.0 - cfg.adam_beta2**step)

    alpha_step = lr * mhat_alpha / (jnp.sqrt(vhat_alpha) + cfg.adam_epsilon)
    sigma_step = lr * mhat_sigma / (jnp.sqrt(vhat_sigma) + cfg.adam_epsilon)

    new_params = {
        "alpha": wrap_angles_jax(params["alpha"] - alpha_step),
        "beta": params["beta"],
        "sigma": params["sigma"] - sigma_step,
        "lambda": params["lambda"],
    }
    new_opt_state = {
        "step": step,
        "m_alpha": m_alpha,
        "v_alpha": v_alpha,
        "m_sigma": m_sigma,
        "v_sigma": v_sigma,
    }
    diagnostics = {
        "step": step,
        "lr": lr,
        "alpha_step_l2": jnp.linalg.norm(alpha_step),
        "sigma_step_l2": jnp.abs(sigma_step),
    }
    return new_params, new_opt_state, diagnostics


def main() -> None:
    args = parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)

    cfg = make_config(args)
    paths = resolve_output_paths(cfg)
    ensure_output_dirs(paths)
    dump_yaml_config(paths["config"], sanitize_jsonable(asdict(cfg)))

    print(f"Preparing formal-style single-agent case `{cfg.case_name}`.", flush=True)
    problem_np = build_problem_numpy(cfg)
    problem_jax = to_jax_problem(problem_np, cfg)

    params_np = make_initial_parameters(cfg)
    params = to_jax_params(params_np)
    opt_state = init_optimizer_state(params)

    cost_fn = lambda current_params: global_cost_jax(current_params, cfg, problem_jax)
    value_and_grad_fn = jax.value_and_grad(cost_fn)

    writer = JsonlWriter(paths["history"])
    history: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    try:
        current_cost, current_grads = value_and_grad_fn(params)
        initial_metrics = compute_metrics(params_np, cfg, problem_np)
        initial_entry = history_entry(
            0,
            initial_metrics,
            {
                "current_cost": current_cost,
                "alpha_grad_l2": vector_norm(current_grads["alpha"]),
                "sigma_grad_l2": scalar_or_none(np.abs(np.asarray(current_grads["sigma"], dtype=np.float64))),
                "alpha_step_l2": 0.0,
                "sigma_step_l2": 0.0,
                "beta_grad_l2": vector_norm(current_grads["beta"]),
                "lambda_grad_l2": scalar_or_none(np.abs(np.asarray(current_grads["lambda"], dtype=np.float64))),
                "lr": cfg.learning_rate,
            },
        )
        history.append(initial_entry)
        writer.write(initial_entry)
        atomic_write_json(
            paths["checkpoint"],
            checkpoint_payload(
                0,
                initial_entry,
                params_np,
                optimizer_state=snapshot_optimizer_state(opt_state),
            ),
        )
        print_progress(initial_entry, force=True)

        for _ in range(cfg.iterations):
            params, opt_state, update_diag = adam_like_step(params, current_grads, opt_state, cfg)
            current_cost, current_grads = value_and_grad_fn(params)
            params_np = to_numpy_params(params)
            iteration = int(opt_state["step"])

            checkpoint_metrics = history[-1]
            if should_record(iteration, cfg.iterations, cfg.report_every):
                metrics = compute_metrics(params_np, cfg, problem_np)
                entry = history_entry(
                    iteration,
                    metrics,
                    {
                        "current_cost": current_cost,
                        "alpha_grad_l2": vector_norm(current_grads["alpha"]),
                        "sigma_grad_l2": scalar_or_none(np.abs(np.asarray(current_grads["sigma"], dtype=np.float64))),
                        "alpha_step_l2": update_diag["alpha_step_l2"],
                        "sigma_step_l2": update_diag["sigma_step_l2"],
                        "beta_grad_l2": vector_norm(current_grads["beta"]),
                        "lambda_grad_l2": scalar_or_none(np.abs(np.asarray(current_grads["lambda"], dtype=np.float64))),
                        "lr": update_diag["lr"],
                    },
                )
                history.append(entry)
                writer.write(entry)
                checkpoint_metrics = entry
                print_progress(entry, force=True)

            atomic_write_json(
                paths["checkpoint"],
                checkpoint_payload(
                    iteration,
                    checkpoint_metrics,
                    params_np,
                    optimizer_state=snapshot_optimizer_state(opt_state),
                ),
            )

    except Exception as exc:
        atomic_write_json(
            paths["checkpoint"],
            checkpoint_payload(
                int(history[-1]["iteration"]) if history else int(opt_state.get("step", 0)),
                history[-1] if history else {"status": "failed"},
                params_np,
                optimizer_state=snapshot_optimizer_state(opt_state),
                failed=True,
                error_message=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise
    finally:
        writer.close()

    final_metrics = compute_metrics(params_np, cfg, problem_np)
    if not history or int(history[-1]["iteration"]) != cfg.iterations:
        final_entry = history_entry(cfg.iterations, final_metrics, None)
        history.append(final_entry)
        atomic_write_json(
            paths["checkpoint"],
            checkpoint_payload(
                cfg.iterations,
                final_entry,
                params_np,
                optimizer_state=snapshot_optimizer_state(opt_state),
            ),
        )

    elapsed_s = time.perf_counter() - start_time
    final_diagnostics = build_final_diagnostics(params_np, cfg, problem_np)
    rescaled_diagnostics = compute_rescaling_diagnostics(
        np.asarray(final_metrics["x_estimate"]),
        np.asarray(problem_np["x_true"]),
        problem_np["a_sparse"],
        np.asarray(problem_np["b_dense"]),
    )

    result = {
        "case": cfg.case_name,
        "config": sanitize_jsonable(asdict(cfg)),
        "problem": {
            "global_qubits": cfg.global_qubits,
            "local_qubits": cfg.local_qubits,
            "global_dim": cfg.global_dim,
            "j_coupling": cfg.j_coupling,
            "kappa": cfg.kappa,
            "eta": float(problem_np["eta"]),
            "zeta": float(problem_np["zeta"]),
            "lambda_min": float(problem_np["lambda_min"]),
            "lambda_max": float(problem_np["lambda_max"]),
            "scaled_lambda_min": float(problem_np["scaled_lambda_min"]),
            "scaled_lambda_max": float(problem_np["scaled_lambda_max"]),
            "b_norm": float(problem_np["b_norm"]),
        },
        "optimization": {
            "iterations_requested": cfg.iterations,
            "iterations_completed": int(history[-1]["iteration"]),
            "elapsed_s": float(elapsed_s),
        },
        "final_state": {
            "alpha": encode_array(np.asarray(params_np["alpha"], dtype=np.float64)),
            "beta": encode_array(np.asarray(params_np["beta"], dtype=np.float64)),
            "sigma": float(np.asarray(params_np["sigma"], dtype=np.float64)),
            "lambda": float(np.asarray(params_np["lambda"], dtype=np.float64)),
            "alpha_preview": format_array_preview(np.asarray(params_np["alpha"]), max_elements=cfg.preview_elements),
            "beta_preview": format_array_preview(np.asarray(params_np["beta"]), max_elements=cfg.preview_elements),
            "x_estimate": encode_array(np.asarray(final_metrics["x_estimate"])),
            "x_true": encode_array(np.asarray(problem_np["x_true"])),
            "x_estimate_preview": format_array_preview(
                np.asarray(final_metrics["x_estimate"]),
                max_elements=cfg.preview_elements,
            ),
            "x_true_preview": format_array_preview(
                np.asarray(problem_np["x_true"]),
                max_elements=cfg.preview_elements,
            ),
        },
        "linear_system": {
            "A_sparse_preview": sparse_matrix_preview(problem_np["a_sparse"], max_elements=cfg.preview_elements),
            "b_preview": format_array_preview(np.asarray(problem_np["b_dense"]), max_elements=cfg.preview_elements),
        },
        "history": history,
        "final_diagnostics": final_diagnostics,
        "rescaled_diagnostics": rescaled_diagnostics,
        "artifacts": {name: str(path.resolve()) for name, path in paths.items()},
    }

    atomic_write_json(paths["json"], result)
    write_report(paths["report"], result)

    print(
        "Completed formal-style single-agent comparison "
        f"in {elapsed_s:.6f} s. Final cost={history[-1]['global_cost']:.12g}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
