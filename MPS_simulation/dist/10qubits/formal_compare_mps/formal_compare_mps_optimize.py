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

from formal_compare_mps_common import (  # noqa: E402
    DEFAULT_PARAM_PATH,
    JsonlWriter,
    atomic_write_json,
    build_direct_problem,
    checkpoint_payload,
    compute_metrics,
    distributed_iteration,
    dump_yaml_config,
    ensure_output_dirs,
    global_cost_jax,
    gradient_norms,
    initialize_state,
    make_config,
    make_initial_parameters,
    metrics_entry,
    resolve_output_paths,
    sanitize_jsonable,
    state_to_numpy_snapshot,
    to_jax_problem,
    to_numpy_params,
    write_report,
    reconstruct_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a formal-vs-MPS apples-to-apples 2x2 Ising comparison with the MPS simulator."
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


def main() -> None:
    args = parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)

    cfg = make_config(args)
    paths = resolve_output_paths(cfg)
    ensure_output_dirs(paths)
    dump_yaml_config(paths["config"], sanitize_jsonable(asdict(cfg)))

    print(f"Preparing formal-compare MPS case `{cfg.case_name}`.", flush=True)
    problem_np = build_direct_problem(cfg)
    problem_jax = to_jax_problem(problem_np)

    params_init_np = make_initial_parameters(cfg)
    cost_fn = lambda params: global_cost_jax(params, cfg, problem_jax)
    value_and_grad_fn = jax.value_and_grad(cost_fn)
    grad_fn = jax.grad(cost_fn)

    writer = JsonlWriter(paths["history"])
    history: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    state = initialize_state(cfg, params_init_np, grad_fn)

    try:
        params_np = to_numpy_params(state["params"])
        grads_np = to_numpy_params(state["prev_grads"])
        init_metrics = compute_metrics(params_np, cfg, problem_np)
        init_entry = metrics_entry(
            0,
            init_metrics,
            gradient_norms(grads_np),
            learning_rate=cfg.learning_rate,
        )
        history.append(init_entry)
        writer.write(init_entry)
        atomic_write_json(
            paths["checkpoint"],
            checkpoint_payload(
                0,
                init_entry,
                params_np,
                state_np=state_to_numpy_snapshot(state),
            ),
        )
        print(
            f"Iteration    0 | cost={init_entry['global_cost']:.12g} "
            f"| residual={init_entry['global_residual_l2']:.12g} "
            f"| consensus={init_entry['consensus_error_l2']:.12g}",
            flush=True,
        )

        for _ in range(cfg.iterations):
            state, diagnostics = distributed_iteration(state, cfg, problem_jax, value_and_grad_fn)
            iteration = int(state["step"])
            params_np = to_numpy_params(state["params"])
            checkpoint_metrics = history[-1]

            if iteration % cfg.report_every == 0 or iteration == cfg.iterations:
                current_metrics = compute_metrics(params_np, cfg, problem_np)
                grad_info = gradient_norms(to_numpy_params(state["prev_grads"]))
                entry = metrics_entry(
                    iteration,
                    current_metrics,
                    grad_info,
                    learning_rate=float(diagnostics["learning_rate"]),
                )
                history.append(entry)
                writer.write(entry)
                checkpoint_metrics = entry
                print(
                    f"Iteration {iteration:4d} | cost={entry['global_cost']:.12g} "
                    f"| residual={entry['global_residual_l2']:.12g} "
                    f"| consensus={entry['consensus_error_l2']:.12g}",
                    flush=True,
                )

            atomic_write_json(
                paths["checkpoint"],
                checkpoint_payload(
                    iteration,
                    checkpoint_metrics,
                    params_np,
                    state_np=state_to_numpy_snapshot(state),
                ),
            )

    except Exception as exc:
        atomic_write_json(
            paths["checkpoint"],
            checkpoint_payload(
                int(state["step"]),
                history[-1] if history else {"status": "failed"},
                to_numpy_params(state["params"]),
                state_np=state_to_numpy_snapshot(state),
                failed=True,
                error_message=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise
    finally:
        writer.close()

    final_params_np = to_numpy_params(state["params"])
    final_diagnostics = reconstruct_diagnostics(final_params_np, cfg, problem_np)
    result = {
        "case": cfg.case_name,
        "config": sanitize_jsonable(asdict(cfg)),
        "problem": {
            "global_qubits": cfg.global_qubits,
            "local_qubits": cfg.local_qubits,
            "j_coupling": cfg.j_coupling,
            "kappa": cfg.kappa,
            "eta": float(problem_np["eta"]),
            "zeta": float(problem_np["zeta"]),
            "column_mix": problem_np["column_mix"].tolist(),
            "row_coeffs": problem_np["row_coeffs"].tolist(),
        },
        "optimization": {
            "iterations_requested": cfg.iterations,
            "iterations_completed": int(state["step"]),
            "elapsed_s": float(time.perf_counter() - start_time),
        },
        "final_state": sanitize_jsonable(final_params_np),
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
