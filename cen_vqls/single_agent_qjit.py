#!/usr/bin/env python3
from __future__ import annotations
"""
Centralized VQLS for Ising instances, optimizing the global residual ||Ax-b||^2 directly as the objective.
Equvialent to a single agent for distribued VQLS.
"""

import argparse
import copy
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pennylane as qml
import yaml

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from common.reporting import JsonlWriter, setup_logger

try:
    from .centralized_vqls_ising import (
        _format_complex_vector_preview,
        _to_float,
        load_centralized_data,
        write_report,
    )
    from .centralized_vqls_ising_qjit import (
        QJITHadamardCentralizedVQLS,
        append_qjit_report_section,
        detect_runtime_backend,
    )
except ImportError:
    from centralized_vqls_ising import (
        _format_complex_vector_preview,
        _to_float,
        load_centralized_data,
        write_report,
    )
    from centralized_vqls_ising_qjit import (
        QJITHadamardCentralizedVQLS,
        append_qjit_report_section,
        detect_runtime_backend,
    )


DEFAULT_CONFIG = {
    "problem": {
        "static_ops_path": "problems/static_ops_16agents_Ising.py",
        "consistency_atol": 1.0e-12,
        "b_state_tolerance": 1.0e-10,
    },
    "ansatz": {
        "layers": 7,
        "init_low": -3.141592653589793,
        "init_high": 3.141592653589793,
        "init_sigma": 1.0,
    },
    "optimization": {
        "steps": 200,
        "learning_rate": 0.03,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1.0e-8,
        "print_every": 20,
        "optimize_metric": "global_residual_sq",
    },
    "runtime": {
        "seed": 0,
        "device": "lightning.qubit",
        "interface": "jax",
        "diff_method": "adjoint",
        "use_qjit": True,
        "fallback_to_jax_jit": True,
        "qjit_autograph": False,
        "jit_training_step": True,
        "warmup_compile": True,
    },
    "report": {
        "out_dir": "cen_vqls/reports",
        "tag": "centralized_vqls_ising_residual",
    },
}


def _deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: Path) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Top-level YAML config must be a mapping.")
    _deep_update(cfg, loaded)
    return cfg


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y"}:
        return True
    if v in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool from: {value}")


def apply_cli_overrides(cfg: dict, args, repo_root: Path) -> dict:
    cfg = copy.deepcopy(cfg)

    if args.static_ops_path is not None:
        cfg["problem"]["static_ops_path"] = str(args.static_ops_path)

    if args.layers is not None:
        cfg["ansatz"]["layers"] = int(args.layers)

    if args.steps is not None:
        cfg["optimization"]["steps"] = int(args.steps)
    if args.learning_rate is not None:
        cfg["optimization"]["learning_rate"] = float(args.learning_rate)
    if args.print_every is not None:
        cfg["optimization"]["print_every"] = int(args.print_every)
    if args.optimize_metric is not None:
        cfg["optimization"]["optimize_metric"] = str(args.optimize_metric)

    if args.seed is not None:
        cfg["runtime"]["seed"] = int(args.seed)
    if args.device is not None:
        cfg["runtime"]["device"] = str(args.device)
    if args.diff_method is not None:
        cfg["runtime"]["diff_method"] = str(args.diff_method)

    use_qjit = _parse_optional_bool(args.use_qjit)
    if use_qjit is not None:
        cfg["runtime"]["use_qjit"] = use_qjit

    fallback = _parse_optional_bool(args.fallback_to_jax_jit)
    if fallback is not None:
        cfg["runtime"]["fallback_to_jax_jit"] = fallback

    if args.out is not None:
        out_dir = Path(args.out)
        if not out_dir.is_absolute():
            out_dir = repo_root / out_dir
        cfg["report"]["out_dir"] = out_dir.as_posix()

    if args.tag is not None:
        cfg["report"]["tag"] = str(args.tag)

    return cfg


def _residual_sq_from_terms(beta, gamma_re, sigma, b_norm: float):
    return sigma * sigma * beta + (b_norm * b_norm) - (2.0 * sigma * b_norm * gamma_re)


@dataclass
class ResidualL2Info:
    abs_error_raw: float
    rel_error_raw: float
    abs_error_aligned: float
    rel_error_aligned: float
    phase_angle_rad: float


def compute_l2_error_from_sigma(
    evaluator: QJITHadamardCentralizedVQLS,
    weights: jnp.ndarray,
    sigma: float,
    x_true: np.ndarray,
) -> ResidualL2Info:
    x_norm = np.array(evaluator.state(weights), dtype=np.complex128)
    x_est_unnorm = float(sigma) * x_norm

    x_true_norm = float(np.linalg.norm(x_true))
    abs_err_raw = float(np.linalg.norm(x_est_unnorm - x_true))
    rel_err_raw = float(abs_err_raw / (x_true_norm + 1.0e-14))

    overlap = np.vdot(x_est_unnorm, x_true)
    phase_angle = float(np.angle(overlap)) if np.abs(overlap) > 1.0e-16 else 0.0
    x_est_aligned = x_est_unnorm * np.exp(-1.0j * phase_angle)
    abs_err_aligned = float(np.linalg.norm(x_est_aligned - x_true))
    rel_err_aligned = float(abs_err_aligned / (x_true_norm + 1.0e-14))

    return ResidualL2Info(
        abs_error_raw=abs_err_raw,
        rel_error_raw=rel_err_raw,
        abs_error_aligned=abs_err_aligned,
        rel_error_aligned=rel_err_aligned,
        phase_angle_rad=phase_angle,
    )


def append_residual_section(report_path: Path, best: dict, final_metrics: Dict[str, float]) -> None:
    lines = []
    lines.append("")
    lines.append("## Residual-Cost Objective")
    lines.append("- Objective (single-agent/global):")
    lines.append("  - `||Ax-b||^2 = sigma^2 * <X|A^dagger A|X> + ||b||^2 - 2*sigma*||b||*Re(<B|A|X>)`")
    lines.append("- All expectation terms are evaluated by Hadamard-test circuits.")
    lines.append("- L2 metrics in single-agent mode use explicit sigma reconstruction:")
    lines.append("  - `x_est = sigma * |X>` with global phase alignment for the aligned L2.")
    lines.append(f"- Best residual^2 iteration: {best['iteration']}")
    lines.append(f"- Best residual^2: {best['loss']:.12e}")
    lines.append(f"- Final residual^2: {final_metrics['global_residual_sq']:.12e}")
    lines.append(f"- Final residual norm: {final_metrics['global_residual_norm']:.12e}")
    lines.append(f"- Final sigma: {final_metrics['sigma']:.12e}")
    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Centralized residual VQLS (Catalyst/JIT): minimize global residual ||Ax-b||^2"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().with_name("config_residual.yaml")),
        help="Path to YAML config file.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--print_every", type=int, default=None)
    parser.add_argument("--optimize_metric", type=str, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--static_ops_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--diff_method", type=str, default=None)
    parser.add_argument("--use_qjit", type=str, default=None)
    parser.add_argument("--fallback_to_jax_jit", type=str, default=None)
    parser.add_argument("--out", type=str, default=None, help="Override report.out_dir")
    parser.add_argument("--tag", type=str, default=None, help="Override report.tag")
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)

    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(Path(args.config).resolve())
    cfg = apply_cli_overrides(cfg, args, repo_root=repo_root)
    backend_info = detect_runtime_backend(cfg)

    data = load_centralized_data(cfg, repo_root=repo_root)
    report_cfg = cfg["report"]
    out_dir = Path(report_cfg["out_dir"])
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"{report_cfg['tag']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    logger = setup_logger(run_dir / "report.txt")
    metrics_writer = JsonlWriter(run_dir / "metrics.jsonl")

    logger.info("[consistency] formula/block max abs diff: %s", f"{data.matrix_max_abs_diff:.6e}")
    logger.info("[consistency] formula/block fro diff: %s", f"{data.matrix_fro_diff:.6e}")
    logger.info("[consistency] allclose: %s", data.matrix_allclose)
    logger.info("[matrix] cond(A): %s", f"{data.condition_number:.12e}")
    if not data.matrix_allclose:
        raise RuntimeError(
            "Global matrix from Pauli-term construction is not consistent with the block-partition matrix. "
            "Please check static-ops parameters and wire ordering."
        )

    logger.info("[backend] requested_qjit: %s", backend_info.requested_qjit)
    logger.info("[backend] active: %s", backend_info.backend)
    if backend_info.catalyst_error:
        logger.info("[backend] catalyst import issue: %s", backend_info.catalyst_error)

    evaluator = QJITHadamardCentralizedVQLS(
        data=data,
        device_name=str(cfg["runtime"]["device"]),
        diff_method=str(cfg["runtime"]["diff_method"]),
        backend=backend_info.backend,
        qjit_autograph=bool(cfg["runtime"].get("qjit_autograph", False)),
    )

    x_true = np.linalg.solve(data.a_block, data.b_unnorm)
    b_norm = float(data.b_unnorm_norm)

    rng = np.random.default_rng(int(cfg["runtime"]["seed"]))
    layers = int(cfg["ansatz"]["layers"])
    theta = jnp.asarray(
        rng.uniform(
            float(cfg["ansatz"]["init_low"]),
            float(cfg["ansatz"]["init_high"]),
            size=(layers, data.n_total_qubits),
        ),
        dtype=jnp.float64,
    )
    sigma = jnp.asarray(float(cfg["ansatz"].get("init_sigma", 1.0)), dtype=jnp.float64)

    optimize_metric = str(cfg["optimization"]["optimize_metric"])
    if optimize_metric != "global_residual_sq":
        raise ValueError("This script optimizes only `global_residual_sq`.")

    def objective_fn(params):
        w, s = params
        vals = evaluator.metrics(w)
        return _residual_sq_from_terms(vals["beta"], vals["gamma_re"], s, b_norm=b_norm)

    value_and_grad = jax.value_and_grad(objective_fn)

    opt = optax.adam(
        learning_rate=float(cfg["optimization"]["learning_rate"]),
        b1=float(cfg["optimization"]["beta1"]),
        b2=float(cfg["optimization"]["beta2"]),
        eps=float(cfg["optimization"]["eps"]),
    )
    params = (theta, sigma)
    opt_state = opt.init(params)

    if bool(cfg["runtime"].get("jit_training_step", True)):

        @jax.jit
        def train_step(cur_params, state):
            loss, grads = value_and_grad(cur_params)
            updates, state = opt.update(grads, state, cur_params)
            cur_params = optax.apply_updates(cur_params, updates)
            return cur_params, state, loss

    else:

        def train_step(cur_params, state):
            loss, grads = value_and_grad(cur_params)
            updates, state = opt.update(grads, state, cur_params)
            cur_params = optax.apply_updates(cur_params, updates)
            return cur_params, state, loss

    if bool(cfg["runtime"].get("warmup_compile", True)):
        warmup_loss = objective_fn(params)
        _ = jax.block_until_ready(warmup_loss)

    steps = int(cfg["optimization"]["steps"])
    print_every = int(cfg["optimization"]["print_every"])
    history: List[dict] = []
    checkpoints: List[dict] = []
    best = {"iteration": 0, "loss": float("inf")}
    t0 = time.time()

    for it in range(1, steps + 1):
        params, opt_state, loss_val = train_step(params, opt_state)
        loss_val = jax.block_until_ready(loss_val)
        theta, sigma = params

        residual_sq = max(0.0, _to_float(loss_val))
        residual_norm = float(np.sqrt(residual_sq))
        sigma_f = float(np.asarray(sigma))
        now_iso = datetime.now().isoformat(timespec="milliseconds")
        elapsed_s = float(time.time() - t0)

        l2 = compute_l2_error_from_sigma(evaluator=evaluator, weights=theta, sigma=sigma_f, x_true=x_true)

        row = {
            "iteration": it,
            "time_iso": now_iso,
            "elapsed_s": elapsed_s,
            "global_residual_sq": residual_sq,
            "global_residual_norm": residual_norm,
            "sigma": sigma_f,
            "l2_abs_raw": l2.abs_error_raw,
            "l2_rel_raw": l2.rel_error_raw,
            "l2_abs_aligned": l2.abs_error_aligned,
            "l2_rel_aligned": l2.rel_error_aligned,
            "phase_angle_rad": l2.phase_angle_rad,
        }
        history.append(row)

        if residual_sq < best["loss"]:
            best = {"iteration": it, "loss": residual_sq}

        if (it % print_every == 0) or (it == 1) or (it == steps):
            m = evaluator.metrics(theta)
            check = dict(row)
            check["beta"] = _to_float(m["beta"])
            check["gamma_re"] = _to_float(m["gamma_re"])
            check["gamma_im"] = _to_float(m["gamma_im"])
            check["global_CG"] = _to_float(m["global_CG"])
            check["global_CL"] = _to_float(m["global_CL"])
            check["global_CG_hat"] = _to_float(m["global_CG_hat"])
            check["global_CL_hat"] = _to_float(m["global_CL_hat"])
            checkpoints.append(check)
            metrics_writer.write(check)
            logger.info(
                "[iter %04d] residual^2=%.8e residual=%.8e sigma=%.8e CG=%.8e CL=%.8e "
                "L2_rel(aligned)=%.8e backend=%s",
                it,
                residual_sq,
                residual_norm,
                sigma_f,
                check["global_CG"],
                check["global_CL"],
                l2.rel_error_aligned,
                backend_info.backend,
            )

    theta, sigma = params
    final_raw = evaluator.metrics(theta)
    final_sigma = float(np.asarray(sigma))
    final_residual_sq = max(0.0, _to_float(objective_fn(params)))
    final_residual_norm = float(np.sqrt(final_residual_sq))
    final_l2 = compute_l2_error_from_sigma(evaluator=evaluator, weights=theta, sigma=final_sigma, x_true=x_true)

    final_metrics: Dict[str, float] = {
        "global_CG": _to_float(final_raw["global_CG"]),
        "global_CL": _to_float(final_raw["global_CL"]),
        "global_CG_hat": _to_float(final_raw["global_CG_hat"]),
        "global_CL_hat": _to_float(final_raw["global_CL_hat"]),
        "beta": _to_float(final_raw["beta"]),
        "overlap_sq": _to_float(final_raw["overlap_sq"]),
        "global_residual_sq": final_residual_sq,
        "global_residual_norm": final_residual_norm,
        "sigma": final_sigma,
        "l2_abs_raw": final_l2.abs_error_raw,
        "l2_rel_raw": final_l2.rel_error_raw,
        "l2_abs_aligned": final_l2.abs_error_aligned,
        "l2_rel_aligned": final_l2.rel_error_aligned,
        "phase_angle_rad": final_l2.phase_angle_rad,
    }

    theta_final = np.asarray(theta, dtype=np.float64)
    x_norm_final = np.array(evaluator.state(theta), dtype=np.complex128)
    x_est_raw = float(final_sigma) * x_norm_final
    x_est_aligned = x_est_raw * np.exp(-1.0j * final_l2.phase_angle_rad)
    solution_txt_path = run_dir / "solution_comparison.txt"
    with solution_txt_path.open("w", encoding="utf-8") as f:
        f.write("# Final variational parameter theta\n")
        f.write(np.array2string(theta_final, precision=16))
        f.write("\n\n")
        f.write(f"# Final sigma\n{final_sigma:.18e}\n\n")
        for name, vec in [
            ("x_true", x_true),
            ("x_est_unnorm_raw", x_est_raw),
            ("x_est_unnorm_aligned", x_est_aligned),
        ]:
            arr = np.asarray(vec, dtype=np.complex128).reshape(-1)
            f.write(f"# {name}: idx real imag\n")
            for i, v in enumerate(arr):
                f.write(f"{i} {v.real:.18e} {v.imag:.18e}\n")
            f.write("\n")

    solution_artifacts = {
        "solution_txt": solution_txt_path,
    }
    solution_previews = {
        "x_true": _format_complex_vector_preview(x_true),
        "x_est_unnorm_raw": _format_complex_vector_preview(x_est_raw),
        "x_est_unnorm_aligned": _format_complex_vector_preview(x_est_aligned),
        "final_theta": np.array2string(theta_final, precision=10),
    }

    write_report(
        run_dir=run_dir,
        cfg=cfg,
        data=data,
        history=history,
        checkpoints=checkpoints,
        best=best,
        final_metrics=final_metrics,
        solution_artifacts=solution_artifacts,
        solution_previews=solution_previews,
    )
    append_residual_section(run_dir / "report.md", best=best, final_metrics=final_metrics)
    append_qjit_report_section(run_dir, backend_info=backend_info, cfg=cfg)

    with (run_dir / "runtime_backend.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "requested_qjit": backend_info.requested_qjit,
                "backend": backend_info.backend,
                "catalyst_available": backend_info.catalyst_available,
                "catalyst_error": backend_info.catalyst_error,
            },
            f,
            indent=2,
        )

    with (run_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    metrics_writer.close()
    logger.info("[done] report directory: %s", run_dir)


if __name__ == "__main__":
    main()
