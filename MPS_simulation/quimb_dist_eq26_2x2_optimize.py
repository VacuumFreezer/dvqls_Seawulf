#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from quimb_dist_eq26_2x2_benchmark import (  # noqa: E402
    DEFAULT_CONFIG as BENCHMARK_DEFAULT_CONFIG,
    build_global_sparse_problem,
    build_partitioned_problem_numpy,
    distributed_iteration,
    exact_solution,
    global_cost_jax,
    global_cost_numpy,
    initialize_state,
    make_initial_parameters,
    to_jax_problem,
)
from quimb_dist_eq26_common import (  # noqa: E402
    DEFAULT_PARAM_PATH,
    JsonlWriter,
    add_mps,
    atomic_write_json,
    build_circuit_numpy,
    dump_yaml_config,
    encode_array,
    format_array_preview,
    merge_section_config,
    parse_int_sequence,
    resolve_qubit_layout,
    sanitize_jsonable,
    scale_mps,
    apply_mpo_to_mps,
)


@dataclass
class Config:
    global_qubits: int
    local_qubits: int
    j_coupling: float
    kappa: float
    row_self_loop_weight: float
    layers: int
    gate_max_bond: int
    gate_cutoff: float
    apply_max_bond: int
    apply_cutoff: float
    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    init_mode: str
    init_seed: int
    init_start: float
    init_stop: float
    x_scale_init: float
    z_scale_init: float
    exact_validation_max_global_qubits: int
    spectrum_bond_dims: list[int]
    spectrum_cutoff: float
    spectrum_tol: float
    spectrum_max_sweeps: int
    preview_elements: int
    iterations: int
    report_every: int
    out_json: str | None
    out_figure: str | None
    out_report: str | None
    out_history: str | None
    out_checkpoint: str | None
    out_config: str | None


DEFAULT_CONFIG: dict[str, Any] = {
    **BENCHMARK_DEFAULT_CONFIG,
    "learning_rate": 0.02,
    "iterations": 200,
    "report_every": 20,
    "out_figure": None,
    "out_history": None,
    "out_checkpoint": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the distributed 2x2 Eq. (26) MPS optimizer with YAML-configured "
            "hyperparameters and crash-safe live metric history."
        )
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_PARAM_PATH))
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--qubits-per-agent", type=int, default=None)
    parser.add_argument("--global-qubits", type=int, default=None)
    parser.add_argument("--local-qubits", type=int, default=None)
    parser.add_argument("--j-coupling", type=float, default=None)
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--row-self-loop-weight", type=float, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--gate-max-bond", type=int, default=None)
    parser.add_argument("--gate-cutoff", type=float, default=None)
    parser.add_argument("--apply-max-bond", type=int, default=None)
    parser.add_argument("--apply-cutoff", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--adam-beta1", type=float, default=None)
    parser.add_argument("--adam-beta2", type=float, default=None)
    parser.add_argument("--adam-epsilon", type=float, default=None)
    parser.add_argument(
        "--init-mode",
        type=str,
        choices=("random_uniform", "structured_linspace"),
        default=None,
    )
    parser.add_argument("--init-seed", type=int, default=None)
    parser.add_argument("--init-start", type=float, default=None)
    parser.add_argument("--init-stop", type=float, default=None)
    parser.add_argument("--x-scale-init", type=float, default=None)
    parser.add_argument("--z-scale-init", type=float, default=None)
    parser.add_argument("--exact-validation-max-global-qubits", type=int, default=None)
    parser.add_argument("--spectrum-bond-dims", type=str, default=None)
    parser.add_argument("--spectrum-cutoff", type=float, default=None)
    parser.add_argument("--spectrum-tol", type=float, default=None)
    parser.add_argument("--spectrum-max-sweeps", type=int, default=None)
    parser.add_argument("--preview-elements", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--report-every", type=int, default=None)
    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument("--out-figure", type=str, default=None)
    parser.add_argument("--out-report", type=str, default=None)
    parser.add_argument("--out-history", type=str, default=None)
    parser.add_argument("--out-checkpoint", type=str, default=None)
    parser.add_argument("--out-config", type=str, default=None)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> Config:
    merged = merge_section_config(DEFAULT_CONFIG, args.config, "optimize", args.case)
    merged = resolve_qubit_layout(
        merged,
        qubits_per_agent=args.qubits_per_agent,
        global_qubits=args.global_qubits,
        local_qubits=args.local_qubits,
    )

    scalar_fields = {
        "j_coupling": args.j_coupling,
        "kappa": args.kappa,
        "row_self_loop_weight": args.row_self_loop_weight,
        "layers": args.layers,
        "gate_max_bond": args.gate_max_bond,
        "gate_cutoff": args.gate_cutoff,
        "apply_max_bond": args.apply_max_bond,
        "apply_cutoff": args.apply_cutoff,
        "learning_rate": args.learning_rate,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_epsilon": args.adam_epsilon,
        "init_mode": args.init_mode,
        "init_seed": args.init_seed,
        "init_start": args.init_start,
        "init_stop": args.init_stop,
        "x_scale_init": args.x_scale_init,
        "z_scale_init": args.z_scale_init,
        "exact_validation_max_global_qubits": args.exact_validation_max_global_qubits,
        "spectrum_cutoff": args.spectrum_cutoff,
        "spectrum_tol": args.spectrum_tol,
        "spectrum_max_sweeps": args.spectrum_max_sweeps,
        "preview_elements": args.preview_elements,
        "iterations": args.iterations,
        "report_every": args.report_every,
        "out_json": args.out_json,
        "out_figure": args.out_figure,
        "out_report": args.out_report,
        "out_history": args.out_history,
        "out_checkpoint": args.out_checkpoint,
        "out_config": args.out_config,
    }
    for key, value in scalar_fields.items():
        if value is not None:
            merged[key] = value

    if args.spectrum_bond_dims is not None:
        merged["spectrum_bond_dims"] = parse_int_sequence(args.spectrum_bond_dims)
    else:
        merged["spectrum_bond_dims"] = parse_int_sequence(
            merged.get("spectrum_bond_dims"),
            default=[16, 32, 64],
        )

    return Config(
        global_qubits=int(merged["global_qubits"]),
        local_qubits=int(merged["local_qubits"]),
        j_coupling=float(merged["j_coupling"]),
        kappa=float(merged["kappa"]),
        row_self_loop_weight=float(merged["row_self_loop_weight"]),
        layers=int(merged["layers"]),
        gate_max_bond=int(merged["gate_max_bond"]),
        gate_cutoff=float(merged["gate_cutoff"]),
        apply_max_bond=int(merged["apply_max_bond"]),
        apply_cutoff=float(merged["apply_cutoff"]),
        learning_rate=float(merged["learning_rate"]),
        adam_beta1=float(merged["adam_beta1"]),
        adam_beta2=float(merged["adam_beta2"]),
        adam_epsilon=float(merged["adam_epsilon"]),
        init_mode=str(merged["init_mode"]),
        init_seed=int(merged["init_seed"]),
        init_start=float(merged["init_start"]),
        init_stop=float(merged["init_stop"]),
        x_scale_init=float(merged["x_scale_init"]),
        z_scale_init=float(merged["z_scale_init"]),
        exact_validation_max_global_qubits=int(merged["exact_validation_max_global_qubits"]),
        spectrum_bond_dims=[int(x) for x in merged["spectrum_bond_dims"]],
        spectrum_cutoff=float(merged["spectrum_cutoff"]),
        spectrum_tol=float(merged["spectrum_tol"]),
        spectrum_max_sweeps=int(merged["spectrum_max_sweeps"]),
        preview_elements=int(merged["preview_elements"]),
        iterations=int(merged["iterations"]),
        report_every=int(merged["report_every"]),
        out_json=merged.get("out_json"),
        out_figure=merged.get("out_figure"),
        out_report=merged.get("out_report"),
        out_history=merged.get("out_history"),
        out_checkpoint=merged.get("out_checkpoint"),
        out_config=merged.get("out_config"),
    )


def resolve_output_paths(cfg: Config) -> dict[str, Path]:
    base = (
        Path(cfg.out_json).with_suffix("")
        if cfg.out_json is not None
        else THIS_DIR / f"quimb_dist_eq26_2x2_optimize_n{cfg.global_qubits}_local{cfg.local_qubits}_iter{cfg.iterations}"
    )
    json_path = Path(cfg.out_json) if cfg.out_json is not None else base.with_suffix(".json")
    figure_path = (
        Path(cfg.out_figure)
        if cfg.out_figure is not None
        else base.with_name(base.name + "_history").with_suffix(".png")
    )
    report_path = (
        Path(cfg.out_report)
        if cfg.out_report is not None
        else base.with_name(base.name + "_report").with_suffix(".md")
    )
    history_path = (
        Path(cfg.out_history)
        if cfg.out_history is not None
        else base.with_name(base.name + "_metrics").with_suffix(".jsonl")
    )
    checkpoint_path = (
        Path(cfg.out_checkpoint)
        if cfg.out_checkpoint is not None
        else base.with_name(base.name + "_checkpoint").with_suffix(".json")
    )
    config_path = (
        Path(cfg.out_config)
        if cfg.out_config is not None
        else base.with_name(base.name + "_config_used").with_suffix(".yaml")
    )
    for path in (json_path, figure_path, report_path, history_path, checkpoint_path, config_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    return {
        "json": json_path,
        "figure": figure_path,
        "report": report_path,
        "history": history_path,
        "checkpoint": checkpoint_path,
        "config": config_path,
    }


def mps_norm_l2(state) -> float:
    return float(np.sqrt(max(float(np.real(state.overlap(state))), 0.0)))


def agent_unit_state(alpha_agent: np.ndarray, cfg: Config):
    return build_circuit_numpy(cfg.local_qubits, alpha_agent[1:], cfg).psi


def agent_block_state(alpha_agent: np.ndarray, cfg: Config):
    return scale_mps(agent_unit_state(alpha_agent, cfg), float(alpha_agent[0]))


def average_states(lhs, rhs, cfg: Config):
    return add_mps(
        scale_mps(lhs, 0.5),
        scale_mps(rhs, 0.5),
        max_bond=cfg.apply_max_bond,
        cutoff=cfg.apply_cutoff,
    )


def difference_state(lhs, rhs, cfg: Config):
    return add_mps(
        lhs,
        scale_mps(rhs, -1.0),
        max_bond=cfg.apply_max_bond,
        cutoff=cfg.apply_cutoff,
    )


def build_state_analysis(alpha: np.ndarray, cfg: Config, problem_np: dict[str, Any]) -> dict[str, Any]:
    unit_states = [[agent_unit_state(alpha[i, j], cfg) for j in range(2)] for i in range(2)]
    block_states = [[scale_mps(unit_states[i][j], float(alpha[i, j, 0])) for j in range(2)] for i in range(2)]

    x1 = average_states(block_states[0][0], block_states[1][0], cfg)
    x2 = average_states(block_states[0][1], block_states[1][1], cfg)

    row_action_1 = add_mps(
        apply_mpo_to_mps(problem_np["blocks"][0][0], x1, max_bond=cfg.apply_max_bond, cutoff=cfg.apply_cutoff),
        apply_mpo_to_mps(problem_np["blocks"][0][1], x2, max_bond=cfg.apply_max_bond, cutoff=cfg.apply_cutoff),
        max_bond=cfg.apply_max_bond,
        cutoff=cfg.apply_cutoff,
    )
    row_action_2 = add_mps(
        apply_mpo_to_mps(problem_np["blocks"][1][0], x1, max_bond=cfg.apply_max_bond, cutoff=cfg.apply_cutoff),
        apply_mpo_to_mps(problem_np["blocks"][1][1], x2, max_bond=cfg.apply_max_bond, cutoff=cfg.apply_cutoff),
        max_bond=cfg.apply_max_bond,
        cutoff=cfg.apply_cutoff,
    )

    row_residual_1 = add_mps(
        row_action_1,
        scale_mps(problem_np["b_row_states"][0], -problem_np["b_row_norms"][0]),
        max_bond=cfg.apply_max_bond,
        cutoff=cfg.apply_cutoff,
    )
    row_residual_2 = add_mps(
        row_action_2,
        scale_mps(problem_np["b_row_states"][1], -problem_np["b_row_norms"][1]),
        max_bond=cfg.apply_max_bond,
        cutoff=cfg.apply_cutoff,
    )

    gap_1 = difference_state(block_states[0][0], block_states[1][0], cfg)
    gap_2 = difference_state(block_states[0][1], block_states[1][1], cfg)

    row_action_residual_norms = [mps_norm_l2(row_residual_1), mps_norm_l2(row_residual_2)]
    consensus_sq = mps_norm_l2(gap_1) ** 2 + mps_norm_l2(gap_2) ** 2

    return {
        "unit_states": unit_states,
        "block_states": block_states,
        "column_states": [x1, x2],
        "row_actions": [row_action_1, row_action_2],
        "row_residuals": [row_residual_1, row_residual_2],
        "x_unit_state_norms": [
            [mps_norm_l2(unit_states[i][j]) for j in range(2)]
            for i in range(2)
        ],
        "x_block_norms": [
            [mps_norm_l2(block_states[i][j]) for j in range(2)]
            for i in range(2)
        ],
        "column_block_norms": [mps_norm_l2(x1), mps_norm_l2(x2)],
        "row_copy_norms": [
            float(np.sqrt(sum(mps_norm_l2(block_states[0][j]) ** 2 for j in range(2)))),
            float(np.sqrt(sum(mps_norm_l2(block_states[1][j]) ** 2 for j in range(2)))),
        ],
        "row_action_residual_norms": row_action_residual_norms,
        "residual_norm_l2": float(np.sqrt(sum(value * value for value in row_action_residual_norms))),
        "consensus_error_l2": float(np.sqrt(consensus_sq)),
        "row_variance_mse": float(consensus_sq / (2 ** (cfg.local_qubits + 1))),
    }


def build_dense_reference(cfg: Config, problem_np: dict[str, Any]) -> dict[str, Any]:
    if not problem_np["dense_reference_available"]:
        return {
            "available": False,
            "reason": (
                "Dense exact validation is disabled because the global Hilbert-space dimension "
                f"2^{cfg.global_qubits} is too large."
            ),
        }

    a_sparse = build_global_sparse_problem(cfg, problem_np["eta"], problem_np["zeta"])
    x_true, b_dense = exact_solution(a_sparse, cfg.global_qubits)
    return {
        "available": True,
        "a_sparse": a_sparse,
        "a_dense": np.asarray(a_sparse.toarray(), dtype=np.complex128),
        "x_true": np.asarray(x_true, dtype=np.complex128),
        "b_dense": np.asarray(b_dense, dtype=np.complex128),
    }


def compute_metrics(
    alpha: np.ndarray,
    beta: np.ndarray,
    cfg: Config,
    problem_np: dict[str, Any],
    reference: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    analysis = build_state_analysis(alpha, cfg, problem_np)
    metrics: dict[str, Any] = {
        "global_cost": float(global_cost_numpy(alpha, beta, cfg, problem_np)),
        "residual_norm_l2": analysis["residual_norm_l2"],
        "consensus_error_l2": analysis["consensus_error_l2"],
        "row_variance_mse": analysis["row_variance_mse"],
        "relative_solution_error_l2": None,
    }

    if reference["available"]:
        x1_dense = np.asarray(analysis["column_states"][0].to_dense(), dtype=np.complex128).reshape(-1)
        x2_dense = np.asarray(analysis["column_states"][1].to_dense(), dtype=np.complex128).reshape(-1)
        x_reconstructed = np.concatenate((x1_dense, x2_dense))
        x_true = np.asarray(reference["x_true"], dtype=np.complex128)
        metrics["relative_solution_error_l2"] = float(
            np.linalg.norm(x_reconstructed - x_true) / max(np.linalg.norm(x_true), 1.0e-12)
        )
        analysis["x_reconstructed_dense"] = x_reconstructed

    return metrics, analysis


def compute_rescaling_diagnostics(
    x_reconstructed: np.ndarray | None,
    x_true: np.ndarray | None,
    a_dense: np.ndarray | None,
    b_dense: np.ndarray | None,
) -> dict[str, Any]:
    if x_reconstructed is None or x_true is None or a_dense is None or b_dense is None:
        return {
            "available": False,
            "reason": "Dense reference quantities are unavailable for this run.",
        }

    denom = np.vdot(x_reconstructed, x_reconstructed)
    best_scale = 0.0 if abs(denom) <= 1.0e-15 else np.vdot(x_reconstructed, x_true) / denom
    x_rescaled = best_scale * x_reconstructed

    return {
        "available": True,
        "best_scale_to_true_real": float(np.real(best_scale)),
        "best_scale_to_true_imag": float(np.imag(best_scale)),
        "raw_x_norm_l2": float(np.linalg.norm(x_reconstructed)),
        "true_x_norm_l2": float(np.linalg.norm(x_true)),
        "rescaled_x_norm_l2": float(np.linalg.norm(x_rescaled)),
        "cosine_similarity_to_true": float(
            abs(np.vdot(x_reconstructed, x_true))
            / max(np.linalg.norm(x_reconstructed) * np.linalg.norm(x_true), 1.0e-15)
        ),
        "rescaled_relative_solution_error_l2": float(
            np.linalg.norm(x_rescaled - x_true) / max(np.linalg.norm(x_true), 1.0e-12)
        ),
        "rescaled_residual_norm_l2": float(np.linalg.norm(a_dense @ x_rescaled - b_dense)),
    }


def build_final_diagnostics(
    alpha: np.ndarray,
    cfg: Config,
    problem_np: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    analysis = build_state_analysis(alpha, cfg, problem_np)
    result: dict[str, Any] = {
        "x_unit_state_norms": analysis["x_unit_state_norms"],
        "x_block_norms": analysis["x_block_norms"],
        "column_block_norms": analysis["column_block_norms"],
        "row_copy_norms": analysis["row_copy_norms"],
        "row_action_residual_norms": analysis["row_action_residual_norms"],
        "dense_vectors_available": bool(reference["available"]),
    }

    if reference["available"]:
        dense_unit_vectors = [
            [np.asarray(analysis["unit_states"][i][j].to_dense(), dtype=np.complex128).reshape(-1) for j in range(2)]
            for i in range(2)
        ]
        dense_block_vectors = [
            [np.asarray(analysis["block_states"][i][j].to_dense(), dtype=np.complex128).reshape(-1) for j in range(2)]
            for i in range(2)
        ]
        dense_row_actions = [
            np.asarray(analysis["row_actions"][idx].to_dense(), dtype=np.complex128).reshape(-1)
            for idx in range(2)
        ]
        result.update(
            {
                "x_unit_vectors": [
                    [encode_array(dense_unit_vectors[i][j]) for j in range(2)]
                    for i in range(2)
                ],
                "x_block_vectors": [
                    [encode_array(dense_block_vectors[i][j]) for j in range(2)]
                    for i in range(2)
                ],
                "row_actions": [encode_array(item) for item in dense_row_actions],
            }
        )

    return result


def checkpoint_payload(
    *,
    iteration: int,
    metrics: dict[str, Any],
    alpha: np.ndarray,
    beta: np.ndarray,
) -> dict[str, Any]:
    return {
        "iteration": int(iteration),
        "latest_metrics": sanitize_jsonable(metrics),
        "alpha": encode_array(alpha),
        "beta": encode_array(beta),
    }


def plot_history(history: list[dict[str, Any]], figure_path: Path) -> None:
    iterations = np.asarray([int(item["iteration"]) for item in history], dtype=np.int64)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), dpi=160)
    series = [
        ("global_cost", "Global cost"),
        ("residual_norm_l2", "Residual norm ||Ax-b||_2"),
        ("consensus_error_l2", "Consensus error ||x_row1-x_row2||_2"),
        ("relative_solution_error_l2", "Relative solution error"),
    ]

    for ax, (key, title) in zip(axes.flat, series):
        values = np.asarray(
            [np.nan if item.get(key) is None else float(item[key]) for item in history],
            dtype=np.float64,
        )
        finite = np.isfinite(values) & (values > 0.0)
        if np.any(finite):
            ax.plot(iterations[finite], values[finite], linewidth=1.8, color="#005f73")
            ax.set_yscale("log")
        else:
            ax.text(0.5, 0.5, "Unavailable for this run", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.25, linewidth=0.8)

    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def write_report(report_path: Path, result: dict[str, Any]) -> None:
    final = result["history"][-1]
    alpha_final = np.asarray(result["final_state"]["alpha_preview"], dtype=np.float64)
    beta_final = np.asarray(result["final_state"]["beta_preview"], dtype=np.float64)
    lines = [
        "# Distributed Eq. (26) 2x2 Optimization Report",
        "",
        "## Setup",
        f"- Iterations: `{result['optimization']['iterations']}`",
        f"- Learning rate: `{result['config']['learning_rate']}`",
        f"- Initialization mode: `{result['initialization']['mode']}`",
        f"- Spectrum source: `{result['problem']['spectrum_method']}`",
        f"- Dense exact reference available: `{result['problem']['dense_reference_available']}`",
        f"- Scaled spectrum check: `lambda_min={result['problem']['scaled_lambda_min']:.12g}`, `lambda_max={result['problem']['scaled_lambda_max']:.12g}`",
        f"- Row Laplacian: `{result['problem']['row_laplacian']}`",
        f"- Row-block norms of `b_i`: `{result['problem']['b_row_norms']}`",
        f"- Agent-block norms of `b_ij`: `{result['problem']['b_agent_norms']}`",
        f"- Column split used for `b_ij`: `{result['problem']['b_column_split']}`",
        "",
        "## Final Metrics",
        f"- Global cost: `{final['global_cost']:.12g}`",
        f"- Residual norm: `{final['residual_norm_l2']:.12g}`",
        f"- Consensus error: `{final['consensus_error_l2']:.12g}`",
        f"- Relative solution error: `{final['relative_solution_error_l2'] if final['relative_solution_error_l2'] is not None else 'n/a'}`",
        f"- Elapsed time: `{result['optimization']['elapsed_s']:.6f} s`",
        "",
        "## Crash-Safe Artifacts",
        f"- Live metrics JSONL: `{result['artifacts']['history']}`",
        f"- Last checkpoint JSON: `{result['artifacts']['checkpoint']}`",
        "",
        "## State Reconstruction Checks",
        f"- Final `|| |X_ij> ||_2`: `{result['final_diagnostics']['x_unit_state_norms']}`",
        f"- Final `||x_ij||_2`: `{result['final_diagnostics']['x_block_norms']}`",
        f"- Final column-block norms `[||x_1||_2, ||x_2||_2]`: `{result['final_diagnostics']['column_block_norms']}`",
        f"- Final row-copy norms: `{result['final_diagnostics']['row_copy_norms']}`",
        f"- Row-action residual norms `[||sum_j A_1j x_j - b_1||_2, ||sum_j A_2j x_j - b_2||_2]`: `{result['final_diagnostics']['row_action_residual_norms']}`",
        "",
        "## Final Parameter Previews",
        "### alpha",
        "```text",
        format_array_preview(alpha_final, max_elements=result["config"]["preview_elements"]),
        "```",
        "",
        "### beta",
        "```text",
        format_array_preview(beta_final, max_elements=result["config"]["preview_elements"]),
        "```",
    ]

    if result["rescaled_diagnostics"]["available"]:
        rescaled = result["rescaled_diagnostics"]
        lines.extend(
            [
                "",
                "## Rescaled-x Diagnostic",
                f"- Best scalar to match true `x`: `{rescaled['best_scale_to_true_real']:.12g}{rescaled['best_scale_to_true_imag']:+.12g}j`",
                f"- Raw reconstructed `||x||_2`: `{rescaled['raw_x_norm_l2']:.12g}`",
                f"- True `||x_true||_2`: `{rescaled['true_x_norm_l2']:.12g}`",
                f"- Rescaled `||x||_2`: `{rescaled['rescaled_x_norm_l2']:.12g}`",
                f"- Cosine similarity to true `x`: `{rescaled['cosine_similarity_to_true']:.12g}`",
                f"- Rescaled relative solution error: `{rescaled['rescaled_relative_solution_error_l2']:.12g}`",
                f"- Rescaled residual norm: `{rescaled['rescaled_residual_norm_l2']:.12g}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Rescaled-x Diagnostic",
                f"- {result['rescaled_diagnostics']['reason']}",
            ]
        )

    if result["problem"]["dense_reference_available"]:
        lines.extend(
            [
                "",
                "## Dense Preview Artifacts",
                "### Final reconstructed x",
                "```text",
                result["final_state"]["x_reconstructed_preview"],
                "```",
                "",
                "### True solution x",
                "```text",
                result["final_state"]["x_true_preview"],
                "```",
                "",
                "### True matrix A",
                "```text",
                result["linear_system"]["A_dense_preview"],
                "```",
                "",
                "### True vector b",
                "```text",
                result["linear_system"]["b_preview"],
                "```",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Dense Preview Artifacts",
                "- Dense vector/matrix previews were skipped for this run because the full global system is too large to materialize.",
            ]
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            f"- JSON: `{result['artifacts']['json']}`",
            f"- Figure: `{result['artifacts']['figure']}`",
            f"- Report: `{result['artifacts']['report']}`",
            f"- Config: `{result['artifacts']['config']}`",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def optimize(cfg: Config, artifact_paths: dict[str, Path]) -> dict[str, Any]:
    import jax

    jax.config.update("jax_enable_x64", True)

    problem_np = build_partitioned_problem_numpy(cfg)
    problem_jax = to_jax_problem(problem_np)
    reference = build_dense_reference(cfg, problem_np)

    cost_fn = lambda a, b: global_cost_jax(a, b, cfg, problem_jax)
    full_grad_fn = jax.value_and_grad(cost_fn, argnums=(0, 1))
    alpha_grad_fn = jax.grad(cost_fn, argnums=0)

    alpha_init_np, beta_init_np = make_initial_parameters(cfg)
    state = initialize_state(cfg, alpha_grad_fn, alpha_init_np, beta_init_np)

    history: list[dict[str, Any]] = []
    metrics_writer = JsonlWriter(artifact_paths["history"])

    try:
        initial_metrics, _ = compute_metrics(alpha_init_np, beta_init_np, cfg, problem_np, reference)
        initial_metrics["iteration"] = 0
        initial_metrics["elapsed_s"] = 0.0
        history.append(initial_metrics)
        metrics_writer.write(initial_metrics)
        atomic_write_json(
            artifact_paths["checkpoint"],
            checkpoint_payload(
                iteration=0,
                metrics=initial_metrics,
                alpha=alpha_init_np,
                beta=beta_init_np,
            ),
        )

        print(
            "[iter=0] "
            f"cost={initial_metrics['global_cost']:.12f} "
            f"residual={initial_metrics['residual_norm_l2']:.12f} "
            f"consensus={initial_metrics['consensus_error_l2']:.12f} "
            f"rel_err={initial_metrics['relative_solution_error_l2'] if initial_metrics['relative_solution_error_l2'] is not None else 'n/a'}"
        )

        t0 = time.perf_counter()
        for iteration in range(1, cfg.iterations + 1):
            state, _ = distributed_iteration(state, cfg, problem_jax, full_grad_fn, alpha_grad_fn)

            alpha_np = np.asarray(state["alpha"], dtype=np.float64)
            beta_np = np.asarray(state["beta"], dtype=np.float64)
            metrics, _ = compute_metrics(alpha_np, beta_np, cfg, problem_np, reference)
            metrics["iteration"] = int(iteration)
            metrics["elapsed_s"] = float(time.perf_counter() - t0)
            history.append(metrics)
            metrics_writer.write(metrics)
            atomic_write_json(
                artifact_paths["checkpoint"],
                checkpoint_payload(
                    iteration=iteration,
                    metrics=metrics,
                    alpha=alpha_np,
                    beta=beta_np,
                ),
            )

            if (iteration % cfg.report_every == 0) or (iteration == cfg.iterations):
                print(
                    f"[iter={iteration}] "
                    f"cost={metrics['global_cost']:.12f} "
                    f"residual={metrics['residual_norm_l2']:.12f} "
                    f"consensus={metrics['consensus_error_l2']:.12f} "
                    f"rel_err={metrics['relative_solution_error_l2'] if metrics['relative_solution_error_l2'] is not None else 'n/a'} "
                    f"elapsed={metrics['elapsed_s']:.2f}s"
                )

        total_elapsed = time.perf_counter() - t0
    finally:
        metrics_writer.close()

    alpha_final = np.asarray(state["alpha"], dtype=np.float64)
    beta_final = np.asarray(state["beta"], dtype=np.float64)
    final_metrics, final_analysis = compute_metrics(alpha_final, beta_final, cfg, problem_np, reference)
    final_diagnostics = build_final_diagnostics(alpha_final, cfg, problem_np, reference)

    x_reconstructed = final_analysis.get("x_reconstructed_dense")
    x_true = reference["x_true"] if reference["available"] else None
    a_dense = reference["a_dense"] if reference["available"] else None
    b_dense = reference["b_dense"] if reference["available"] else None
    rescaled_diagnostics = compute_rescaling_diagnostics(x_reconstructed, x_true, a_dense, b_dense)

    result: dict[str, Any] = {
        "config": asdict(cfg),
        "problem": {
            "lambda_min": float(problem_np["lambda_min"]),
            "lambda_max": float(problem_np["lambda_max"]),
            "eta": float(problem_np["eta"]),
            "zeta": float(problem_np["zeta"]),
            "scaled_lambda_min": float(problem_np["scaled_lambda_min"]),
            "scaled_lambda_max": float(problem_np["scaled_lambda_max"]),
            "row_laplacian": problem_np["row_laplacian"].tolist(),
            "b_row_norms": list(problem_np["b_row_norms"]),
            "b_agent_norms": np.asarray(problem_np["b_norms"], dtype=np.float64).tolist(),
            "b_column_split": np.asarray(problem_np["column_split"], dtype=np.float64).tolist(),
            "spectrum_method": str(problem_np["spectrum_method"]),
            "dense_reference_available": bool(reference["available"]),
            "dense_reference_reason": None if reference["available"] else reference["reason"],
        },
        "optimization": {
            "optimizer": "distributed_adam_gradient_tracking",
            "iterations": cfg.iterations,
            "elapsed_s": float(total_elapsed),
        },
        "initialization": {
            "mode": cfg.init_mode,
            "sigma_init": cfg.x_scale_init,
            "lambda_init": cfg.z_scale_init,
            "angle_range": [cfg.init_start, cfg.init_stop],
        },
        "rescaled_diagnostics": rescaled_diagnostics,
        "final_diagnostics": final_diagnostics,
        "final_state": {
            "alpha": encode_array(alpha_final),
            "beta": encode_array(beta_final),
            "alpha_preview": alpha_final.tolist(),
            "beta_preview": beta_final.tolist(),
        },
        "linear_system": {},
        "history": history,
    }
    if cfg.init_mode == "random_uniform":
        result["initialization"]["seed"] = cfg.init_seed

    if reference["available"]:
        result["final_state"].update(
            {
                "x_reconstructed": encode_array(x_reconstructed),
                "x_true": encode_array(x_true),
                "x_reconstructed_preview": format_array_preview(
                    x_reconstructed,
                    max_elements=cfg.preview_elements,
                ),
                "x_true_preview": format_array_preview(
                    x_true,
                    max_elements=cfg.preview_elements,
                ),
            }
        )
        result["linear_system"] = {
            "A_dense": encode_array(a_dense),
            "b": encode_array(b_dense),
            "A_dense_preview": format_array_preview(a_dense, max_elements=cfg.preview_elements),
            "b_preview": format_array_preview(b_dense, max_elements=cfg.preview_elements),
        }

    history[-1] = {**history[-1], **final_metrics}
    return result


def main() -> None:
    cfg = make_config(parse_args())
    artifact_paths = resolve_output_paths(cfg)
    dump_yaml_config(artifact_paths["config"], asdict(cfg))
    result = optimize(cfg, artifact_paths)
    result["artifacts"] = {key: str(path.resolve()) for key, path in artifact_paths.items()}

    artifact_paths["json"].write_text(json.dumps(sanitize_jsonable(result), indent=2), encoding="utf-8")
    plot_history(result["history"], artifact_paths["figure"])
    write_report(artifact_paths["report"], result)

    print("\nFinal summary:")
    print(json.dumps(sanitize_jsonable(result["history"][-1]), indent=2))
    print(f"\nWrote JSON to {artifact_paths['json']}")
    print(f"Wrote figure to {artifact_paths['figure']}")
    print(f"Wrote report to {artifact_paths['report']}")
    print(f"Wrote live history to {artifact_paths['history']}")
    print(f"Wrote checkpoint to {artifact_paths['checkpoint']}")


if __name__ == "__main__":
    main()
