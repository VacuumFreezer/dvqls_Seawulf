#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MPS_simulation.dist.quimb_dist_eq26_2x2_benchmark import (  # noqa: E402
    apply_block_mpo,
    build_circuit_numpy,
    build_partitioned_problem_numpy,
    distributed_iteration,
    global_cost_jax,
    global_cost_numpy,
    initialize_state,
    make_initial_parameters,
    to_jax_problem,
)
from MPS_simulation.quimb_dist_eq26_common import (  # noqa: E402
    JsonlWriter,
    add_mps,
    atomic_write_json,
    dump_yaml_config,
    encode_array,
    format_array_preview,
    sanitize_jsonable,
    scale_mps,
)


def _load_helper_module(module_name: str, filename: str):
    module_path = THIS_DIR / "30qubits" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load helper module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    apply_no_compress: bool
    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    iterations: int
    report_every: int
    init_mode: str
    init_seed: int
    init_start: float
    init_stop: float
    x_scale_init: float
    z_scale_init: float
    dense_reference_max_global_qubits: int
    preview_elements: int
    out_dir: str | None
    out_json: str | None
    out_report: str | None
    out_history: str | None
    out_checkpoint: str | None
    out_config: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the low-memory distributed Eq. (26) MPS optimizer with "
            "crash-safe JSONL metric history and inner-product residual checks."
        )
    )
    parser.add_argument("--global-qubits", type=int, default=30)
    parser.add_argument("--local-qubits", type=int, default=29)
    parser.add_argument("--j-coupling", type=float, default=0.1)
    parser.add_argument("--kappa", type=float, default=20.0)
    parser.add_argument("--row-self-loop-weight", type=float, default=1.0)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--gate-max-bond", type=int, default=32)
    parser.add_argument("--gate-cutoff", type=float, default=1.0e-10)
    parser.add_argument("--apply-max-bond", type=int, default=64)
    parser.add_argument("--apply-cutoff", type=float, default=1.0e-10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument(
        "--init-mode",
        type=str,
        choices=("structured_linspace", "random_uniform"),
        default="structured_linspace",
    )
    parser.add_argument("--init-seed", type=int, default=1234)
    parser.add_argument("--init-start", type=float, default=0.01)
    parser.add_argument("--init-stop", type=float, default=0.2)
    parser.add_argument("--x-scale-init", type=float, default=0.75)
    parser.add_argument("--z-scale-init", type=float, default=0.10)
    parser.add_argument("--dense-reference-max-global-qubits", type=int, default=11)
    parser.add_argument("--preview-elements", type=int, default=200)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument("--out-report", type=str, default=None)
    parser.add_argument("--out-history", type=str, default=None)
    parser.add_argument("--out-checkpoint", type=str, default=None)
    parser.add_argument("--out-config", type=str, default=None)
    return parser.parse_args(argv)


def make_config(args: argparse.Namespace) -> Config:
    return Config(
        global_qubits=int(args.global_qubits),
        local_qubits=int(args.local_qubits),
        j_coupling=float(args.j_coupling),
        kappa=float(args.kappa),
        row_self_loop_weight=float(args.row_self_loop_weight),
        layers=int(args.layers),
        gate_max_bond=int(args.gate_max_bond),
        gate_cutoff=float(args.gate_cutoff),
        apply_max_bond=int(args.apply_max_bond),
        apply_cutoff=float(args.apply_cutoff),
        apply_no_compress=True,
        learning_rate=float(args.learning_rate),
        adam_beta1=float(args.adam_beta1),
        adam_beta2=float(args.adam_beta2),
        adam_epsilon=float(args.adam_epsilon),
        iterations=int(args.iterations),
        report_every=int(args.report_every),
        init_mode=str(args.init_mode),
        init_seed=int(args.init_seed),
        init_start=float(args.init_start),
        init_stop=float(args.init_stop),
        x_scale_init=float(args.x_scale_init),
        z_scale_init=float(args.z_scale_init),
        dense_reference_max_global_qubits=int(args.dense_reference_max_global_qubits),
        preview_elements=int(args.preview_elements),
        out_dir=args.out_dir,
        out_json=args.out_json,
        out_report=args.out_report,
        out_history=args.out_history,
        out_checkpoint=args.out_checkpoint,
        out_config=args.out_config,
    )


def resolve_output_paths(cfg: Config) -> dict[str, Path]:
    if cfg.out_dir is not None:
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return {
            "json": Path(cfg.out_json) if cfg.out_json is not None else out_dir / "optimize.json",
            "report": Path(cfg.out_report) if cfg.out_report is not None else out_dir / "optimize_report.md",
            "history": Path(cfg.out_history) if cfg.out_history is not None else out_dir / "optimize_metrics.jsonl",
            "checkpoint": (
                Path(cfg.out_checkpoint)
                if cfg.out_checkpoint is not None
                else out_dir / "optimize_checkpoint.json"
            ),
            "config": Path(cfg.out_config) if cfg.out_config is not None else out_dir / "optimize_config_used.yaml",
        }

    base = THIS_DIR / (
        f"quimb_dist_eq26_2x2_optimize_lowmem_n{cfg.global_qubits}_local{cfg.local_qubits}"
        f"_k{int(cfg.kappa)}_nocompress_iter{cfg.iterations}"
    )
    return {
        "json": Path(cfg.out_json) if cfg.out_json is not None else base.with_suffix(".json"),
        "report": Path(cfg.out_report) if cfg.out_report is not None else base.with_name(base.name + "_report").with_suffix(".md"),
        "history": Path(cfg.out_history) if cfg.out_history is not None else base.with_name(base.name + "_metrics").with_suffix(".jsonl"),
        "checkpoint": (
            Path(cfg.out_checkpoint)
            if cfg.out_checkpoint is not None
            else base.with_name(base.name + "_checkpoint").with_suffix(".json")
        ),
        "config": Path(cfg.out_config) if cfg.out_config is not None else base.with_name(base.name + "_config_used").with_suffix(".yaml"),
    }


def ensure_output_dirs(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def build_problem_numpy(cfg: Config) -> tuple[dict[str, object], str]:
    if cfg.global_qubits == 30 and cfg.local_qubits == 29:
        if abs(cfg.j_coupling) <= 1.0e-15:
            direct30 = _load_helper_module(
                "dist_direct30_nocoupling",
                "quimb_dist_eq26_2x2_direct_mpo_30q_nocoupling.py",
            )
            return direct30.build_direct_problem(cfg), "direct_mpo_analytic_30q_nocoupling"
        direct30 = _load_helper_module(
            "dist_direct30",
            "quimb_dist_eq26_2x2_direct_mpo_smoke_30q.py",
        )
        return direct30.build_direct_problem(cfg), "direct_mpo_analytic_30q"
    return build_partitioned_problem_numpy(cfg), "dense_exact_block_split"


def build_global_sparse_problem(cfg: Config, eta: float, zeta: float):
    import quimb as qu
    import quimb.tensor as qtn

    builder = qtn.SpinHam1D(cyclic=False)
    builder += 1.0 / zeta, qu.pauli("X")
    builder += cfg.j_coupling / zeta, qu.pauli("Z"), qu.pauli("Z")
    h_sparse = builder.build_sparse(cfg.global_qubits).tocsr()
    return (h_sparse + (eta / zeta) * identity(h_sparse.shape[0], format="csr")).tocsr()


def exact_solution(a_sparse, n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    dim = 2**n_qubits
    b_dense = np.full(dim, 1.0 / math.sqrt(dim), dtype=np.complex128)
    x_true = spsolve(a_sparse, b_dense)
    return np.asarray(x_true, dtype=np.complex128), np.asarray(b_dense, dtype=np.complex128)


def build_dense_reference(cfg: Config, problem_np: dict[str, object]) -> dict[str, Any]:
    if cfg.global_qubits > cfg.dense_reference_max_global_qubits:
        return {
            "available": False,
            "reason": (
                "Dense exact validation is disabled because the global Hilbert-space dimension "
                f"2^{cfg.global_qubits} is too large."
            ),
        }

    a_sparse = build_global_sparse_problem(cfg, float(problem_np["eta"]), float(problem_np["zeta"]))
    a_dense = np.asarray(a_sparse.toarray(), dtype=np.complex128)
    x_true, b_dense = exact_solution(a_sparse, cfg.global_qubits)
    return {
        "available": True,
        "a_sparse": a_sparse,
        "a_dense": a_dense,
        "x_true": x_true,
        "b_dense": b_dense,
    }


def mps_norm_sq(state) -> float:
    return max(float(np.real(state.overlap(state))), 0.0)


def mps_norm_l2(state) -> float:
    return float(math.sqrt(mps_norm_sq(state)))


def mps_overlap_real(lhs, rhs) -> float:
    return float(np.real(lhs.overlap(rhs)))


def average_states(lhs, rhs, cfg: Config):
    return add_mps(
        scale_mps(lhs, 0.5),
        scale_mps(rhs, 0.5),
        max_bond=cfg.apply_max_bond,
        cutoff=cfg.apply_cutoff,
    )


def row_action_norm_sq(action_terms: list[Any]) -> float:
    total = sum(mps_norm_sq(term) for term in action_terms)
    for left in range(len(action_terms)):
        for right in range(left + 1, len(action_terms)):
            total += 2.0 * mps_overlap_real(action_terms[left], action_terms[right])
    return max(total, 0.0)


def residual_norm_from_terms(action_terms: list[Any], b_state, b_norm: float) -> float:
    cross = sum(mps_overlap_real(b_state, term) for term in action_terms)
    residual_sq = row_action_norm_sq(action_terms) - 2.0 * b_norm * cross + b_norm * b_norm
    return float(math.sqrt(max(residual_sq, 0.0)))


def state_gap_l2(lhs, rhs) -> float:
    gap_sq = mps_norm_sq(lhs) + mps_norm_sq(rhs) - 2.0 * mps_overlap_real(lhs, rhs)
    return float(math.sqrt(max(gap_sq, 0.0)))


def build_row_action_state(action_terms: list[Any], cfg: Config):
    combined = action_terms[0]
    for term in action_terms[1:]:
        combined = add_mps(
            combined,
            term,
            max_bond=cfg.apply_max_bond,
            cutoff=cfg.apply_cutoff,
        )
    return combined


def build_state_analysis(alpha: np.ndarray, cfg: Config, problem_np: dict[str, object]) -> dict[str, Any]:
    unit_states = [[build_circuit_numpy(cfg.local_qubits, alpha[i, j, 1:], cfg).psi for j in range(2)] for i in range(2)]
    block_states = [[scale_mps(unit_states[i][j], float(alpha[i, j, 0])) for j in range(2)] for i in range(2)]
    column_states = [average_states(block_states[0][j], block_states[1][j], cfg) for j in range(2)]

    row_action_terms = []
    row_action_residual_norms = []
    for i in range(2):
        terms = [apply_block_mpo(problem_np["blocks"][i][j], column_states[j], cfg) for j in range(2)]
        row_action_terms.append(terms)
        row_action_residual_norms.append(
            residual_norm_from_terms(
                terms,
                problem_np["b_row_states"][i],
                float(problem_np["b_row_norms"][i]),
            )
        )

    consensus_gaps = [state_gap_l2(block_states[0][j], block_states[1][j]) for j in range(2)]
    consensus_sq = sum(value * value for value in consensus_gaps)

    return {
        "unit_states": unit_states,
        "block_states": block_states,
        "column_states": column_states,
        "row_action_terms": row_action_terms,
        "x_unit_state_norms": [[mps_norm_l2(unit_states[i][j]) for j in range(2)] for i in range(2)],
        "x_block_norms": [[mps_norm_l2(block_states[i][j]) for j in range(2)] for i in range(2)],
        "column_block_norms": [mps_norm_l2(column_states[j]) for j in range(2)],
        "row_copy_norms": [
            float(math.sqrt(sum(mps_norm_sq(block_states[0][j]) for j in range(2)))),
            float(math.sqrt(sum(mps_norm_sq(block_states[1][j]) for j in range(2)))),
        ],
        "row_action_residual_norms": row_action_residual_norms,
        "residual_norm_l2": float(math.sqrt(sum(value * value for value in row_action_residual_norms))),
        "consensus_error_l2": float(math.sqrt(consensus_sq)),
        "row_variance_mse": float(consensus_sq / (2 ** (cfg.local_qubits + 1))),
    }


def compute_metrics(
    alpha: np.ndarray,
    beta: np.ndarray,
    cfg: Config,
    problem_np: dict[str, object],
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
    analysis: dict[str, Any],
    cfg: Config,
    reference: dict[str, Any],
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "x_unit_state_norms": analysis["x_unit_state_norms"],
        "x_block_norms": analysis["x_block_norms"],
        "column_block_norms": analysis["column_block_norms"],
        "row_copy_norms": analysis["row_copy_norms"],
        "row_action_residual_norms": analysis["row_action_residual_norms"],
        "dense_vectors_available": bool(reference["available"]),
    }

    if reference["available"]:
        dense_row_actions = [
            np.asarray(
                build_row_action_state(analysis["row_action_terms"][row], cfg).to_dense(),
                dtype=np.complex128,
            ).reshape(-1)
            for row in range(2)
        ]
        diagnostics["row_action_previews"] = [
            format_array_preview(item, max_elements=cfg.preview_elements)
            for item in dense_row_actions
        ]

    return diagnostics


def checkpoint_payload(
    *,
    iteration: int,
    metrics: dict[str, Any],
    alpha: np.ndarray,
    beta: np.ndarray,
    failed: bool = False,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "iteration": int(iteration),
        "latest_metrics": sanitize_jsonable(metrics),
        "alpha": encode_array(alpha),
        "beta": encode_array(beta),
        "failed": bool(failed),
        "error_message": error_message,
    }


def write_report(report_path: Path, result: dict[str, Any]) -> None:
    final = result["history"][-1]
    alpha_final = np.asarray(result["final_state"]["alpha_preview"], dtype=np.float64)
    beta_final = np.asarray(result["final_state"]["beta_preview"], dtype=np.float64)
    lines = [
        "# Low-Memory Distributed Eq. (26) Optimization Report",
        "",
        "## Setup",
        f"- Problem source: `{result['problem']['source']}`",
        f"- Global system: `{result['problem']['global_qubits']}` qubits",
        f"- Local block size: `{result['problem']['local_qubits']}` qubits",
        f"- Iterations: `{result['optimization']['iterations']}`",
        f"- Learning rate: `{result['config']['learning_rate']}`",
        f"- Initialization mode: `{result['initialization']['mode']}`",
        f"- No-compression MPO apply: `{result['config']['apply_no_compress']}`",
        f"- Dense reference available: `{result['problem']['dense_reference_available']}`",
        f"- Row Laplacian: `{result['problem']['row_laplacian']}`",
        f"- `eta = {result['problem']['eta']:.12g}`",
        f"- `zeta = {result['problem']['zeta']:.12g}`",
        "",
        "## Final Metrics",
        f"- Global cost: `{final['global_cost']:.12g}`",
        f"- Residual norm: `{final['residual_norm_l2']:.12g}`",
        f"- Consensus error: `{final['consensus_error_l2']:.12g}`",
        f"- Relative solution error: `{final['relative_solution_error_l2'] if final['relative_solution_error_l2'] is not None else 'n/a'}`",
        f"- Alpha gradient L2: `{final['alpha_grad_l2'] if final['alpha_grad_l2'] is not None else 'n/a'}`",
        f"- Beta gradient L2: `{final['beta_grad_l2'] if final['beta_grad_l2'] is not None else 'n/a'}`",
        f"- Total elapsed time: `{result['optimization']['elapsed_s']:.6f} s`",
        "",
        "## Crash-Safe Artifacts",
        f"- Live metrics JSONL: `{result['artifacts']['history']}`",
        f"- Last checkpoint JSON: `{result['artifacts']['checkpoint']}`",
        "",
        "## Final State Diagnostics",
        f"- Final `|| |X_ij> ||_2`: `{result['final_diagnostics']['x_unit_state_norms']}`",
        f"- Final `||x_ij||_2`: `{result['final_diagnostics']['x_block_norms']}`",
        f"- Final column-block norms: `{result['final_diagnostics']['column_block_norms']}`",
        f"- Final row-copy norms: `{result['final_diagnostics']['row_copy_norms']}`",
        f"- Row-action residual norms: `{result['final_diagnostics']['row_action_residual_norms']}`",
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
        if "row_action_previews" in result["final_diagnostics"]:
            lines.extend(
                [
                    "",
                    "## Row-Action Previews",
                    "### sum_j A_1j x_j",
                    "```text",
                    result["final_diagnostics"]["row_action_previews"][0],
                    "```",
                    "",
                    "### sum_j A_2j x_j",
                    "```text",
                    result["final_diagnostics"]["row_action_previews"][1],
                    "```",
                ]
            )
    else:
        lines.extend(
            [
                "",
                "## Dense Preview Artifacts",
                "- Dense vector/matrix previews were skipped because the full global system is too large to materialize.",
            ]
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            f"- JSON: `{result['artifacts']['json']}`",
            f"- Report: `{result['artifacts']['report']}`",
            f"- Config: `{result['artifacts']['config']}`",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def optimize(cfg: Config, artifact_paths: dict[str, Path]) -> dict[str, Any]:
    import jax

    jax.config.update("jax_enable_x64", True)

    problem_np, problem_source = build_problem_numpy(cfg)
    problem_jax = to_jax_problem(problem_np)
    reference = build_dense_reference(cfg, problem_np)

    cost_fn = lambda a, b: global_cost_jax(a, b, cfg, problem_jax)
    full_grad_fn = jax.value_and_grad(cost_fn, argnums=(0, 1))
    alpha_grad_fn = jax.grad(cost_fn, argnums=0)

    alpha_init_np, beta_init_np = make_initial_parameters(cfg)
    state = initialize_state(cfg, alpha_grad_fn, alpha_init_np, beta_init_np)

    history: list[dict[str, Any]] = []
    metrics_writer = JsonlWriter(artifact_paths["history"])
    t0 = time.perf_counter()

    try:
        initial_metrics, _ = compute_metrics(alpha_init_np, beta_init_np, cfg, problem_np, reference)
        _, (init_g_alpha, init_g_beta) = full_grad_fn(state["alpha"], state["beta"])
        initial_metrics.update(
            {
                "iteration": 0,
                "elapsed_s": 0.0,
                "alpha_grad_l2": float(np.linalg.norm(np.asarray(init_g_alpha))),
                "beta_grad_l2": float(np.linalg.norm(np.asarray(init_g_beta))),
                "alpha_step_l2": None,
                "beta_step_l2": None,
            }
        )
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
            f"rel_err={initial_metrics['relative_solution_error_l2'] if initial_metrics['relative_solution_error_l2'] is not None else 'n/a'}",
            flush=True,
        )

        for iteration in range(1, cfg.iterations + 1):
            state, diag = distributed_iteration(state, cfg, problem_jax, full_grad_fn, alpha_grad_fn)

            alpha_np = np.asarray(state["alpha"], dtype=np.float64)
            beta_np = np.asarray(state["beta"], dtype=np.float64)
            metrics, _ = compute_metrics(alpha_np, beta_np, cfg, problem_np, reference)
            metrics.update(
                {
                    "iteration": int(iteration),
                    "elapsed_s": float(time.perf_counter() - t0),
                    "alpha_grad_l2": float(diag["alpha_grad_l2"]),
                    "beta_grad_l2": float(diag["beta_grad_l2"]),
                    "alpha_step_l2": float(diag["alpha_step_l2"]),
                    "beta_step_l2": float(diag["beta_step_l2"]),
                }
            )
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
                    f"elapsed={metrics['elapsed_s']:.2f}s",
                    flush=True,
                )
    except Exception as exc:
        alpha_fail = np.asarray(state["alpha"], dtype=np.float64)
        beta_fail = np.asarray(state["beta"], dtype=np.float64)
        latest_metrics = history[-1] if history else {}
        atomic_write_json(
            artifact_paths["checkpoint"],
            checkpoint_payload(
                iteration=int(latest_metrics.get("iteration", 0)),
                metrics=latest_metrics,
                alpha=alpha_fail,
                beta=beta_fail,
                failed=True,
                error_message=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise
    finally:
        metrics_writer.close()

    total_elapsed = float(time.perf_counter() - t0)
    alpha_final = np.asarray(state["alpha"], dtype=np.float64)
    beta_final = np.asarray(state["beta"], dtype=np.float64)
    final_metrics, final_analysis = compute_metrics(alpha_final, beta_final, cfg, problem_np, reference)
    if history:
        history[-1] = {
            **history[-1],
            **final_metrics,
        }

    x_reconstructed = final_analysis.get("x_reconstructed_dense")
    x_true = reference["x_true"] if reference["available"] else None
    a_dense = reference["a_dense"] if reference["available"] else None
    b_dense = reference["b_dense"] if reference["available"] else None
    rescaled_diagnostics = compute_rescaling_diagnostics(x_reconstructed, x_true, a_dense, b_dense)
    final_diagnostics = build_final_diagnostics(final_analysis, cfg, reference)

    result: dict[str, Any] = {
        "config": asdict(cfg),
        "problem": {
            "source": problem_source,
            "global_qubits": cfg.global_qubits,
            "local_qubits": cfg.local_qubits,
            "eta": float(problem_np["eta"]),
            "zeta": float(problem_np["zeta"]),
            "row_laplacian": np.asarray(problem_np["row_laplacian"], dtype=np.float64).tolist(),
            "b_row_norms": np.asarray(problem_np["b_row_norms"], dtype=np.float64).tolist(),
            "b_agent_norms": np.asarray(problem_np["b_norms"], dtype=np.float64).tolist(),
            "b_column_split": (
                np.asarray(problem_np["column_split"], dtype=np.float64).tolist()
                if "column_split" in problem_np
                else None
            ),
            "block_formula": problem_np.get("block_formula"),
            "dense_reference_available": bool(reference["available"]),
            "dense_reference_reason": None if reference["available"] else reference["reason"],
        },
        "optimization": {
            "optimizer": "distributed_adam_gradient_tracking",
            "iterations": cfg.iterations,
            "elapsed_s": total_elapsed,
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
            "A_dense_shape": list(a_dense.shape),
            "b": encode_array(b_dense),
            "A_dense_preview": format_array_preview(a_dense, max_elements=cfg.preview_elements),
            "b_preview": format_array_preview(b_dense, max_elements=cfg.preview_elements),
        }

    return result


def main(argv: list[str] | None = None) -> None:
    cfg = make_config(parse_args(argv))
    artifact_paths = resolve_output_paths(cfg)
    ensure_output_dirs(artifact_paths)
    dump_yaml_config(artifact_paths["config"], asdict(cfg))
    result = optimize(cfg, artifact_paths)
    result["artifacts"] = {key: str(path.resolve()) for key, path in artifact_paths.items()}

    artifact_paths["json"].write_text(json.dumps(sanitize_jsonable(result), indent=2) + "\n", encoding="utf-8")
    write_report(artifact_paths["report"], result)

    print("\nFinal summary:")
    print(json.dumps(sanitize_jsonable(result["history"][-1]), indent=2))
    print(f"\nWrote JSON to {artifact_paths['json']}")
    print(f"Wrote report to {artifact_paths['report']}")
    print(f"Wrote live history to {artifact_paths['history']}")
    print(f"Wrote checkpoint to {artifact_paths['checkpoint']}")


if __name__ == "__main__":
    main()
