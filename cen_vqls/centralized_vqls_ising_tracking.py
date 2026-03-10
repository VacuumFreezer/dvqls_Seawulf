#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

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
        CentralizedIsingData,
        _apply_controlled_pauli_word,
        _global_ansatz,
        _multiply_pauli_words,
        _to_float,
        _word_with_single_pauli,
        _format_complex_vector_preview,
        compute_l2_error_unnormalized,
        load_centralized_data,
        write_report,
    )
except ImportError:
    from centralized_vqls_ising import (
        CentralizedIsingData,
        _apply_controlled_pauli_word,
        _global_ansatz,
        _multiply_pauli_words,
        _to_float,
        _word_with_single_pauli,
        _format_complex_vector_preview,
        compute_l2_error_unnormalized,
        load_centralized_data,
        write_report,
    )


PauliWord = Tuple[str, ...]


DEFAULT_CONFIG = {
    "problem": {
        "static_ops_path": "problems/static_ops_3net_Ising.py",
        "system_key": "4x4",
        "prefer_centralized_problem": True,
        "centralized_problem_key": "centralized",
        "consistency_system_key": "4x4",
        "consistency_atol": 1.0e-12,
        "b_consistency_atol": 1.0e-12,
        "b_state_tolerance": 1.0e-10,
    },
    "ansatz": {
        "layers": 7,
        "init_low": -3.141592653589793,
        "init_high": 3.141592653589793,
    },
    "optimization": {
        "steps": 200,
        "learning_rate": 0.03,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1.0e-8,
        "print_every": 20,
        "optimize_metric": "global_CG",
    },
    "runtime": {
        "seed": 0,
        "device": "lightning.qubit",
        "interface": "jax",
        "diff_method": "adjoint",
        "use_qjit": True,
        "fallback_to_jax_jit": True,
        "qjit_autograph": False,
        # distributed-style: compile loss/grad wrappers, not the whole train step
        "grad_mode": "qjit",  # qjit | jax_jit | jax
        # qnode compile policy: none is safest for memory, especially >=8 qubits
        "qnode_compile": "none",  # none | jax_jit | qjit
        "jit_training_step": False,
        "warmup_compile": True,
        "log_local_cost": False,
    },
    "report": {
        "out_dir": "cen_vqls/reports",
        "tag": "centralized_vqls_ising_tracking",
    },
}


@dataclass
class RuntimeCompileInfo:
    requested_grad_mode: str
    resolved_grad_mode: str
    requested_qnode_compile: str
    resolved_qnode_compile: str
    catalyst_available: bool
    catalyst_error: str | None


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


def dump_config(path: Path, cfg: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


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
    if args.grad_mode is not None:
        cfg["runtime"]["grad_mode"] = str(args.grad_mode)
    if args.qnode_compile is not None:
        cfg["runtime"]["qnode_compile"] = str(args.qnode_compile)

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


def resolve_compile_info(cfg: dict) -> RuntimeCompileInfo:
    runtime = cfg["runtime"]
    fallback_to_jax = bool(runtime.get("fallback_to_jax_jit", True))
    use_qjit_flag = bool(runtime.get("use_qjit", False))

    requested_grad = str(runtime.get("grad_mode", "jax_jit")).strip().lower()
    requested_qnode = str(runtime.get("qnode_compile", "none")).strip().lower()
    valid_grad = {"qjit", "jax_jit", "jax"}
    valid_qnode = {"qjit", "jax_jit", "none"}
    if requested_grad not in valid_grad:
        raise ValueError(f"runtime.grad_mode must be one of {sorted(valid_grad)}")
    if requested_qnode not in valid_qnode:
        raise ValueError(f"runtime.qnode_compile must be one of {sorted(valid_qnode)}")

    if not use_qjit_flag:
        if requested_grad == "qjit":
            requested_grad = "jax_jit"
        if requested_qnode == "qjit":
            requested_qnode = "none"
        return RuntimeCompileInfo(
            requested_grad_mode=str(runtime.get("grad_mode", "jax_jit")).strip().lower(),
            resolved_grad_mode=requested_grad,
            requested_qnode_compile=str(runtime.get("qnode_compile", "none")).strip().lower(),
            resolved_qnode_compile=requested_qnode,
            catalyst_available=False,
            catalyst_error=None,
        )

    catalyst_available = False
    catalyst_error = None
    try:
        import catalyst  # noqa: F401

        catalyst_available = True
    except Exception as exc:  # pragma: no cover - env dependent
        catalyst_error = str(exc)

    resolved_grad = requested_grad
    resolved_qnode = requested_qnode
    if (requested_grad == "qjit" or requested_qnode == "qjit") and not catalyst_available:
        if not fallback_to_jax:
            raise RuntimeError(
                "qjit mode requested but Catalyst import failed. "
                "Set runtime.fallback_to_jax_jit=true to continue with fallback. "
                f"Original error: {catalyst_error}"
            )
        if requested_grad == "qjit":
            resolved_grad = "jax_jit"
        if requested_qnode == "qjit":
            resolved_qnode = "none"

    return RuntimeCompileInfo(
        requested_grad_mode=requested_grad,
        resolved_grad_mode=resolved_grad,
        requested_qnode_compile=requested_qnode,
        resolved_qnode_compile=resolved_qnode,
        catalyst_available=catalyst_available,
        catalyst_error=catalyst_error,
    )


class QJITHadamardCentralizedVQLS:
    def __init__(
        self,
        data: CentralizedIsingData,
        device_name: str,
        diff_method: str,
        qnode_compile: str,
        qjit_autograph: bool,
        enable_mu_terms: bool,
    ):
        self.data = data
        self.n = data.n_total_qubits
        self.terms = data.terms
        self.identity_word = tuple("I" for _ in range(self.n))
        self.diff_method = str(diff_method)
        self.qnode_compile = str(qnode_compile)
        self.qjit_autograph = bool(qjit_autograph)
        self.enable_mu_terms = bool(enable_mu_terms)

        self.control_wire = 0
        self.system_wires = tuple(range(1, self.n + 1))
        self.dev_h = qml.device(device_name, wires=self.n + 1)
        self.dev_state = qml.device(device_name, wires=self.n)

        self.b_mode = data.b_unitary_info.mode
        self.b_unitary_dag = np.conjugate(data.b_unitary_info.unitary.T)

        self._build_linear_combinations()
        self._build_compiled_functions()

    def _compile_fn(self, fn):
        if self.qnode_compile == "qjit":
            try:
                return qml.qjit(fn, autograph=self.qjit_autograph)
            except TypeError:
                return qml.qjit(fn)
        if self.qnode_compile == "jax_jit":
            return jax.jit(fn)
        return fn

    def _apply_b_dagger_controlled(self) -> None:
        if self.b_mode == "hadamard_tensor":
            for wire in self.system_wires:
                qml.ctrl(qml.Hadamard, control=self.control_wire)(wires=wire)
            return

        def _b_dag():
            qml.QubitUnitary(self.b_unitary_dag, wires=self.system_wires)

        qml.ctrl(_b_dag, control=self.control_wire)()

    def _build_linear_combinations(self) -> None:
        self.beta_word_weights: Dict[PauliWord, complex] = {}
        if self.enable_mu_terms:
            self.mu_word_weights_by_wire: List[Dict[PauliWord, complex]] = [
                {} for _ in range(self.n)
            ]
        else:
            self.mu_word_weights_by_wire = []

        for c_l, w_l in self.terms:
            for c_lp, w_lp in self.terms:
                pair_coeff = complex(c_l) * complex(c_lp)

                phase_beta, w_beta = _multiply_pauli_words(w_l, w_lp)
                self.beta_word_weights[w_beta] = (
                    self.beta_word_weights.get(w_beta, 0.0 + 0.0j) + pair_coeff * phase_beta
                )

                if self.enable_mu_terms:
                    for wire in range(self.n):
                        x_word = _word_with_single_pauli(self.n, wire, "X")
                        phase_1, tmp = _multiply_pauli_words(w_l, x_word)
                        phase_2, w_full = _multiply_pauli_words(tmp, w_lp)
                        row = self.mu_word_weights_by_wire[wire]
                        row[w_full] = row.get(w_full, 0.0 + 0.0j) + pair_coeff * phase_1 * phase_2

        all_words = set(self.beta_word_weights.keys())
        if self.enable_mu_terms:
            for row in self.mu_word_weights_by_wire:
                all_words.update(row.keys())
        self.all_words = sorted(all_words)

    def _make_expect_word_fn(self, pauli_word: PauliWord):
        @qml.qnode(self.dev_h, interface="jax", diff_method=self.diff_method)
        def qnode(weights: jnp.ndarray):
            qml.Hadamard(wires=self.control_wire)
            _global_ansatz(weights, wires=self.system_wires)
            _apply_controlled_pauli_word(pauli_word, self.control_wire, self.system_wires)
            qml.Hadamard(wires=self.control_wire)
            return qml.expval(qml.PauliZ(self.control_wire))

        return self._compile_fn(qnode)

    def _make_gamma_real_fn(self, pauli_word: PauliWord):
        @qml.qnode(self.dev_h, interface="jax", diff_method=self.diff_method)
        def qnode(weights: jnp.ndarray):
            qml.Hadamard(wires=self.control_wire)
            qml.ctrl(_global_ansatz, control=self.control_wire)(weights, self.system_wires)
            _apply_controlled_pauli_word(pauli_word, self.control_wire, self.system_wires)
            self._apply_b_dagger_controlled()
            qml.Hadamard(wires=self.control_wire)
            return qml.expval(qml.PauliZ(self.control_wire))

        return self._compile_fn(qnode)

    def _make_gamma_imag_fn(self, pauli_word: PauliWord):
        @qml.qnode(self.dev_h, interface="jax", diff_method=self.diff_method)
        def qnode(weights: jnp.ndarray):
            qml.Hadamard(wires=self.control_wire)
            qml.ctrl(_global_ansatz, control=self.control_wire)(weights, self.system_wires)
            _apply_controlled_pauli_word(pauli_word, self.control_wire, self.system_wires)
            self._apply_b_dagger_controlled()
            qml.adjoint(qml.S)(wires=self.control_wire)
            qml.Hadamard(wires=self.control_wire)
            return qml.expval(qml.PauliZ(self.control_wire))

        return self._compile_fn(qnode)

    def _make_state_fn(self):
        # State readout is only for reporting (L2/residual), not for gradients.
        # Keep diff_method disabled so runtime "adjoint" does not break qml.state()
        # on lightning.qubit.
        @qml.qnode(self.dev_state, interface="jax", diff_method=None)
        def qnode(weights: jnp.ndarray):
            _global_ansatz(weights, wires=range(self.n))
            return qml.state()

        return self._compile_fn(qnode)

    def _build_compiled_functions(self) -> None:
        self._expect_word_fn: Dict[PauliWord, callable] = {}
        for word in self.all_words:
            if word == self.identity_word:
                continue
            self._expect_word_fn[word] = self._make_expect_word_fn(word)

        self._gamma_real_fn: List[callable] = []
        self._gamma_imag_fn: List[callable] = []
        for _, word in self.terms:
            self._gamma_real_fn.append(self._make_gamma_real_fn(word))
            self._gamma_imag_fn.append(self._make_gamma_imag_fn(word))

        self._state_fn = self._make_state_fn()

    def state(self, weights: jnp.ndarray):
        return self._state_fn(weights)

    def _expect_word_on_x(self, weights: jnp.ndarray, word: PauliWord):
        if word == self.identity_word:
            return jnp.asarray(1.0, dtype=jnp.float64)
        return self._expect_word_fn[word](weights)

    def metrics(self, weights: jnp.ndarray, include_local_cost: bool | None = None) -> Dict[str, jnp.ndarray]:
        if include_local_cost is None:
            include_local_cost = self.enable_mu_terms
        include_local_cost = bool(include_local_cost)

        mu_cache: Dict[PauliWord, jnp.ndarray] = {}
        for word in self.all_words:
            mu_cache[word] = self._expect_word_on_x(weights, word)

        beta = jnp.asarray(0.0, dtype=jnp.float64)
        for word, alpha in self.beta_word_weights.items():
            beta = beta + float(np.real(alpha)) * mu_cache[word]

        if include_local_cost:
            if not self.enable_mu_terms:
                raise RuntimeError(
                    "Local-cost metrics requested but mu terms were not built. "
                    "Set enable_mu_terms=True."
                )
            plus_sum = jnp.asarray(0.0, dtype=jnp.float64)
            for wire in range(self.n):
                mu_j = jnp.asarray(0.0, dtype=jnp.float64)
                for word, alpha in self.mu_word_weights_by_wire[wire].items():
                    mu_j = mu_j + float(np.real(alpha)) * mu_cache[word]
                plus_sum = plus_sum + 0.5 * (beta + mu_j)
        else:
            plus_sum = jnp.asarray(np.nan, dtype=jnp.float64)

        gamma_re = jnp.asarray(0.0, dtype=jnp.float64)
        gamma_im = jnp.asarray(0.0, dtype=jnp.float64)
        for idx, (coeff, _) in enumerate(self.terms):
            gamma_re = gamma_re + coeff * self._gamma_real_fn[idx](weights)
            gamma_im = gamma_im + coeff * self._gamma_imag_fn[idx](weights)

        overlap_sq = gamma_re * gamma_re + gamma_im * gamma_im

        cghat = beta - overlap_sq
        clhat = 0.5 * beta - plus_sum / (2.0 * self.n) if include_local_cost else jnp.asarray(np.nan, dtype=jnp.float64)

        beta_safe = beta + 1.0e-14
        cg = 1.0 - overlap_sq / beta_safe
        cl = (
            0.5 - plus_sum / (2.0 * self.n * beta_safe)
            if include_local_cost
            else jnp.asarray(np.nan, dtype=jnp.float64)
        )

        return {
            "beta": beta,
            "plus_sum": plus_sum,
            "gamma_re": gamma_re,
            "gamma_im": gamma_im,
            "overlap_sq": overlap_sq,
            "global_CG": cg,
            "global_CL": cl,
            "global_CG_hat": cghat,
            "global_CL_hat": clhat,
        }

    def objective(self, weights: jnp.ndarray, metric_name: str):
        need_local = metric_name in {"global_CL", "global_CL_hat"}
        vals = self.metrics(weights, include_local_cost=need_local)
        if metric_name not in vals:
            raise ValueError(f"Unknown optimize metric: {metric_name}")
        return vals[metric_name]


def append_compile_report_section(run_dir: Path, compile_info: RuntimeCompileInfo, cfg: dict) -> None:
    report_path = run_dir / "report.md"
    lines = []
    lines.append("")
    lines.append("## Compile Strategy")
    lines.append(f"- Requested grad mode: {compile_info.requested_grad_mode}")
    lines.append(f"- Resolved grad mode: {compile_info.resolved_grad_mode}")
    lines.append(f"- Requested qnode compile: {compile_info.requested_qnode_compile}")
    lines.append(f"- Resolved qnode compile: {compile_info.resolved_qnode_compile}")
    lines.append(f"- Catalyst available: {compile_info.catalyst_available}")
    if compile_info.catalyst_error:
        lines.append("- Catalyst import error:")
        lines.append(f"  - `{compile_info.catalyst_error}`")
    lines.append(f"- qjit_autograph: {bool(cfg['runtime'].get('qjit_autograph', False))}")
    lines.append(f"- jit_training_step: {bool(cfg['runtime'].get('jit_training_step', True))}")
    lines.append(f"- log_local_cost: {bool(cfg['runtime'].get('log_local_cost', False))}")

    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Centralized VQLS (distributed-style compile flow: compute_loss/compute_grad wrappers)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().with_name("config.yaml")),
        help="Path to YAML config file.",
    )
    # CLI overrides for task-array generation (distributed-style usage).
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
    parser.add_argument("--grad_mode", type=str, default=None, help="qjit | jax_jit | jax")
    parser.add_argument("--qnode_compile", type=str, default=None, help="none | jax_jit | qjit")
    parser.add_argument("--fallback_to_jax_jit", type=str, default=None)
    parser.add_argument("--out", type=str, default=None, help="Override report.out_dir")
    parser.add_argument("--tag", type=str, default=None, help="Override report.tag")
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)

    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(Path(args.config).resolve())
    cfg = apply_cli_overrides(cfg, args, repo_root=repo_root)

    compile_info = resolve_compile_info(cfg)
    report_cfg = cfg["report"]
    out_dir = Path(report_cfg["out_dir"])
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"{report_cfg['tag']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    dump_config(run_dir / "config_used.yaml", cfg)
    logger = setup_logger(run_dir / "report.txt")
    metrics_writer = JsonlWriter(run_dir / "metrics.jsonl")

    data = load_centralized_data(cfg, repo_root=repo_root)

    logger.info("[problem] source: %s", data.problem_source)
    logger.info("[consistency] formula/block max abs diff: %s", f"{data.matrix_max_abs_diff:.6e}")
    logger.info("[consistency] formula/block fro diff: %s", f"{data.matrix_fro_diff:.6e}")
    logger.info("[consistency] allclose: %s", data.matrix_allclose)
    logger.info("[matrix] cond(A): %s", f"{data.condition_number:.12e}")
    if not data.matrix_allclose:
        raise RuntimeError(
            "Matrix from loaded terms is not consistent with the loaded centralized target matrix. "
            "Please check static-ops centralized terms/matrix definition and wire ordering."
        )
    logger.info("[reference] system key: %s", data.reference_system_key)
    logger.info(
        "[reference] A_target vs A_ref max abs diff: %s",
        f"{data.reference_matrix_max_abs_diff:.6e}",
    )
    logger.info(
        "[reference] A_target vs A_ref fro diff: %s",
        f"{data.reference_matrix_fro_diff:.6e}",
    )
    logger.info("[reference] A_target vs A_ref allclose: %s", data.reference_matrix_allclose)
    logger.info(
        "[reference] b_target vs b_ref max abs diff: %s",
        f"{data.reference_b_max_abs_diff:.6e}",
    )
    logger.info(
        "[reference] b_target vs b_ref l2 diff: %s",
        f"{data.reference_b_l2_diff:.6e}",
    )
    logger.info("[reference] b_target vs b_ref allclose: %s", data.reference_b_allclose)

    logger.info("[compile] requested grad mode: %s", compile_info.requested_grad_mode)
    logger.info("[compile] resolved grad mode: %s", compile_info.resolved_grad_mode)
    logger.info("[compile] requested qnode compile: %s", compile_info.requested_qnode_compile)
    logger.info("[compile] resolved qnode compile: %s", compile_info.resolved_qnode_compile)
    logger.info("[compile] catalyst available: %s", compile_info.catalyst_available)
    if compile_info.catalyst_error:
        logger.info("[compile] catalyst import issue: %s", compile_info.catalyst_error)

    optimize_metric = str(cfg["optimization"]["optimize_metric"])
    need_local_for_objective = optimize_metric in {"global_CL", "global_CL_hat"}
    log_local_cost = bool(cfg["runtime"].get("log_local_cost", need_local_for_objective))
    include_local_in_reports = bool(log_local_cost or need_local_for_objective)
    enable_mu_terms = bool(need_local_for_objective or log_local_cost)

    evaluator = QJITHadamardCentralizedVQLS(
        data=data,
        device_name=str(cfg["runtime"]["device"]),
        diff_method=str(cfg["runtime"]["diff_method"]),
        qnode_compile=compile_info.resolved_qnode_compile,
        qjit_autograph=bool(cfg["runtime"].get("qjit_autograph", False)),
        enable_mu_terms=enable_mu_terms,
    )
    logger.info("[metrics] optimize_metric: %s", optimize_metric)
    logger.info("[metrics] log_local_cost: %s", log_local_cost)
    logger.info("[metrics] include_local_in_reports: %s", include_local_in_reports)
    logger.info("[metrics] mu_terms_enabled: %s", enable_mu_terms)
    logger.info("[metrics] beta_word_count: %d", len(evaluator.beta_word_weights))
    logger.info("[metrics] all_word_count: %d", len(evaluator.all_words))
    if enable_mu_terms:
        logger.info(
            "[metrics] mu_word_count_per_wire: %s",
            ",".join(str(len(row)) for row in evaluator.mu_word_weights_by_wire),
        )
    else:
        logger.info("[metrics] mu_word_count_per_wire: disabled")

    x_true = np.linalg.solve(data.a_block, data.b_unnorm)

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

    def objective_fn(w):
        return evaluator.objective(w, optimize_metric)
    grad_mode = compile_info.resolved_grad_mode

    opt = optax.adam(
        learning_rate=float(cfg["optimization"]["learning_rate"]),
        b1=float(cfg["optimization"]["beta1"]),
        b2=float(cfg["optimization"]["beta2"]),
        eps=float(cfg["optimization"]["eps"]),
    )
    opt_state = opt.init(theta)

    if grad_mode == "qjit":
        import catalyst

        def _qjit_wrap(fn):
            try:
                return qml.qjit(fn, autograph=bool(cfg["runtime"].get("qjit_autograph", False)))
            except TypeError:
                return qml.qjit(fn)

        @_qjit_wrap
        def compute_loss(params):
            return objective_fn(params)

        @_qjit_wrap
        def compute_grad(params):
            return catalyst.grad(objective_fn, method="auto")(params)
    elif grad_mode == "jax_jit":
        compute_loss = jax.jit(objective_fn)
        compute_grad = jax.jit(jax.grad(objective_fn))
    else:
        compute_loss = objective_fn
        compute_grad = jax.grad(objective_fn)

    if bool(cfg["runtime"].get("warmup_compile", True)):
        warmup_grads = compute_grad(theta)
        try:
            _ = jax.block_until_ready(warmup_grads)
        except Exception:
            pass
        warmup_loss = compute_loss(theta)
        try:
            _ = jax.block_until_ready(warmup_loss)
        except Exception:
            pass

    steps = int(cfg["optimization"]["steps"])
    print_every = int(cfg["optimization"]["print_every"])

    history: List[dict] = []
    checkpoints: List[dict] = []
    best = {"iteration": 0, "loss": float("inf")}
    t0 = time.time()

    for it in range(1, steps + 1):
        grads = compute_grad(theta)
        updates, opt_state = opt.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        loss_val = compute_loss(theta)
        try:
            loss_val = jax.block_until_ready(loss_val)
        except Exception:
            pass
        loss_float = _to_float(loss_val)
        elapsed_s = float(time.time() - t0)
        now_iso = datetime.now().isoformat(timespec="milliseconds")

        l2 = compute_l2_error_unnormalized(
            evaluator=evaluator,
            weights=theta,
            a_dense=data.a_block,
            b_unnorm=data.b_unnorm,
            b_unnorm_norm=data.b_unnorm_norm,
            x_true=x_true,
        )

        row = {
            "iteration": it,
            "time_iso": now_iso,
            "elapsed_s": elapsed_s,
            optimize_metric: loss_float,
            "l2_abs_raw": l2.abs_error_raw,
            "l2_rel_raw": l2.rel_error_raw,
            "l2_abs_aligned": l2.abs_error_aligned,
            "l2_rel_aligned": l2.rel_error_aligned,
            "phase_angle_rad": l2.phase_angle_rad,
            "ax_norm": l2.ax_norm,
            "scale_ratio": l2.scale_ratio,
            "residual_ax_minus_b_abs_raw": l2.residual_ax_minus_b_abs_raw,
            "residual_ax_minus_b_rel_raw": l2.residual_ax_minus_b_rel_raw,
            "residual_ax_minus_b_abs_aligned": l2.residual_ax_minus_b_abs_aligned,
            "residual_ax_minus_b_rel_aligned": l2.residual_ax_minus_b_rel_aligned,
            "residual_ax_minus_b_abs": l2.residual_ax_minus_b_abs,
            "residual_ax_minus_b_rel": l2.residual_ax_minus_b_rel,
        }
        history.append(row)

        if loss_float < best["loss"]:
            best = {"iteration": it, "loss": loss_float}

        if (it % print_every == 0) or (it == 1) or (it == steps):
            m = evaluator.metrics(theta, include_local_cost=include_local_in_reports)
            check = {
                "iteration": it,
                "time_iso": now_iso,
                "elapsed_s": elapsed_s,
                optimize_metric: loss_float,
                "global_CG": _to_float(m["global_CG"]),
                "global_CL": _to_float(m["global_CL"]) if include_local_in_reports else float("nan"),
                "global_CG_hat": _to_float(m["global_CG_hat"]),
                "global_CL_hat": _to_float(m["global_CL_hat"]) if include_local_in_reports else float("nan"),
                "beta": _to_float(m["beta"]),
                "overlap_sq": _to_float(m["overlap_sq"]),
                "l2_abs_raw": l2.abs_error_raw,
                "l2_rel_raw": l2.rel_error_raw,
                "l2_abs_aligned": l2.abs_error_aligned,
                "l2_rel_aligned": l2.rel_error_aligned,
                "phase_angle_rad": l2.phase_angle_rad,
                "ax_norm": l2.ax_norm,
                "scale_ratio": l2.scale_ratio,
                "residual_ax_minus_b_abs_raw": l2.residual_ax_minus_b_abs_raw,
                "residual_ax_minus_b_rel_raw": l2.residual_ax_minus_b_rel_raw,
                "residual_ax_minus_b_abs_aligned": l2.residual_ax_minus_b_abs_aligned,
                "residual_ax_minus_b_rel_aligned": l2.residual_ax_minus_b_rel_aligned,
                "residual_ax_minus_b_abs": l2.residual_ax_minus_b_abs,
                "residual_ax_minus_b_rel": l2.residual_ax_minus_b_rel,
            }
            checkpoints.append(check)
            metrics_writer.write(check)
            logger.info(
                "[iter %04d] %s=%.8e CG=%.8e CL=%s ||Ax_est-b||(aligned)=%.8e L2_rel(aligned)=%.8e "
                "L2_rel(raw)=%.8e backend=%s",
                it,
                optimize_metric,
                loss_float,
                check["global_CG"],
                "disabled" if not include_local_in_reports else f"{check['global_CL']:.8e}",
                l2.residual_ax_minus_b_abs_aligned,
                l2.rel_error_aligned,
                l2.rel_error_raw,
                grad_mode,
            )

    final_raw = evaluator.metrics(theta, include_local_cost=include_local_in_reports)
    final_metrics = {
        "global_CG": _to_float(final_raw["global_CG"]),
        "global_CL": _to_float(final_raw["global_CL"]) if include_local_in_reports else float("nan"),
        "global_CG_hat": _to_float(final_raw["global_CG_hat"]),
        "global_CL_hat": _to_float(final_raw["global_CL_hat"]) if include_local_in_reports else float("nan"),
        "beta": _to_float(final_raw["beta"]),
        "overlap_sq": _to_float(final_raw["overlap_sq"]),
    }
    final_l2 = compute_l2_error_unnormalized(
        evaluator=evaluator,
        weights=theta,
        a_dense=data.a_block,
        b_unnorm=data.b_unnorm,
        b_unnorm_norm=data.b_unnorm_norm,
        x_true=x_true,
    )
    final_metrics["l2_abs_raw"] = final_l2.abs_error_raw
    final_metrics["l2_rel_raw"] = final_l2.rel_error_raw
    final_metrics["l2_abs_aligned"] = final_l2.abs_error_aligned
    final_metrics["l2_rel_aligned"] = final_l2.rel_error_aligned
    final_metrics["phase_angle_rad"] = final_l2.phase_angle_rad
    final_metrics["residual_ax_minus_b_abs_raw"] = final_l2.residual_ax_minus_b_abs_raw
    final_metrics["residual_ax_minus_b_rel_raw"] = final_l2.residual_ax_minus_b_rel_raw
    final_metrics["residual_ax_minus_b_abs_aligned"] = final_l2.residual_ax_minus_b_abs_aligned
    final_metrics["residual_ax_minus_b_rel_aligned"] = final_l2.residual_ax_minus_b_rel_aligned
    final_metrics["residual_ax_minus_b_abs"] = final_l2.residual_ax_minus_b_abs
    final_metrics["residual_ax_minus_b_rel"] = final_l2.residual_ax_minus_b_rel

    theta_final = np.asarray(theta, dtype=np.float64)
    x_norm_final = np.array(evaluator.state(theta), dtype=np.complex128)
    ax_final = data.a_block @ x_norm_final
    scale_final = float(data.b_unnorm_norm / (np.linalg.norm(ax_final) + 1.0e-14))
    x_est_unnorm_raw = scale_final * x_norm_final
    x_est_unnorm_aligned = x_est_unnorm_raw * np.exp(-1.0j * final_l2.phase_angle_rad)

    solution_txt_path = run_dir / "solution_comparison.txt"
    with solution_txt_path.open("w", encoding="utf-8") as f:
        f.write("# Final variational parameter theta\n")
        f.write(np.array2string(theta_final, precision=16))
        f.write("\n\n")
        for name, vec in [
            ("x_true", x_true),
            ("x_est_unnorm_raw", x_est_unnorm_raw),
            ("x_est_unnorm_aligned", x_est_unnorm_aligned),
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
        "x_est_unnorm_raw": _format_complex_vector_preview(x_est_unnorm_raw),
        "x_est_unnorm_aligned": _format_complex_vector_preview(x_est_unnorm_aligned),
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
    append_compile_report_section(run_dir, compile_info=compile_info, cfg=cfg)

    with (run_dir / "runtime_backend.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "requested_grad_mode": compile_info.requested_grad_mode,
                "resolved_grad_mode": compile_info.resolved_grad_mode,
                "requested_qnode_compile": compile_info.requested_qnode_compile,
                "resolved_qnode_compile": compile_info.resolved_qnode_compile,
                "catalyst_available": compile_info.catalyst_available,
                "catalyst_error": compile_info.catalyst_error,
            },
            f,
            indent=2,
        )

    metrics_writer.close()
    logger.info("[done] report directory: %s", run_dir)


if __name__ == "__main__":
    main()
