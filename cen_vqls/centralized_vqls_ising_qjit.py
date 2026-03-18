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
        _apply_controlled_b_dagger_from_spec,
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
        _apply_controlled_b_dagger_from_spec,
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
        "jit_training_step": True,
        "warmup_compile": True,
    },
    "report": {
        "out_dir": "cen_vqls/reports",
        "tag": "centralized_vqls_ising_qjit",
    },
}


@dataclass
class RuntimeBackendInfo:
    requested_qjit: bool
    backend: str
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


def detect_runtime_backend(cfg: dict) -> RuntimeBackendInfo:
    requested_qjit = bool(cfg["runtime"].get("use_qjit", True))
    fallback_to_jax = bool(cfg["runtime"].get("fallback_to_jax_jit", True))

    if not requested_qjit:
        return RuntimeBackendInfo(
            requested_qjit=False,
            backend="jax_jit",
            catalyst_available=False,
            catalyst_error=None,
        )

    try:
        import catalyst  # noqa: F401

        return RuntimeBackendInfo(
            requested_qjit=True,
            backend="qjit",
            catalyst_available=True,
            catalyst_error=None,
        )
    except Exception as exc:
        if fallback_to_jax:
            return RuntimeBackendInfo(
                requested_qjit=True,
                backend="jax_jit",
                catalyst_available=False,
                catalyst_error=str(exc),
            )
        raise RuntimeError(
            "Catalyst qjit requested but Catalyst import failed. "
            "Set runtime.fallback_to_jax_jit=true to continue with JAX-JIT fallback. "
            f"Original error: {exc}"
        ) from exc


class QJITHadamardCentralizedVQLS:
    def __init__(
        self,
        data: CentralizedIsingData,
        device_name: str,
        diff_method: str,
        backend: str,
        qjit_autograph: bool,
    ):
        self.data = data
        self.n = data.n_total_qubits
        self.terms = data.terms
        self.identity_word = tuple("I" for _ in range(self.n))
        self.diff_method = str(diff_method)
        self.backend = str(backend)
        self.qjit_autograph = bool(qjit_autograph)

        self.control_wire = 0
        self.system_wires = tuple(range(1, self.n + 1))
        self.dev_h = qml.device(device_name, wires=self.n + 1)
        self.dev_state = qml.device(device_name, wires=self.n)

        self.b_mode = data.b_unitary_info.mode
        self.b_prep_spec = data.b_unitary_info.prep_spec
        self.b_unitary_dag = (
            None
            if data.b_unitary_info.unitary is None
            else np.conjugate(data.b_unitary_info.unitary.T)
        )

        self._build_linear_combinations()
        self._build_compiled_functions()

    def _compile_fn(self, fn):
        if self.backend == "qjit":
            try:
                return qml.qjit(fn, autograph=self.qjit_autograph)
            except TypeError:
                return qml.qjit(fn)
        return jax.jit(fn)

    def _apply_b_dagger_controlled(self) -> None:
        if self.b_mode == "hadamard_tensor":
            for wire in self.system_wires:
                qml.ctrl(qml.Hadamard, control=self.control_wire)(wires=wire)
            return

        if self.b_prep_spec is not None:
            _apply_controlled_b_dagger_from_spec(
                self.b_prep_spec,
                self.control_wire,
                wire_offset=1,
            )
            return

        if self.b_unitary_dag is None:
            raise RuntimeError(f"Missing b-unitary implementation for mode `{self.b_mode}`.")

        def _b_dag():
            qml.QubitUnitary(self.b_unitary_dag, wires=self.system_wires)

        qml.ctrl(_b_dag, control=self.control_wire)()

    def _build_linear_combinations(self) -> None:
        self.beta_word_weights: Dict[PauliWord, complex] = {}
        self.mu_word_weights_by_wire: List[Dict[PauliWord, complex]] = [
            {} for _ in range(self.n)
        ]

        for c_l, w_l in self.terms:
            for c_lp, w_lp in self.terms:
                pair_coeff = complex(c_l) * complex(c_lp)

                phase_beta, w_beta = _multiply_pauli_words(w_l, w_lp)
                self.beta_word_weights[w_beta] = (
                    self.beta_word_weights.get(w_beta, 0.0 + 0.0j) + pair_coeff * phase_beta
                )

                for wire in range(self.n):
                    x_word = _word_with_single_pauli(self.n, wire, "X")
                    phase_1, tmp = _multiply_pauli_words(w_l, x_word)
                    phase_2, w_full = _multiply_pauli_words(tmp, w_lp)
                    row = self.mu_word_weights_by_wire[wire]
                    row[w_full] = row.get(w_full, 0.0 + 0.0j) + pair_coeff * phase_1 * phase_2

        all_words = set(self.beta_word_weights.keys())
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
        # State readout is metric-only; disable differentiation to avoid
        # adjoint + qml.state incompatibility on lightning.qubit.
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

    def metrics(self, weights: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        mu_cache: Dict[PauliWord, jnp.ndarray] = {}
        for word in self.all_words:
            mu_cache[word] = self._expect_word_on_x(weights, word)

        beta = jnp.asarray(0.0, dtype=jnp.float64)
        for word, alpha in self.beta_word_weights.items():
            beta = beta + float(np.real(alpha)) * mu_cache[word]

        plus_sum = jnp.asarray(0.0, dtype=jnp.float64)
        for wire in range(self.n):
            mu_j = jnp.asarray(0.0, dtype=jnp.float64)
            for word, alpha in self.mu_word_weights_by_wire[wire].items():
                mu_j = mu_j + float(np.real(alpha)) * mu_cache[word]
            plus_sum = plus_sum + 0.5 * (beta + mu_j)

        gamma_re = jnp.asarray(0.0, dtype=jnp.float64)
        gamma_im = jnp.asarray(0.0, dtype=jnp.float64)
        for idx, (coeff, _) in enumerate(self.terms):
            gamma_re = gamma_re + coeff * self._gamma_real_fn[idx](weights)
            gamma_im = gamma_im + coeff * self._gamma_imag_fn[idx](weights)

        overlap_sq = gamma_re * gamma_re + gamma_im * gamma_im

        cghat = beta - overlap_sq
        clhat = 0.5 * beta - plus_sum / (2.0 * self.n)

        beta_safe = beta + 1.0e-14
        cg = 1.0 - overlap_sq / beta_safe
        cl = 0.5 - plus_sum / (2.0 * self.n * beta_safe)

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
        vals = self.metrics(weights)
        if metric_name not in vals:
            raise ValueError(f"Unknown optimize metric: {metric_name}")
        return vals[metric_name]


def append_qjit_report_section(run_dir: Path, backend_info: RuntimeBackendInfo, cfg: dict) -> None:
    report_path = run_dir / "report.md"
    lines = []
    lines.append("")
    lines.append("## QJIT Backend")
    lines.append(f"- Requested qjit: {backend_info.requested_qjit}")
    lines.append(f"- Active backend: {backend_info.backend}")
    lines.append(f"- Catalyst available: {backend_info.catalyst_available}")
    if backend_info.catalyst_error:
        lines.append("- Catalyst import error:")
        lines.append(f"  - `{backend_info.catalyst_error}`")
    lines.append(f"- qjit_autograph: {bool(cfg['runtime'].get('qjit_autograph', False))}")
    lines.append(f"- jit_training_step: {bool(cfg['runtime'].get('jit_training_step', True))}")

    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Centralized VQLS (Catalyst/JIT trial) for Ising block system")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().with_name("config_qjit.yaml")),
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
    parser.add_argument("--fallback_to_jax_jit", type=str, default=None)
    parser.add_argument("--out", type=str, default=None, help="Override report.out_dir")
    parser.add_argument("--tag", type=str, default=None, help="Override report.tag")
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)

    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(Path(args.config).resolve())
    cfg = apply_cli_overrides(cfg, args, repo_root=repo_root)

    backend_info = detect_runtime_backend(cfg)
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

    optimize_metric = str(cfg["optimization"]["optimize_metric"])

    def objective_fn(w):
        return evaluator.objective(w, optimize_metric)

    value_and_grad = jax.value_and_grad(objective_fn)

    opt = optax.adam(
        learning_rate=float(cfg["optimization"]["learning_rate"]),
        b1=float(cfg["optimization"]["beta1"]),
        b2=float(cfg["optimization"]["beta2"]),
        eps=float(cfg["optimization"]["eps"]),
    )
    opt_state = opt.init(theta)

    if bool(cfg["runtime"].get("jit_training_step", True)):

        @jax.jit
        def train_step(params, state):
            loss, grads = value_and_grad(params)
            updates, state = opt.update(grads, state, params)
            params = optax.apply_updates(params, updates)
            return params, state, loss

    else:

        def train_step(params, state):
            loss, grads = value_and_grad(params)
            updates, state = opt.update(grads, state, params)
            params = optax.apply_updates(params, updates)
            return params, state, loss

    if bool(cfg["runtime"].get("warmup_compile", True)):
        warmup_loss = objective_fn(theta)
        _ = jax.block_until_ready(warmup_loss)

    steps = int(cfg["optimization"]["steps"])
    print_every = int(cfg["optimization"]["print_every"])

    history: List[dict] = []
    checkpoints: List[dict] = []
    best = {"iteration": 0, "loss": float("inf")}
    t0 = time.time()

    for it in range(1, steps + 1):
        theta, opt_state, loss_val = train_step(theta, opt_state)
        loss_val = jax.block_until_ready(loss_val)
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
            m = evaluator.metrics(theta)
            check = {
                "iteration": it,
                "time_iso": now_iso,
                "elapsed_s": elapsed_s,
                optimize_metric: loss_float,
                "global_CG": _to_float(m["global_CG"]),
                "global_CL": _to_float(m["global_CL"]),
                "global_CG_hat": _to_float(m["global_CG_hat"]),
                "global_CL_hat": _to_float(m["global_CL_hat"]),
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
                "[iter %04d] %s=%.8e CG=%.8e CL=%.8e ||Ax_est-b||(aligned)=%.8e L2_rel(aligned)=%.8e "
                "L2_rel(raw)=%.8e backend=%s",
                it,
                optimize_metric,
                loss_float,
                check["global_CG"],
                check["global_CL"],
                l2.residual_ax_minus_b_abs_aligned,
                l2.rel_error_aligned,
                l2.rel_error_raw,
                backend_info.backend,
            )

    final_raw = evaluator.metrics(theta)
    final_metrics = {
        "global_CG": _to_float(final_raw["global_CG"]),
        "global_CL": _to_float(final_raw["global_CL"]),
        "global_CG_hat": _to_float(final_raw["global_CG_hat"]),
        "global_CL_hat": _to_float(final_raw["global_CL_hat"]),
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

    metrics_writer.close()
    logger.info("[done] report directory: %s", run_dir)


if __name__ == "__main__":
    main()
