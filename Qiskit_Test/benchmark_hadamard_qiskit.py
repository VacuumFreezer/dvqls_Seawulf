#!/usr/bin/env python3
"""Benchmark 20-qubit Hadamard-test expectation and gradient timing with Qiskit Aer."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit_algorithms.gradients import ReverseEstimatorGradient, SPSAEstimatorGradient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qiskit benchmark for Hadamard test expectation/gradient.")
    parser.add_argument("--compute-target", required=True, choices=["cpu", "gpu"])
    parser.add_argument("--eval-mode", required=True, choices=["estimator", "sampler_v2"])
    parser.add_argument("--gradient-method", required=True, choices=["reverse", "spsa"])
    parser.add_argument("--num-qubits", type=int, default=20)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument(
        "--num-parameters",
        type=int,
        default=None,
        help="If set, build a fixed-parameter ansatz with exactly this many trainable parameters.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sampler-shots", type=int, default=1024)
    parser.add_argument("--eval-warmup", type=int, default=2)
    parser.add_argument("--eval-repeats", type=int, default=6)
    parser.add_argument("--grad-warmup", type=int, default=1)
    parser.add_argument("--grad-repeats", type=int, default=3)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def stats(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "median": None}
    arr = np.asarray(samples, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def classify_error(message: str) -> str:
    text = message.lower()
    if "no module named" in text or "cannot import" in text:
        return "missing_dependency"
    unsupported_tokens = [
        "does not support",
        "not supported",
        "unsupported",
        "cannot run",
        "invalid",
        "no cuda",
        "gpu",
    ]
    if any(tok in text for tok in unsupported_tokens):
        return "unsupported"
    return "failed"


def build_hadamard_test(
    num_qubits: int,
    layers: int,
    num_parameters: int | None = None,
) -> tuple[QuantumCircuit, ParameterVector, np.ndarray]:
    if num_qubits < 2:
        raise ValueError("num_qubits must be >= 2.")
    if layers < 1:
        raise ValueError("layers must be >= 1.")

    anc = 0
    system = list(range(1, num_qubits))
    n_system = len(system)
    if num_parameters is None:
        n_params = layers * n_system * 3
    else:
        if num_parameters < 1:
            raise ValueError("num_parameters must be >= 1 when provided.")
        n_params = int(num_parameters)
    theta = ParameterVector("theta", n_params)

    qc = QuantumCircuit(num_qubits, 1)
    qc.h(anc)

    if num_parameters is None:
        idx = 0
        for _ in range(layers):
            for q in system:
                qc.ry(theta[idx], q)
                idx += 1
                qc.rz(theta[idx], q)
                idx += 1
                qc.rx(theta[idx], q)
                idx += 1
            for i in range(n_system - 1):
                qc.cx(system[i], system[i + 1])
            if n_system > 2:
                qc.cx(system[-1], system[0])
    else:
        # Fixed-parameter shallow ansatz: consume each parameter exactly once.
        # First fill local rotations, then use any remainder on pairwise ZZ couplings.
        idx = 0
        for gate in ("ry", "rz", "rx"):
            for q in system:
                if idx >= n_params:
                    break
                if gate == "ry":
                    qc.ry(theta[idx], q)
                elif gate == "rz":
                    qc.rz(theta[idx], q)
                else:
                    qc.rx(theta[idx], q)
                idx += 1
            if idx >= n_params:
                break

        if n_system > 1 and idx < n_params:
            ring = [(system[i], system[(i + 1) % n_system]) for i in range(n_system)]
            edge_idx = 0
            while idx < n_params:
                a, b = ring[edge_idx % len(ring)]
                qc.rzz(theta[idx], a, b)
                idx += 1
                edge_idx += 1

        # Lightweight entangling scaffold with no additional parameters.
        for i in range(n_system - 1):
            qc.cx(system[i], system[i + 1])
        if n_system > 2:
            qc.cx(system[-1], system[0])

    probe_angles = np.linspace(0.11, 1.91, n_system, dtype=float)
    for i, q in enumerate(system):
        qc.crz(float(probe_angles[i]), anc, q)
    for i in range(n_system - 1):
        qc.ccx(anc, system[i], system[i + 1])
    if n_system > 2:
        qc.ccx(anc, system[-1], system[0])

    qc.h(anc)
    qc.measure(anc, 0)
    return qc, theta, probe_angles


def z_on_ancilla_observable(num_qubits: int) -> SparsePauliOp:
    # In Qiskit Pauli labels, the rightmost character corresponds to qubit 0.
    label = "I" * (num_qubits - 1) + "Z"
    return SparsePauliOp.from_list([(label, 1.0)])


def make_backend_options(compute_target: str) -> dict[str, Any]:
    device = "GPU" if compute_target == "gpu" else "CPU"
    return {
        "device": device,
        "method": "statevector",
    }


def bitarray_counts(pub_result) -> dict[str, int]:
    # SamplerV2 stores measured bit data under the classical register key.
    if hasattr(pub_result, "data"):
        for value in pub_result.data.values():
            if hasattr(value, "get_counts"):
                return value.get_counts()
    raise RuntimeError("Could not extract counts from SamplerV2 result.")


def expectation_from_counts(counts: dict[str, int]) -> float:
    zeros = int(counts.get("0", 0))
    ones = int(counts.get("1", 0))
    shots = zeros + ones
    if shots == 0:
        raise ValueError("No shots recorded in counts.")
    return float((zeros - ones) / shots)


def time_eval_estimator(
    estimator: EstimatorV2,
    circuit_no_meas: QuantumCircuit,
    observable: SparsePauliOp,
    params: np.ndarray,
    warmup: int,
    repeats: int,
) -> tuple[list[float], float, dict[str, Any]]:
    pub = (circuit_no_meas, observable, [params])
    for _ in range(warmup):
        _ = estimator.run([pub]).result()[0].data.evs

    samples: list[float] = []
    last_value = float("nan")
    last_metadata: dict[str, Any] = {}
    for _ in range(repeats):
        start = time.perf_counter()
        res = estimator.run([pub]).result()[0]
        samples.append(time.perf_counter() - start)
        last_value = float(np.asarray(res.data.evs).reshape(-1)[0])
        last_metadata = dict(res.metadata)
    return samples, last_value, last_metadata


def time_eval_sampler(
    sampler: SamplerV2,
    circuit_meas: QuantumCircuit,
    params: np.ndarray,
    warmup: int,
    repeats: int,
    shots: int,
) -> tuple[list[float], float, dict[str, Any], dict[str, int]]:
    pub = (circuit_meas, [params])
    for _ in range(warmup):
        _ = sampler.run([pub], shots=shots).result()[0]

    samples: list[float] = []
    last_value = float("nan")
    last_metadata: dict[str, Any] = {}
    last_counts: dict[str, int] = {}
    for _ in range(repeats):
        start = time.perf_counter()
        pub_res = sampler.run([pub], shots=shots).result()[0]
        samples.append(time.perf_counter() - start)
        counts = bitarray_counts(pub_res)
        last_counts = dict(counts)
        last_value = expectation_from_counts(counts)
        last_metadata = dict(pub_res.metadata)
    return samples, last_value, last_metadata, last_counts


def time_gradient(
    method: str,
    estimator: EstimatorV2,
    circuit_no_meas: QuantumCircuit,
    observable: SparsePauliOp,
    params: np.ndarray,
    warmup: int,
    repeats: int,
    seed: int,
) -> tuple[list[float], np.ndarray]:
    if method == "reverse":
        grad_impl = ReverseEstimatorGradient(estimator)
    elif method == "spsa":
        grad_impl = SPSAEstimatorGradient(estimator, epsilon=0.01, batch_size=1, seed=seed)
    else:
        raise ValueError(f"Unknown gradient method: {method}")

    pub = ([circuit_no_meas], [observable], [params])
    for _ in range(warmup):
        _ = grad_impl.run(*pub).result().gradients[0]

    samples: list[float] = []
    grad_array = np.array([])
    for _ in range(repeats):
        start = time.perf_counter()
        grad_res = grad_impl.run(*pub).result()
        samples.append(time.perf_counter() - start)
        grad_array = np.asarray(grad_res.gradients[0], dtype=float)
    return samples, grad_array


def collect_versions() -> dict[str, str]:
    import qiskit
    import qiskit_aer
    import qiskit_algorithms

    return {
        "python": sys.version.split()[0],
        "qiskit": getattr(qiskit, "__version__", "unknown"),
        "qiskit_aer": getattr(qiskit_aer, "__version__", "unknown"),
        "qiskit_algorithms": getattr(qiskit_algorithms, "__version__", "unknown"),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    qc_meas, theta, _ = build_hadamard_test(
        num_qubits=args.num_qubits,
        layers=args.layers,
        num_parameters=args.num_parameters,
    )
    qc_est = qc_meas.remove_final_measurements(inplace=False)
    obs = z_on_ancilla_observable(args.num_qubits)
    n_params = len(theta)
    params = rng.uniform(low=-0.25, high=0.25, size=n_params)

    backend_opts = make_backend_options(args.compute_target)
    estimator = EstimatorV2(options={"backend_options": backend_opts, "run_options": {}})
    sampler = SamplerV2(
        default_shots=args.sampler_shots,
        seed=args.seed,
        options={"backend_options": backend_opts, "run_options": {}},
    )

    if args.eval_mode == "estimator":
        eval_samples, eval_value, eval_md = time_eval_estimator(
            estimator=estimator,
            circuit_no_meas=qc_est,
            observable=obs,
            params=params,
            warmup=args.eval_warmup,
            repeats=args.eval_repeats,
        )
        sampler_counts: dict[str, int] = {}
    else:
        eval_samples, eval_value, eval_md, sampler_counts = time_eval_sampler(
            sampler=sampler,
            circuit_meas=qc_meas,
            params=params,
            warmup=args.eval_warmup,
            repeats=args.eval_repeats,
            shots=args.sampler_shots,
        )

    grad_samples, grad_arr = time_gradient(
        method=args.gradient_method,
        estimator=estimator,
        circuit_no_meas=qc_est,
        observable=obs,
        params=params,
        warmup=args.grad_warmup,
        repeats=args.grad_repeats,
        seed=args.seed + 77,
    )

    return {
        "status": "ok",
        "configuration": {
            "compute_target": args.compute_target,
            "eval_mode": args.eval_mode,
            "gradient_method": args.gradient_method,
            "num_qubits_total": args.num_qubits,
            "num_system_qubits": args.num_qubits - 1,
            "layers": args.layers,
            "num_parameters": n_params,
            "requested_num_parameters": args.num_parameters,
            "ansatz_mode": "fixed_param_budget" if args.num_parameters is not None else "layered_default",
            "sampler_shots": args.sampler_shots,
            "seed": args.seed,
            "eval_warmup": args.eval_warmup,
            "eval_repeats": args.eval_repeats,
            "grad_warmup": args.grad_warmup,
            "grad_repeats": args.grad_repeats,
            "backend_options": backend_opts,
        },
        "environment": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            **collect_versions(),
        },
        "timings": {
            "eval_seconds": stats(eval_samples),
            "grad_seconds": stats(grad_samples),
        },
        "samples": {
            "eval_seconds": eval_samples,
            "grad_seconds": grad_samples,
        },
        "metrics": {
            "last_expectation": float(eval_value),
            "gradient_shape": list(grad_arr.shape),
            "gradient_l2_norm": float(np.linalg.norm(grad_arr.ravel())),
            "sampler_last_counts": sampler_counts,
        },
        "metadata": {
            "eval_result_metadata": eval_md,
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> int:
    args = parse_args()
    output = Path(args.json_out)
    base = {
        "status": "failed",
        "configuration": {
            "compute_target": args.compute_target,
            "eval_mode": args.eval_mode,
            "gradient_method": args.gradient_method,
            "num_qubits_total": args.num_qubits,
            "layers": args.layers,
            "sampler_shots": args.sampler_shots,
        },
        "environment": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            **collect_versions(),
        },
    }

    try:
        payload = run_benchmark(args)
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        payload = base
        payload["status"] = classify_error(msg)
        payload["error"] = {"message": msg, "traceback": traceback.format_exc()}
        if args.verbose:
            print(msg, file=sys.stderr)
            print(payload["error"]["traceback"], file=sys.stderr)
    else:
        if args.verbose:
            print(
                "OK "
                f"target={args.compute_target} eval={args.eval_mode} grad={args.gradient_method} "
                f"eval_mean={payload['timings']['eval_seconds']['mean']:.6f}s "
                f"grad_mean={payload['timings']['grad_seconds']['mean']:.6f}s",
                flush=True,
            )

    write_json(output, payload)
    if args.verbose:
        print(f"Wrote {output}", flush=True)

    if args.strict and payload.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
