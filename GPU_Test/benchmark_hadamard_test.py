#!/usr/bin/env python3
"""Benchmark a 20-qubit Hadamard-test circuit in PennyLane.

This script runs one configuration:
- device: default.qubit or lightning.qubit
- interface: numpy, torch, or jax
- diff method: backprop, adjoint, or finite_diff
- compute target: cpu or gpu (used to place interface tensors when possible)

It writes structured JSON output so a Slurm array can sweep all combinations.
"""

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
import pennylane as qml


DIFF_METHOD_MAP = {
    "backprop": "backprop",
    "adjoint": "adjoint",
    "finite_diff": "finite-diff",
    "finite-diff": "finite-diff",
}

INTERFACE_TO_QNODE = {
    # "numpy" in the request is mapped to PennyLane's autograd interface.
    "numpy": "autograd",
    "torch": "torch",
    "jax": "jax",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Hadamard test timing in PennyLane.")
    parser.add_argument("--device", required=True, choices=["default.qubit", "lightning.qubit"])
    parser.add_argument("--interface", required=True, choices=["numpy", "torch", "jax"])
    parser.add_argument(
        "--diff-method",
        required=True,
        choices=["backprop", "adjoint", "finite_diff", "finite-diff"],
    )
    parser.add_argument("--compute-target", required=True, choices=["cpu", "gpu"])
    parser.add_argument("--num-qubits", type=int, default=20, help="Total qubits including ancilla.")
    parser.add_argument("--layers", type=int, default=2, help="Shallow ansatz depth.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval-warmup", type=int, default=1)
    parser.add_argument("--eval-repeats", type=int, default=3)
    parser.add_argument("--grad-warmup", type=int, default=1)
    parser.add_argument("--grad-repeats", type=int, default=1)
    parser.add_argument("--shots", type=int, default=None, help="None means analytic mode.")
    parser.add_argument("--json-out", required=True, help="Where to write JSON results.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if status is not ok.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def classify_error(message: str) -> str:
    msg = message.lower()
    if "no module named" in msg or "cannot import" in msg:
        return "missing_dependency"
    unsupported_tokens = [
        "not support",
        "unsupported",
        "does not provide",
        "cannot differentiate",
        "differentiation method",
        "gradient method",
        "not implemented",
    ]
    if any(token in msg for token in unsupported_tokens):
        return "unsupported"
    return "failed"


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


def apply_controlled_probe_unitary(ancilla: int, system_wires: list[int], angles: np.ndarray) -> None:
    # Diagonal phase probe + shallow ring entangling, all controlled by ancilla.
    for idx, wire in enumerate(system_wires):
        qml.ctrl(qml.RZ, control=ancilla)(float(angles[idx]), wires=wire)
    for idx in range(len(system_wires) - 1):
        qml.ctrl(qml.CNOT, control=ancilla)(wires=[system_wires[idx], system_wires[idx + 1]])
    if len(system_wires) > 2:
        qml.ctrl(qml.CNOT, control=ancilla)(wires=[system_wires[-1], system_wires[0]])


def hadamard_test_circuit(weights: Any, unitary_angles: np.ndarray, num_qubits: int) -> Any:
    ancilla = 0
    system_wires = list(range(1, num_qubits))
    qml.Hadamard(wires=ancilla)
    qml.StronglyEntanglingLayers(weights=weights, wires=system_wires)
    apply_controlled_probe_unitary(ancilla=ancilla, system_wires=system_wires, angles=unitary_angles)
    qml.Hadamard(wires=ancilla)
    return qml.expval(qml.PauliZ(ancilla))


def import_torch():
    import torch  # type: ignore

    return torch


def import_jax():
    from jax import config as jax_config  # type: ignore

    jax_config.update("jax_enable_x64", True)
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore

    return jax, jnp


def initialize_parameters(
    interface: str,
    shape: tuple[int, ...],
    seed: int,
    compute_target: str,
) -> tuple[Any, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    init = rng.uniform(low=-0.25, high=0.25, size=shape)

    if interface == "numpy":
        params = qml.numpy.array(init, requires_grad=True)
        return params, {"parameter_backend": "autograd_numpy", "parameter_device": "cpu"}

    if interface == "torch":
        torch = import_torch()
        want_cuda = compute_target == "gpu"
        has_cuda = bool(torch.cuda.is_available())
        tensor_device = torch.device("cuda" if (want_cuda and has_cuda) else "cpu")
        params = torch.tensor(init, dtype=torch.float64, device=tensor_device, requires_grad=True)
        return params, {"parameter_backend": "torch", "parameter_device": str(tensor_device)}

    if interface == "jax":
        jax, jnp = import_jax()
        params = jnp.asarray(init, dtype=jnp.float64)
        return params, {"parameter_backend": "jax", "parameter_device": jax.default_backend()}

    raise ValueError(f"Unsupported interface: {interface}")


def sync_value(interface: str, value: Any) -> float:
    if interface == "torch":
        torch = import_torch()
        if isinstance(value, torch.Tensor) and value.is_cuda:
            torch.cuda.synchronize(value.device)
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    if interface == "jax":
        jax, _ = import_jax()
        ready = jax.block_until_ready(value)
        return float(np.asarray(ready))

    return float(np.asarray(value))


def sync_gradient(interface: str, grad_value: Any) -> np.ndarray:
    if interface == "torch":
        torch = import_torch()
        if isinstance(grad_value, torch.Tensor) and grad_value.is_cuda:
            torch.cuda.synchronize(grad_value.device)
        if isinstance(grad_value, torch.Tensor):
            return grad_value.detach().cpu().numpy()
        return np.asarray(grad_value)

    if interface == "jax":
        jax, _ = import_jax()
        ready = jax.block_until_ready(grad_value)
        return np.asarray(ready)

    return np.asarray(grad_value)


def time_eval(circuit, params: Any, interface: str, warmup: int, repeats: int) -> tuple[list[float], float]:
    if interface == "torch":
        torch = import_torch()
        with torch.no_grad():
            for _ in range(warmup):
                _ = sync_value(interface, circuit(params))
            timings = []
            last_value = float("nan")
            for _ in range(repeats):
                start = time.perf_counter()
                out = circuit(params)
                last_value = sync_value(interface, out)
                timings.append(time.perf_counter() - start)
            return timings, last_value

    for _ in range(warmup):
        _ = sync_value(interface, circuit(params))
    timings = []
    last_value = float("nan")
    for _ in range(repeats):
        start = time.perf_counter()
        out = circuit(params)
        last_value = sync_value(interface, out)
        timings.append(time.perf_counter() - start)
    return timings, last_value


def time_grad(circuit, params: Any, interface: str, warmup: int, repeats: int) -> tuple[list[float], tuple[int, ...], float]:
    if interface == "numpy":
        grad_fn = qml.grad(circuit)
        for _ in range(warmup):
            _ = sync_gradient(interface, grad_fn(params))
        timings = []
        grad_np = np.array([])
        for _ in range(repeats):
            start = time.perf_counter()
            grad = grad_fn(params)
            grad_np = sync_gradient(interface, grad)
            timings.append(time.perf_counter() - start)
        return timings, tuple(grad_np.shape), float(np.linalg.norm(grad_np.ravel()))

    if interface == "torch":
        torch = import_torch()

        def grad_once() -> Any:
            if params.grad is not None:
                params.grad.zero_()
            out = circuit(params)
            out.backward()
            if params.is_cuda:
                torch.cuda.synchronize(params.device)
            return params.grad

        for _ in range(warmup):
            _ = grad_once()
        timings = []
        grad_np = np.array([])
        for _ in range(repeats):
            start = time.perf_counter()
            grad = grad_once()
            grad_np = sync_gradient(interface, grad)
            timings.append(time.perf_counter() - start)
        return timings, tuple(grad_np.shape), float(np.linalg.norm(grad_np.ravel()))

    if interface == "jax":
        jax, _ = import_jax()
        grad_fn = jax.grad(circuit)
        for _ in range(warmup):
            warm_grad = grad_fn(params)
            _ = sync_gradient(interface, warm_grad)
        timings = []
        grad_np = np.array([])
        for _ in range(repeats):
            start = time.perf_counter()
            grad = grad_fn(params)
            grad_np = sync_gradient(interface, grad)
            timings.append(time.perf_counter() - start)
        return timings, tuple(grad_np.shape), float(np.linalg.norm(grad_np.ravel()))

    raise ValueError(f"Unsupported interface for grad timing: {interface}")


def collect_environment(interface: str) -> dict[str, Any]:
    env: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python_version": sys.version.split()[0],
        "pennylane_version": qml.__version__,
    }
    if interface == "torch":
        torch = import_torch()
        env["torch_version"] = getattr(torch, "__version__", "unknown")
        env["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
    elif interface == "jax":
        jax, _ = import_jax()
        env["jax_version"] = getattr(jax, "__version__", "unknown")
        env["jax_default_backend"] = jax.default_backend()
        env["jax_devices"] = [str(d) for d in jax.devices()]
    return env


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_qubits < 2:
        raise ValueError("--num-qubits must be >= 2 for Hadamard test (ancilla + system).")
    if args.layers < 1:
        raise ValueError("--layers must be >= 1.")
    if args.eval_repeats < 1 or args.grad_repeats < 1:
        raise ValueError("--eval-repeats and --grad-repeats must be >= 1.")

    qnode_interface = INTERFACE_TO_QNODE[args.interface]
    diff_method = DIFF_METHOD_MAP[args.diff_method]
    system_qubits = args.num_qubits - 1
    weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=args.layers, n_wires=system_qubits)
    num_parameters = int(np.prod(weight_shape))
    unitary_angles = np.linspace(0.11, 1.91, system_qubits, dtype=float)

    dev = qml.device(args.device, wires=args.num_qubits, shots=args.shots)

    @qml.qnode(dev, interface=qnode_interface, diff_method=diff_method)
    def circuit(weights):
        return hadamard_test_circuit(weights=weights, unitary_angles=unitary_angles, num_qubits=args.num_qubits)

    params, param_meta = initialize_parameters(
        interface=args.interface,
        shape=weight_shape,
        seed=args.seed,
        compute_target=args.compute_target,
    )

    eval_samples, last_value = time_eval(
        circuit=circuit,
        params=params,
        interface=args.interface,
        warmup=args.eval_warmup,
        repeats=args.eval_repeats,
    )
    grad_samples, grad_shape, grad_l2 = time_grad(
        circuit=circuit,
        params=params,
        interface=args.interface,
        warmup=args.grad_warmup,
        repeats=args.grad_repeats,
    )

    return {
        "status": "ok",
        "configuration": {
            "compute_target": args.compute_target,
            "device": args.device,
            "interface_requested": args.interface,
            "qnode_interface_used": qnode_interface,
            "diff_method_requested": args.diff_method,
            "diff_method_used": diff_method,
            "num_qubits_total": args.num_qubits,
            "num_system_qubits": system_qubits,
            "layers": args.layers,
            "num_parameters": num_parameters,
            "shots": args.shots,
            "seed": args.seed,
            "eval_warmup": args.eval_warmup,
            "eval_repeats": args.eval_repeats,
            "grad_warmup": args.grad_warmup,
            "grad_repeats": args.grad_repeats,
        },
        "parameter_runtime": param_meta,
        "environment": collect_environment(args.interface),
        "timings": {
            "eval_seconds": stats(eval_samples),
            "grad_seconds": stats(grad_samples),
        },
        "samples": {
            "eval_seconds": eval_samples,
            "grad_seconds": grad_samples,
        },
        "metrics": {
            "last_expectation": last_value,
            "gradient_shape": list(grad_shape),
            "gradient_l2_norm": grad_l2,
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> int:
    args = parse_args()
    output_path = Path(args.json_out)
    base_payload: dict[str, Any] = {
        "status": "failed",
        "configuration": {
            "compute_target": args.compute_target,
            "device": args.device,
            "interface_requested": args.interface,
            "diff_method_requested": args.diff_method,
            "num_qubits_total": args.num_qubits,
            "layers": args.layers,
        },
        "environment": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "python_version": sys.version.split()[0],
            "pennylane_version": qml.__version__,
        },
    }

    try:
        payload = run_benchmark(args)
    except Exception as exc:  # noqa: BLE001
        message = f"{type(exc).__name__}: {exc}"
        payload = base_payload
        payload["status"] = classify_error(message)
        payload["error"] = {
            "message": message,
            "traceback": traceback.format_exc(),
        }
        if args.verbose:
            print(message, file=sys.stderr)
            print(payload["error"]["traceback"], file=sys.stderr)
    else:
        if args.verbose:
            eval_mean = payload["timings"]["eval_seconds"]["mean"]
            grad_mean = payload["timings"]["grad_seconds"]["mean"]
            print(
                "OK "
                f"target={args.compute_target} device={args.device} "
                f"interface={args.interface} diff={args.diff_method} "
                f"eval_mean={eval_mean:.6f}s grad_mean={grad_mean:.6f}s",
                flush=True,
            )

    write_json(output_path, payload)
    if args.verbose:
        print(f"Wrote benchmark JSON: {output_path}", flush=True)

    if args.strict and payload.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
