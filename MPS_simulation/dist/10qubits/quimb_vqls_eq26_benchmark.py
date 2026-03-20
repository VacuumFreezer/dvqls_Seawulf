#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

warnings.filterwarnings(
    "ignore",
    message=r".*Couldn't find `optuna`, `cmaes`, or `nevergrad`.*",
)


@dataclass
class Config:
    n_qubits: list[int]
    j_coupling: float
    kappa: float
    layers: int
    gate_max_bond: int
    gate_cutoff: float
    apply_max_bond: int
    apply_cutoff: float
    forward_repeats: int
    reverse_repeats: int
    init_start: float
    init_stop: float
    out_json: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark direct MPS evaluation of the VQLS Eq. (26) Ising-like "
            "problem in quimb, plus gradient timings for reverse-mode and "
            "parameter shift."
        )
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        nargs="+",
        default=[26, 30],
        help="Problem sizes to benchmark.",
    )
    parser.add_argument(
        "--j-coupling",
        type=float,
        default=0.1,
        help="Nearest-neighbor ZZ coefficient J in Eq. (26).",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=20.0,
        help="Target condition number used to set eta and zeta.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=4,
        help=(
            "Number of paper-inspired Ry/CZ layers. Each layer uses an Ry "
            "block, odd/even CZ sublayers, then another Ry block."
        ),
    )
    parser.add_argument(
        "--gate-max-bond",
        type=int,
        default=32,
        help="Bond cap used while applying the variational circuit.",
    )
    parser.add_argument(
        "--gate-cutoff",
        type=float,
        default=1.0e-10,
        help="SVD cutoff used while applying the variational circuit.",
    )
    parser.add_argument(
        "--apply-max-bond",
        type=int,
        default=64,
        help="Bond cap used when applying the MPO A to the MPS state.",
    )
    parser.add_argument(
        "--apply-cutoff",
        type=float,
        default=1.0e-10,
        help="SVD cutoff used when applying the MPO A to the MPS state.",
    )
    parser.add_argument(
        "--forward-repeats",
        type=int,
        default=3,
        help="Number of timed forward cost evaluations after warmup.",
    )
    parser.add_argument(
        "--reverse-repeats",
        type=int,
        default=1,
        help="Number of timed reverse-mode evaluations after warmup.",
    )
    parser.add_argument(
        "--init-start",
        type=float,
        default=0.01,
        help="Start of the linspace initialization for the benchmark angles.",
    )
    parser.add_argument(
        "--init-stop",
        type=float,
        default=0.2,
        help="End of the linspace initialization for the benchmark angles.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional path to write the benchmark results as JSON.",
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> Config:
    return Config(
        n_qubits=list(args.n_qubits),
        j_coupling=float(args.j_coupling),
        kappa=float(args.kappa),
        layers=int(args.layers),
        gate_max_bond=int(args.gate_max_bond),
        gate_cutoff=float(args.gate_cutoff),
        apply_max_bond=int(args.apply_max_bond),
        apply_cutoff=float(args.apply_cutoff),
        forward_repeats=int(args.forward_repeats),
        reverse_repeats=int(args.reverse_repeats),
        init_start=float(args.init_start),
        init_stop=float(args.init_stop),
        out_json=args.out_json,
    )


def block_until_ready(obj):
    if hasattr(obj, "block_until_ready"):
        obj.block_until_ready()
        return
    if isinstance(obj, tuple):
        for item in obj:
            block_until_ready(item)
        return
    if isinstance(obj, list):
        for item in obj:
            block_until_ready(item)
        return
    if isinstance(obj, dict):
        for item in obj.values():
            block_until_ready(item)


def summarize_times(times: list[float]) -> dict[str, float]:
    arr = np.asarray(times, dtype=np.float64)
    out = {
        "mean_s": float(arr.mean()),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
    }
    if arr.size > 1:
        out["std_s"] = float(arr.std(ddof=1))
    else:
        out["std_s"] = 0.0
    return out


def time_callable(fn: Callable[[], object], repeats: int, warmup: int = 1) -> dict[str, object]:
    last_value = None
    for _ in range(warmup):
        last_value = fn()
        block_until_ready(last_value)

    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        last_value = fn()
        block_until_ready(last_value)
        times.append(time.perf_counter() - t0)

    return {
        "timing": summarize_times(times),
        "last_value": last_value,
    }


def count_nonfinite(arr) -> int:
    arr_np = np.asarray(arr)
    return int(arr_np.size - np.count_nonzero(np.isfinite(arr_np)))


def build_scaled_problem_numpy(n: int, j_coupling: float, kappa: float):
    import quimb as qu
    import quimb.tensor as qtn

    x_op = qu.pauli("X")
    z_op = qu.pauli("Z")
    ident = qu.eye(2)

    h0_builder = qtn.SpinHam1D(cyclic=False)
    h0_builder += 1.0, x_op
    h0_builder += j_coupling, z_op, z_op
    h0_mpo = h0_builder.build_mpo(n)

    dmrg_min = qtn.DMRG2(
        h0_mpo,
        which="SA",
        bond_dims=[8, 16, 32, 64],
        cutoffs=[1.0e-8, 1.0e-10, 1.0e-12],
    )
    dmrg_min.solve(tol=1.0e-8, max_sweeps=6, verbosity=0)

    dmrg_max = qtn.DMRG2(
        h0_mpo,
        which="LA",
        bond_dims=[8, 16, 32, 64],
        cutoffs=[1.0e-8, 1.0e-10, 1.0e-12],
    )
    dmrg_max.solve(tol=1.0e-8, max_sweeps=6, verbosity=0)

    lambda_min = float(dmrg_min.energy)
    lambda_max = float(dmrg_max.energy)
    eta = (lambda_max - kappa * lambda_min) / (kappa - 1.0)
    zeta = lambda_max + eta

    a_builder = qtn.SpinHam1D(cyclic=False)
    a_builder += 1.0 / zeta, x_op
    a_builder += j_coupling / zeta, z_op, z_op
    a_builder += eta / zeta, ident

    a_mpo = a_builder.build_mpo(n)
    b_state = qtn.MPS_computational_state("+" * n, dtype="float64")

    return {
        "x_op": x_op,
        "z_op": z_op,
        "identity": ident,
        "A": a_mpo,
        "b": b_state,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "eta": float(eta),
        "zeta": float(zeta),
    }


def to_jax_problem(problem_np: dict[str, object]):
    import jax.numpy as jnp

    problem_jax = {
        "A": problem_np["A"].copy(),
        "b": problem_np["b"].copy(),
        "X": jnp.asarray(problem_np["x_op"]),
    }
    problem_jax["A"].apply_to_arrays(jnp.asarray)
    problem_jax["b"].apply_to_arrays(jnp.asarray)
    return problem_jax


def make_angles(n: int, layers: int, start: float, stop: float) -> np.ndarray:
    n_params = 2 * layers * n
    return np.linspace(start, stop, n_params, dtype=np.float64)


def build_circuit_numpy(n: int, angles: np.ndarray, cfg: Config):
    import quimb.tensor as qtn

    circ = qtn.CircuitMPS(
        n,
        cutoff=cfg.gate_cutoff,
        max_bond=cfg.gate_max_bond,
    )
    k = 0
    for _ in range(cfg.layers):
        for i in range(n):
            circ.ry(float(angles[k]), i)
            k += 1
        for start in (0, 1):
            for i in range(start, n - 1, 2):
                circ.cz(i, i + 1)
        for i in range(n):
            circ.ry(float(angles[k]), i)
            k += 1
    return circ


def build_circuit_jax(n: int, angles, cfg: Config):
    import quimb.tensor as qtn

    circ = qtn.CircuitMPS(
        n,
        cutoff=cfg.gate_cutoff,
        max_bond=cfg.gate_max_bond,
    )
    k = 0
    for _ in range(cfg.layers):
        for i in range(n):
            circ.ry(angles[k], i)
            k += 1
        for start in (0, 1):
            for i in range(start, n - 1, 2):
                circ.cz(i, i + 1)
        for i in range(n):
            circ.ry(angles[k], i)
            k += 1
    return circ


def cost_global_numpy(angles: np.ndarray, n: int, cfg: Config, problem_np: dict[str, object]) -> float:
    psi = build_circuit_numpy(n, angles, cfg).psi
    y = problem_np["A"].apply(
        psi,
        contract=True,
        compress=True,
        max_bond=cfg.apply_max_bond,
        cutoff=cfg.apply_cutoff,
    )
    norm = y.overlap(y)
    overlap = problem_np["b"].overlap(y)
    return float(1.0 - np.real(np.conjugate(overlap) * overlap / norm))


def cost_local_numpy(angles: np.ndarray, n: int, cfg: Config, problem_np: dict[str, object]) -> float:
    psi = build_circuit_numpy(n, angles, cfg).psi
    y = problem_np["A"].apply(
        psi,
        contract=True,
        compress=True,
        max_bond=cfg.apply_max_bond,
        cutoff=cfg.apply_cutoff,
    )
    norm = y.overlap(y)
    sx = 0.0
    for j in range(n):
        sx += y.local_expectation_canonical(problem_np["x_op"], j, normalized=False)
    return float(0.5 - 0.5 * np.real(sx / (n * norm)))


def cost_local_jax(angles, n: int, cfg: Config, problem_jax: dict[str, object]):
    import jax.numpy as jnp

    psi = build_circuit_jax(n, angles, cfg).psi
    y = problem_jax["A"].apply(
        psi,
        contract=True,
        compress=True,
        max_bond=cfg.apply_max_bond,
        cutoff=cfg.apply_cutoff,
    )
    norm = y.overlap(y)
    sx = 0.0
    for j in range(n):
        sx = sx + y.local_expectation_canonical(problem_jax["X"], j, normalized=False)
    return 0.5 - 0.5 * jnp.real(sx / (n * norm))


def parameter_shift_gradient(
    cost_fn: Callable[[np.ndarray], float],
    angles: np.ndarray,
    shift: float = math.pi / 2.0,
) -> np.ndarray:
    grad = np.empty_like(angles)
    for i in range(angles.size):
        plus = angles.copy()
        minus = angles.copy()
        plus[i] += shift
        minus[i] -= shift
        grad[i] = 0.5 * (cost_fn(plus) - cost_fn(minus))
    return grad


def benchmark_problem_size(n: int, cfg: Config) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    print(f"\n[n={n}] Building Eq. (26) MPO and scaling to kappa={cfg.kappa:.3f}")
    problem_np = build_scaled_problem_numpy(n, cfg.j_coupling, cfg.kappa)
    problem_jax = to_jax_problem(problem_np)
    angles_np = make_angles(n, cfg.layers, cfg.init_start, cfg.init_stop)
    angles_jax = jnp.asarray(angles_np)

    print(
        f"[n={n}] eta={problem_np['eta']:.12f}, zeta={problem_np['zeta']:.12f}, "
        f"num_params={angles_np.size}"
    )

    forward_global = time_callable(
        lambda: cost_global_numpy(angles_np, n, cfg, problem_np),
        repeats=cfg.forward_repeats,
        warmup=1,
    )
    print(
        f"[n={n}] CG forward mean={forward_global['timing']['mean_s']:.6f}s "
        f"value={forward_global['last_value']:.12f}"
    )

    forward_local = time_callable(
        lambda: cost_local_numpy(angles_np, n, cfg, problem_np),
        repeats=cfg.forward_repeats,
        warmup=1,
    )
    print(
        f"[n={n}] CL forward mean={forward_local['timing']['mean_s']:.6f}s "
        f"value={forward_local['last_value']:.12f}"
    )

    reverse_fn = jax.value_and_grad(
        lambda arr: cost_local_jax(arr, n, cfg, problem_jax)
    )
    reverse = time_callable(
        lambda: reverse_fn(angles_jax),
        repeats=cfg.reverse_repeats,
        warmup=1,
    )
    reverse_cost, reverse_grad = reverse["last_value"]
    reverse_grad_np = np.asarray(reverse_grad)
    reverse_nonfinite = count_nonfinite(reverse_grad_np)
    reverse_status = "ok" if reverse_nonfinite == 0 else "nonfinite"
    print(
        f"[n={n}] CL reverse-mode mean={reverse['timing']['mean_s']:.6f}s "
        f"status={reverse_status} nonfinite={reverse_nonfinite}"
    )

    shift_cost_fn = lambda arr: cost_local_numpy(arr, n, cfg, problem_np)
    shift = time_callable(
        lambda: parameter_shift_gradient(shift_cost_fn, angles_np),
        repeats=1,
        warmup=0,
    )
    shift_grad = np.asarray(shift["last_value"])
    shift_nonfinite = count_nonfinite(shift_grad)
    shift_status = "ok" if shift_nonfinite == 0 else "nonfinite"
    print(
        f"[n={n}] CL parameter-shift time={shift['timing']['mean_s']:.6f}s "
        f"status={shift_status} nonfinite={shift_nonfinite}"
    )

    comparison = None
    if reverse_nonfinite == 0 and shift_nonfinite == 0:
        denom = np.linalg.norm(shift_grad)
        comparison = {
            "l2_diff": float(np.linalg.norm(reverse_grad_np - shift_grad)),
            "relative_l2_diff": float(
                np.linalg.norm(reverse_grad_np - shift_grad) / max(denom, 1.0e-12)
            ),
        }

    return {
        "n_qubits": n,
        "num_params": int(angles_np.size),
        "lambda_min": problem_np["lambda_min"],
        "lambda_max": problem_np["lambda_max"],
        "eta": problem_np["eta"],
        "zeta": problem_np["zeta"],
        "forward": {
            "CG": {
                "value": float(forward_global["last_value"]),
                **forward_global["timing"],
            },
            "CL": {
                "value": float(forward_local["last_value"]),
                **forward_local["timing"],
            },
        },
        "gradients": {
            "reverse_mode_jax": {
                "objective": "CL",
                "value": float(np.asarray(reverse_cost)),
                "status": reverse_status,
                "nonfinite_count": reverse_nonfinite,
                "grad_l2_norm": float(np.linalg.norm(np.nan_to_num(reverse_grad_np))),
                **reverse["timing"],
            },
            "parameter_shift": {
                "objective": "CL",
                "status": shift_status,
                "nonfinite_count": shift_nonfinite,
                "grad_l2_norm": float(np.linalg.norm(np.nan_to_num(shift_grad))),
                **shift["timing"],
            },
        },
        "gradient_comparison": comparison,
    }


def main() -> None:
    args = parse_args()
    cfg = make_config(args)

    import jax

    jax.config.update("jax_enable_x64", True)

    output = {
        "config": asdict(cfg),
        "results": [benchmark_problem_size(n, cfg) for n in cfg.n_qubits],
    }

    print("\nJSON summary:")
    print(json.dumps(output, indent=2))

    if cfg.out_json is not None:
        out_path = Path(cfg.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
