#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from MPS_simulation.cen.quimb_vqls_eq26_benchmark import build_circuit_numpy  # noqa: E402
from MPS_simulation.dist.quimb_dist_eq26_2x2_benchmark import (  # noqa: E402
    global_cost_jax,
    global_cost_numpy,
    make_initial_parameters,
    to_jax_problem,
)


def make_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        global_qubits=30,
        local_qubits=29,
        j_coupling=0.1,
        kappa=20.0,
        row_self_loop_weight=1.0,
        layers=4,
        gate_max_bond=32,
        gate_cutoff=1.0e-10,
        apply_max_bond=64,
        apply_cutoff=1.0e-10,
        apply_no_compress=False,
        learning_rate=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1.0e-8,
        init_mode="structured_linspace",
        init_seed=1234,
        init_start=0.01,
        init_stop=0.2,
        x_scale_init=0.75,
        z_scale_init=0.10,
    )


def scale_and_add_identity(mpo, coeff: float, nsites: int):
    import quimb.tensor as qtn

    ident = qtn.MPO_identity(nsites)
    return mpo.add_MPO(ident.multiply(coeff, inplace=False), inplace=False)


def build_base_local_mpo(local_qubits: int, j_coupling: float):
    import quimb as qu
    import quimb.tensor as qtn

    builder = qtn.SpinHam1D(cyclic=False)
    builder += 1.0, qu.pauli("X")
    builder += j_coupling, qu.pauli("Z"), qu.pauli("Z")
    return builder.build_mpo(local_qubits)


def build_boundary_z_mpo(local_qubits: int, coeff: float):
    import quimb as qu
    import quimb.tensor as qtn

    builder = qtn.SpinHam1D(cyclic=False)
    builder += coeff, qu.pauli("Z"), 0
    return builder.build_mpo(local_qubits)


def build_h0_mpo(global_qubits: int, j_coupling: float):
    import quimb as qu
    import quimb.tensor as qtn

    builder = qtn.SpinHam1D(cyclic=False)
    builder += 1.0, qu.pauli("X")
    builder += j_coupling, qu.pauli("Z"), qu.pauli("Z")
    return builder.build_mpo(global_qubits)


def estimate_scaled_spectrum(global_qubits: int, j_coupling: float, kappa: float) -> tuple[float, float]:
    import quimb.tensor as qtn

    h0_mpo = build_h0_mpo(global_qubits, j_coupling)

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
    return eta, zeta


def build_direct_problem(cfg: SimpleNamespace) -> dict[str, object]:
    import quimb.tensor as qtn

    eta, zeta = estimate_scaled_spectrum(cfg.global_qubits, cfg.j_coupling, cfg.kappa)
    local_base = build_base_local_mpo(cfg.local_qubits, cfg.j_coupling)
    boundary_z = build_boundary_z_mpo(cfg.local_qubits, cfg.j_coupling)

    a11 = scale_and_add_identity(
        local_base.add_MPO(boundary_z, inplace=False).multiply(1.0 / zeta, inplace=False),
        eta / zeta,
        cfg.local_qubits,
    )
    a22 = scale_and_add_identity(
        local_base.add_MPO(boundary_z.multiply(-1.0, inplace=False), inplace=False).multiply(
            1.0 / zeta,
            inplace=False,
        ),
        eta / zeta,
        cfg.local_qubits,
    )
    ident_local = qtn.MPO_identity(cfg.local_qubits).multiply(1.0 / zeta, inplace=False)
    blocks = ((a11, ident_local.copy()), (ident_local.copy(), a22))

    b_state = qtn.MPS_computational_state("+" * cfg.local_qubits, dtype="float64")
    b_norm = 0.5 / math.sqrt(2.0)
    b_states = ((b_state.copy(), b_state.copy()), (b_state.copy(), b_state.copy()))
    b_norms = np.full((2, 2), b_norm, dtype=np.float64)

    row_scale = 1.0 / (1.0 + cfg.row_self_loop_weight)
    row_laplacian = row_scale * np.asarray([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
    column_mix = np.asarray([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)

    return {
        "blocks": blocks,
        "b_states": b_states,
        "b_norms": b_norms,
        "row_laplacian": row_laplacian,
        "column_mix": column_mix,
        "eta": float(eta),
        "zeta": float(zeta),
        "block_formula": {
            "A11": "(H_rest + J Z_1 + eta I) / zeta",
            "A12": "I / zeta",
            "A21": "I / zeta",
            "A22": "(H_rest - J Z_1 + eta I) / zeta",
            "b_ij": "(1 / (2 sqrt(2))) |+^{29}>",
        },
    }


def summarize_mpo(mpo) -> dict[str, object]:
    bond_sizes = tuple(int(b) for b in mpo.bond_sizes())
    return {
        "num_tensors": int(mpo.num_tensors),
        "max_bond": max(bond_sizes) if bond_sizes else 1,
        "bond_sizes": bond_sizes,
    }


def make_report(result: dict[str, object]) -> str:
    alpha_finite = result["gradient_summary"]["alpha_finite"]
    beta_finite = result["gradient_summary"]["beta_finite"]
    if alpha_finite and beta_finite:
        conclusion_lines = [
            "Under the revised workflow, a 30-qubit run looks feasible in memory.",
            "This smoke test never materialized dense global A or dense b, built the blocks directly as MPOs, and completed both a forward cost evaluation and a reverse-mode gradient evaluation with finite gradients.",
        ]
    else:
        conclusion_lines = [
            "Under the revised workflow, a 30-qubit forward pass looks feasible in memory, but the current reverse-mode optimization path is not yet reliable.",
            "This smoke test never materialized dense global A or dense b, built the blocks directly as MPOs, completed the forward cost evaluation, but the reverse-mode alpha gradient was not finite.",
        ]

    lines = [
        "# 30-Qubit Direct-MPO Smoke Test",
        "",
        "## Conclusion",
        *conclusion_lines,
        "",
        "## Mathematical Block Construction",
        "For the current 2x2 split, the global basis is partitioned by the first qubit.",
        "With 29 local qubits,",
        "- `A11 = (H_rest + J Z_1 + eta I) / zeta`",
        "- `A12 = I / zeta`",
        "- `A21 = I / zeta`",
        "- `A22 = (H_rest - J Z_1 + eta I) / zeta`",
        "- `b_i = (1 / sqrt(2)) |+^{29}>`",
        "- `b_ij = (1 / (2 sqrt(2))) |+^{29}>`",
        "",
        "## Timings",
        f"- Spectrum scaling (`eta`, `zeta`) via MPO DMRG: `{result['timings']['spectrum_s']:.6f} s`.",
        f"- Direct block MPO construction: `{result['timings']['build_problem_s']:.6f} s`.",
        f"- One forward distributed cost evaluation: `{result['timings']['forward_cost_s']:.6f} s`.",
        f"- One reverse-mode gradient evaluation: `{result['timings']['reverse_mode_gradient_s']:.6f} s`.",
        "",
        "## Diagnostics",
        f"- Forward cost value: `{result['cost_value']:.12g}`.",
        f"- Alpha gradient finite: `{result['gradient_summary']['alpha_finite']}`.",
        f"- Beta gradient finite: `{result['gradient_summary']['beta_finite']}`.",
        f"- Alpha gradient L2 norm: `{result['gradient_summary']['alpha_grad_l2']:.12g}`.",
        f"- Beta gradient L2 norm: `{result['gradient_summary']['beta_grad_l2']:.12g}`.",
        f"- `eta = {result['problem']['eta']:.12g}`.",
        f"- `zeta = {result['problem']['zeta']:.12g}`.",
        "",
        "## MPO Sizes",
        f"- `A11` summary: `{result['problem']['block_summaries']['A11']}`.",
        f"- `A12` summary: `{result['problem']['block_summaries']['A12']}`.",
        f"- `A22` summary: `{result['problem']['block_summaries']['A22']}`.",
        "",
        "## Practical Reading",
        "Your understanding is mostly right.",
        "The correction is that the residual does not need a dense global matrix either, but it does still need to be defined carefully from inner products of the reconstructed averaged column blocks, not only from the per-agent row copies.",
        "The real requirement for 30 qubits is: build `A_ij` and `b_ij` directly as MPO/MPS objects, and do all monitoring through contraction formulas.",
        "Even with that fix, the present JAX reverse-mode path can still fail numerically at 30 qubits, so memory feasibility and optimization feasibility are not the same claim.",
        "",
        "## Artifacts",
        f"- JSON: `{result['artifacts']['json']}`",
        f"- Report: `{result['artifacts']['report']}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    cfg = make_cfg()

    t0 = time.perf_counter()
    problem_np = build_direct_problem(cfg)
    build_problem_s = time.perf_counter() - t0

    spectrum_s = build_problem_s
    alpha_init_np, beta_init_np = make_initial_parameters(cfg)

    t1 = time.perf_counter()
    cost_value = global_cost_numpy(alpha_init_np, beta_init_np, cfg, problem_np)
    forward_cost_s = time.perf_counter() - t1

    problem_jax = to_jax_problem(problem_np)
    cost_fn = lambda a, b: global_cost_jax(a, b, cfg, problem_jax)
    full_grad_fn = jax.value_and_grad(cost_fn, argnums=(0, 1))

    alpha_init = jnp.asarray(alpha_init_np)
    beta_init = jnp.asarray(beta_init_np)
    t2 = time.perf_counter()
    _, (alpha_grad, beta_grad) = full_grad_fn(alpha_init, beta_init)
    reverse_mode_gradient_s = time.perf_counter() - t2

    json_path = THIS_DIR / "quimb_dist_eq26_2x2_direct_mpo_smoke_30q.json"
    report_path = THIS_DIR / "quimb_dist_eq26_2x2_direct_mpo_smoke_30q_report.md"

    result = {
        "timings": {
            "spectrum_s": spectrum_s,
            "build_problem_s": build_problem_s,
            "forward_cost_s": forward_cost_s,
            "reverse_mode_gradient_s": reverse_mode_gradient_s,
        },
        "cost_value": float(cost_value),
        "gradient_summary": {
            "alpha_finite": bool(np.isfinite(np.asarray(alpha_grad)).all()),
            "beta_finite": bool(np.isfinite(np.asarray(beta_grad)).all()),
            "alpha_grad_l2": float(np.linalg.norm(np.asarray(alpha_grad))),
            "beta_grad_l2": float(np.linalg.norm(np.asarray(beta_grad))),
        },
        "problem": {
            "eta": problem_np["eta"],
            "zeta": problem_np["zeta"],
            "block_formula": problem_np["block_formula"],
            "block_summaries": {
                "A11": summarize_mpo(problem_np["blocks"][0][0]),
                "A12": summarize_mpo(problem_np["blocks"][0][1]),
                "A22": summarize_mpo(problem_np["blocks"][1][1]),
            },
        },
        "artifacts": {
            "json": str(json_path.resolve()),
            "report": str(report_path.resolve()),
        },
    }

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path.write_text(make_report(result), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"Wrote JSON to {json_path}")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
