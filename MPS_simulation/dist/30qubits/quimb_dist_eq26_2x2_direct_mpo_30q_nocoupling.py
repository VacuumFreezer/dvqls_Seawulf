#!/usr/bin/env python3
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np


def scale_and_add_identity(mpo, coeff: float, nsites: int):
    import quimb.tensor as qtn

    ident = qtn.MPO_identity(nsites)
    return mpo.add_MPO(ident.multiply(coeff, inplace=False), inplace=False)


def build_transverse_field_local_mpo(local_qubits: int):
    import quimb as qu
    import quimb.tensor as qtn

    builder = qtn.SpinHam1D(cyclic=False)
    builder += 1.0, qu.pauli("X")
    return builder.build_mpo(local_qubits)


def build_transverse_field_global_mpo(global_qubits: int):
    import quimb as qu
    import quimb.tensor as qtn

    builder = qtn.SpinHam1D(cyclic=False)
    builder += 1.0, qu.pauli("X")
    return builder.build_mpo(global_qubits)


def estimate_scaled_spectrum(global_qubits: int, kappa: float) -> tuple[float, float]:
    import quimb.tensor as qtn

    h0_mpo = build_transverse_field_global_mpo(global_qubits)

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

    eta, zeta = estimate_scaled_spectrum(cfg.global_qubits, cfg.kappa)
    local_base = build_transverse_field_local_mpo(cfg.local_qubits)
    diag_block = scale_and_add_identity(
        local_base.multiply(1.0 / zeta, inplace=False),
        eta / zeta,
        cfg.local_qubits,
    )
    ident_local = qtn.MPO_identity(cfg.local_qubits).multiply(1.0 / zeta, inplace=False)
    blocks = ((diag_block.copy(), ident_local.copy()), (ident_local.copy(), diag_block.copy()))

    b_state = qtn.MPS_computational_state("+" * cfg.local_qubits, dtype="float64")
    b_row_norm = 1.0 / math.sqrt(2.0)
    b_norm = 0.5 / math.sqrt(2.0)
    b_row_states = (b_state.copy(), b_state.copy())
    b_row_norms = np.full(2, b_row_norm, dtype=np.float64)
    b_states = ((b_state.copy(), b_state.copy()), (b_state.copy(), b_state.copy()))
    b_norms = np.full((2, 2), b_norm, dtype=np.float64)

    row_scale = 1.0 / (1.0 + cfg.row_self_loop_weight)
    row_laplacian = row_scale * np.asarray([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
    column_mix = np.asarray([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)

    return {
        "blocks": blocks,
        "b_row_states": b_row_states,
        "b_row_norms": b_row_norms,
        "b_states": b_states,
        "b_norms": b_norms,
        "row_laplacian": row_laplacian,
        "column_mix": column_mix,
        "eta": float(eta),
        "zeta": float(zeta),
        "block_formula": {
            "A11": "(H_rest + eta I) / zeta",
            "A12": "I / zeta",
            "A21": "I / zeta",
            "A22": "(H_rest + eta I) / zeta",
            "hamiltonian": "sum_j X_j with all ZZ couplings removed from the MPO",
            "b_ij": "(1 / (2 sqrt(2))) |+^{29}>",
        },
    }
