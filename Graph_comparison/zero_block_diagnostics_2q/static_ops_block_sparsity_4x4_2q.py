from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import pennylane as qml
from pennylane import numpy as np

from Graph_comparison.Ising_2qubits.static_ops_ising_4x4_2q import LinearSystemData
from problems import static_ops_16agents_eq1_kappa196 as eq1_ref


TARGET_LAMBDA_MAX = 1.0
TARGET_LAMBDA_MIN = 1.0 / 50.0
N_AGENTS = 4
DATA_WIRES = [0, 1]
J = 0.1


def _identity_factory():
    return qml.Identity(wires=DATA_WIRES[0])


def _as_matrix(gate_fn: Callable[[], qml.operation.Operator], data_wires: Sequence[int]) -> np.ndarray:
    def qfunc():
        gate_fn()

    return np.array(qml.matrix(qfunc, wire_order=list(data_wires))())


def _build_matrix_from_blocks(
    gates_grid: Sequence[Sequence[Sequence[Callable[[], qml.operation.Operator]]]],
    coeffs_grid: Sequence[Sequence[Sequence[float]]],
    *,
    data_wires: Sequence[int],
) -> np.ndarray:
    block_rows = []
    for row_id in range(len(gates_grid)):
        row_blocks = []
        for col_id in range(len(gates_grid[row_id])):
            gates = gates_grid[row_id][col_id]
            coeffs = coeffs_grid[row_id][col_id]
            mats = [_as_matrix(gate, data_wires) for gate in gates]
            block = np.zeros_like(mats[0], dtype=np.complex128)
            for coeff, mat in zip(coeffs, mats):
                block = block + float(coeff) * mat
            row_blocks.append(block)
        block_rows.append(row_blocks)
    return np.block(block_rows)


def _affine_tune_coeffs(
    raw_gates,
    raw_coeffs,
    *,
    data_wires: Sequence[int],
) -> tuple[list, list, dict]:
    raw_matrix = np.asarray(
        _build_matrix_from_blocks(raw_gates, raw_coeffs, data_wires=data_wires),
        dtype=np.complex128,
    )
    evals = np.linalg.eigvalsh(raw_matrix)
    raw_min = float(np.min(evals).real)
    raw_max = float(np.max(evals).real)
    alpha = (TARGET_LAMBDA_MAX - TARGET_LAMBDA_MIN) / (raw_max - raw_min)
    beta = TARGET_LAMBDA_MAX - alpha * raw_max

    tuned_gates = []
    tuned_coeffs = []
    for row_id in range(len(raw_gates)):
        gate_row = []
        coeff_row = []
        for col_id in range(len(raw_gates[row_id])):
            gate_list = list(raw_gates[row_id][col_id])
            coeff_list = [float(alpha) * float(coeff) for coeff in raw_coeffs[row_id][col_id]]
            if row_id == col_id:
                gate_list.append(_identity_factory)
                coeff_list.append(float(beta))
            gate_row.append(gate_list)
            coeff_row.append(coeff_list)
        tuned_gates.append(gate_row)
        tuned_coeffs.append(coeff_row)

    tuned_matrix = np.asarray(
        _build_matrix_from_blocks(tuned_gates, tuned_coeffs, data_wires=data_wires),
        dtype=np.complex128,
    )
    tuned_evals = np.linalg.eigvalsh(tuned_matrix)
    metadata = {
        "raw_lambda_min": raw_min,
        "raw_lambda_max": raw_max,
        "alpha_scale": float(alpha),
        "beta_shift": float(beta),
        "spectrum": {
            "lambda_min": float(np.min(tuned_evals).real),
            "lambda_max": float(np.max(tuned_evals).real),
            "condition_number": float(np.max(tuned_evals).real / np.min(tuned_evals).real),
        },
    }
    return tuned_gates, tuned_coeffs, metadata


def _block_stats(system: LinearSystemData, *, tol: float = 1.0e-12) -> dict:
    block_norms = []
    zero_count = 0
    diag_zero_count = 0
    offdiag_zero_count = 0
    diag_norm_sum = 0.0
    offdiag_norm_sum = 0.0

    for row_id in range(system.n):
        row_norms = []
        for col_id in range(system.n):
            block = np.asarray(system.get_block_matrix(row_id, col_id), dtype=np.complex128)
            norm_val = float(np.linalg.norm(block))
            row_norms.append(norm_val)
            if row_id == col_id:
                diag_norm_sum += norm_val
            else:
                offdiag_norm_sum += norm_val
            if norm_val <= tol:
                zero_count += 1
                if row_id == col_id:
                    diag_zero_count += 1
                else:
                    offdiag_zero_count += 1
        block_norms.append(row_norms)

    return {
        "block_norm_matrix": block_norms,
        "zero_block_count": int(zero_count),
        "diag_zero_block_count": int(diag_zero_count),
        "offdiag_zero_block_count": int(offdiag_zero_count),
        "diag_block_norm_sum": float(diag_norm_sum),
        "offdiag_block_norm_sum": float(offdiag_norm_sum),
    }


def _build_ising_raw(fill_scale: float):
    data_wires = DATA_WIRES

    def i_op():
        return qml.Identity(wires=data_wires[0])

    def x_op(k: int):
        return lambda k=k: qml.PauliX(wires=data_wires[k])

    def z_op(k: int):
        return lambda k=k: qml.PauliZ(wires=data_wires[k])

    def zz_op(k: int):
        return lambda k=k: qml.prod(
            qml.PauliZ(wires=data_wires[k]),
            qml.PauliZ(wires=data_wires[k + 1]),
        )

    z0 = z_op(0)

    def make_diag_block(sign_q0q1: float, sign_q1d0: float):
        gates: List[Callable[[], qml.operation.Operator]] = []
        coeffs: List[float] = []

        gates.append(i_op)
        coeffs.append(sign_q0q1 * J)

        gates.append(z0)
        coeffs.append(sign_q1d0 * J)

        for idx in range(len(data_wires)):
            gates.append(x_op(idx))
            coeffs.append(1.0)

        for idx in range(len(data_wires) - 1):
            gates.append(zz_op(idx))
            coeffs.append(J)

        return gates, coeffs

    d00_g, d00_c = make_diag_block(sign_q0q1=+1.0, sign_q1d0=+1.0)
    d01_g, d01_c = make_diag_block(sign_q0q1=-1.0, sign_q1d0=-1.0)
    d10_g, d10_c = make_diag_block(sign_q0q1=-1.0, sign_q1d0=+1.0)
    d11_g, d11_c = make_diag_block(sign_q0q1=+1.0, sign_q1d0=-1.0)

    id_g, id_c = [i_op], [1.0]
    fill_g, fill_c = [i_op], [float(fill_scale)]

    raw_gates = [
        [d00_g, id_g, id_g, fill_g],
        [id_g, d01_g, fill_g, id_g],
        [id_g, fill_g, d10_g, id_g],
        [fill_g, id_g, id_g, d11_g],
    ]

    raw_coeffs = [
        [d00_c, id_c, id_c, fill_c],
        [id_c, d01_c, fill_c, id_c],
        [id_c, fill_c, d10_c, id_c],
        [fill_c, id_c, id_c, d11_c],
    ]

    def h_all():
        return qml.prod(*[qml.Hadamard(wires=wire) for wire in data_wires])

    raw_b_gates = [[h_all for _ in range(N_AGENTS)] for __ in range(N_AGENTS)]
    return raw_gates, raw_coeffs, raw_b_gates


def _build_eq1_raw():
    raw_gates = [[list(cell) for cell in row] for row in eq1_ref.RAW_GATES]
    raw_coeffs = []
    for row_id in range(len(eq1_ref.RAW_GATES)):
        coeff_row = []
        for col_id in range(len(eq1_ref.RAW_GATES[row_id])):
            coeff_row.append(
                list(
                    map(
                        float,
                        eq1_ref.get_coeffs(
                            float(eq1_ref.KAPPAS[row_id][col_id]),
                            len(eq1_ref.RAW_GATES[row_id][col_id]),
                        ),
                    )
                )
            )
        raw_coeffs.append(coeff_row)
    raw_b_gates = [[gate for gate in row] for row in eq1_ref.RAW_B_GATES]
    return raw_gates, raw_coeffs, raw_b_gates


def _build_system(
    *,
    name: str,
    raw_gates,
    raw_coeffs,
    raw_b_gates,
    variant_metadata: dict,
    tune_to_cond50: bool,
) -> LinearSystemData:
    if tune_to_cond50:
        tuned_gates, tuned_coeffs, tune_meta = _affine_tune_coeffs(
            raw_gates,
            raw_coeffs,
            data_wires=DATA_WIRES,
        )
    else:
        tuned_gates = [[list(cell) for cell in row] for row in raw_gates]
        tuned_coeffs = [[list(map(float, cell)) for cell in row] for row in raw_coeffs]
        raw_matrix = np.asarray(
            _build_matrix_from_blocks(tuned_gates, tuned_coeffs, data_wires=DATA_WIRES),
            dtype=np.complex128,
        )
        singular_values = np.linalg.svd(raw_matrix, compute_uv=False)
        tune_meta = {
            "svd_condition_number": float(np.max(singular_values) / np.min(singular_values)),
            "spectral_norm": float(np.max(singular_values)),
            "min_singular_value": float(np.min(singular_values)),
        }
    system = LinearSystemData(
        tuned_gates,
        tuned_coeffs,
        raw_b_gates,
        data_wires=DATA_WIRES,
        b_weights_grid=[[1.0 for _ in range(N_AGENTS)] for __ in range(N_AGENTS)],
        zeta=1.0,
        name=name,
        metadata={
            "benchmark_name": name,
            "n_agents": N_AGENTS,
            "n_local_data_qubits": len(DATA_WIRES),
            "n_total_qubits": 4,
            **variant_metadata,
            **tune_meta,
        },
    )
    system.metadata.update(_block_stats(system))
    return system


def _build_systems() -> Dict[str, LinearSystemData]:
    systems: Dict[str, LinearSystemData] = {}

    for fill_scale, key in ((0.0, "ising_zero"), (0.5, "ising_fill050"), (1.0, "ising_fill100")):
        raw_gates, raw_coeffs, raw_b_gates = _build_ising_raw(fill_scale)
        systems[key] = _build_system(
            name=key,
            raw_gates=raw_gates,
            raw_coeffs=raw_coeffs,
            raw_b_gates=raw_b_gates,
            variant_metadata={
                "family": "ising",
                "fill_scale": float(fill_scale),
                "description": "4x4 Ising-inspired system with zero-block filling on the anti-diagonal pair",
            },
            tune_to_cond50=True,
        )

    eq1_gates, eq1_coeffs, eq1_b_gates = _build_eq1_raw()
    eq1_system = _build_system(
        name="eq1_full_k196",
        raw_gates=eq1_gates,
        raw_coeffs=eq1_coeffs,
        raw_b_gates=eq1_b_gates,
        variant_metadata={
            "family": "eq1",
            "source_problem": "problems/static_ops_16agents_eq1_kappa196.py",
            "description": "Fully active heterogeneous 4x4 control problem kept in its original eq1_k196 form",
        },
        tune_to_cond50=False,
    )
    systems["eq1_full_k196"] = eq1_system
    systems["eq1_full_cond50"] = eq1_system

    return systems


SYSTEMS = _build_systems()
SYSTEM = SYSTEMS["ising_zero"]
DATA_WIRES_BY_SYSTEM = {key: list(DATA_WIRES) for key in SYSTEMS}

RAW_SUMMARY = {
    key: {
        "metadata": system.metadata,
    }
    for key, system in SYSTEMS.items()
}


if __name__ == "__main__":
    print("=== block-sparsity 4x4 2q diagnostic systems ===")
    for key, system in SYSTEMS.items():
        spectrum = system.metadata["spectrum"]
        print(
            f"{key}: lambda_min={spectrum['lambda_min']:.8f}, "
            f"lambda_max={spectrum['lambda_max']:.8f}, "
            f"cond={spectrum['condition_number']:.8f}, "
            f"zero_blocks={system.metadata['zero_block_count']}"
        )
