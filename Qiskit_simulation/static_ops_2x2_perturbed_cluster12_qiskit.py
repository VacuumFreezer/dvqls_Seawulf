"""Qiskit-native 12-qubit perturbed cluster benchmark with qubit 1 as the block index."""

from __future__ import annotations

from itertools import product
from typing import Dict, List, Tuple

import numpy as np

from .static_ops_2x2_cluster30_qiskit import BStateSpec, LinearSystemDataQiskit, PauliWord


N_TOTAL_QUBITS = 12
INDEX_GLOBAL_QUBIT = 1
N_DATA_QUBITS_2X2 = N_TOTAL_QUBITS - 1
N_AGENTS_2X2 = 2

DATA_WIRES_2X2 = list(range(N_DATA_QUBITS_2X2))
LOCAL_GLOBAL_QUBITS = list(range(2, N_TOTAL_QUBITS + 1))
GLOBAL_TO_LOCAL = {g: i for i, g in enumerate(LOCAL_GLOBAL_QUBITS)}
LOCAL_LAST_WIRE = GLOBAL_TO_LOCAL[N_TOTAL_QUBITS]

EPSILON = 0.01
ALPHA = 0.51
STABILIZER_COEFF = 49.0 / 400.0 - EPSILON / 4.0

STABILIZER_INDEX_SET = (1, 4, 7, 10)
GLOBAL_TERMS: List[Tuple[float, Dict[int, str], str]] = [
    (ALPHA, {}, "I"),
    (STABILIZER_COEFF, {1: "X", 2: "Z"}, "K_1"),
    (STABILIZER_COEFF, {3: "Z", 4: "X", 5: "Z"}, "K_4"),
    (STABILIZER_COEFF, {6: "Z", 7: "X", 8: "Z"}, "K_7"),
    (STABILIZER_COEFF, {9: "Z", 10: "X", 11: "Z"}, "K_10"),
    (EPSILON, {12: "Z"}, "Z_12"),
]

COMMUTING_TERM_COEFFS = (STABILIZER_COEFF,) * len(STABILIZER_INDEX_SET) + (EPSILON,)
DISTINCT_EIGENVALUES = sorted(
    {
        ALPHA + sum(sign * coeff for sign, coeff in zip(sign_pattern, COMMUTING_TERM_COEFFS))
        for sign_pattern in product((-1.0, 1.0), repeat=len(COMMUTING_TERM_COEFFS))
    }
)
SPECTRUM_INFO = {
    "epsilon": float(EPSILON),
    "lambda_min": 1.0 / 50.0,
    "lambda_max": 1.0,
    "condition_number": 50.0,
    "n_distinct_eigenvalues": len(DISTINCT_EIGENVALUES),
    "distinct_eigenvalues": DISTINCT_EIGENVALUES,
    "b_is_eigenvector": False,
}

CLUSTER_SCAFFOLD_EDGES_LOCAL = tuple((idx, idx + 1) for idx in range(N_DATA_QUBITS_2X2 - 1))
EXACT_ALPHA = float((1.0 - EPSILON) / (1.0 - 2.0 * EPSILON))
EXACT_BETA = float(-EPSILON / (1.0 - 2.0 * EPSILON))
EXACT_HALF_NORM = float(np.sqrt((EXACT_ALPHA**2 + EXACT_BETA**2) / 2.0))
EXACT_LAST_WIRE_ANGLE = float(np.pi / 2.0 - 2.0 * np.arctan(EPSILON / (1.0 - EPSILON)))


def _make_local_pauli_word(local_ops: Dict[int, str], label: str) -> PauliWord:
    return PauliWord(tuple(sorted((int(k), str(v)) for k, v in local_ops.items())), label)


B_ROW0 = BStateSpec(row_id=0, z_after_prep=(), label="cluster_removed_q1")
B_ROW1 = BStateSpec(row_id=1, z_after_prep=(GLOBAL_TO_LOCAL[2],), label="cluster_removed_q1_z2")


def _decompose_global_terms_to_2x2() -> Tuple[List[List[List[PauliWord]]], List[List[np.ndarray]]]:
    diag00_gates: List[PauliWord] = []
    diag00_coeffs: List[float] = []
    diag11_gates: List[PauliWord] = []
    diag11_coeffs: List[float] = []
    off01_gates: List[PauliWord] = []
    off01_coeffs: List[float] = []
    off10_gates: List[PauliWord] = []
    off10_coeffs: List[float] = []

    for coeff, pauli_map, label in GLOBAL_TERMS:
        index_pauli = pauli_map.get(INDEX_GLOBAL_QUBIT, "I")
        local_paulis = {GLOBAL_TO_LOCAL[g]: p for g, p in pauli_map.items() if g != INDEX_GLOBAL_QUBIT}
        word = _make_local_pauli_word(local_paulis, label)

        if index_pauli == "I":
            diag00_gates.append(word)
            diag00_coeffs.append(float(coeff))
            diag11_gates.append(word)
            diag11_coeffs.append(float(coeff))
        elif index_pauli == "X":
            off01_gates.append(word)
            off01_coeffs.append(float(coeff))
            off10_gates.append(word)
            off10_coeffs.append(float(coeff))
        else:
            raise ValueError(f"Unexpected Pauli {index_pauli!r} on index qubit for this real benchmark")

    return (
        [[diag00_gates, off01_gates], [off10_gates, diag11_gates]],
        [
            [np.asarray(diag00_coeffs, dtype=np.float32), np.asarray(off01_coeffs, dtype=np.float32)],
            [np.asarray(off10_coeffs, dtype=np.float32), np.asarray(diag11_coeffs, dtype=np.float32)],
        ],
    )


RAW_GATES_2X2, RAW_COEFFS_2X2 = _decompose_global_terms_to_2x2()
RAW_B_SPECS_2X2 = [[B_ROW0, B_ROW0], [B_ROW1, B_ROW1]]
_PER_AGENT_B_WEIGHT = float(1.0 / (2.0 * np.sqrt(2.0)))
B_WEIGHTS_2X2 = [[_PER_AGENT_B_WEIGHT for _ in range(N_AGENTS_2X2)] for __ in range(N_AGENTS_2X2)]
BLOCK_TERM_COUNTS_2X2 = [[len(cell) for cell in row] for row in RAW_GATES_2X2]

SYSTEM_2X2 = LinearSystemDataQiskit(
    gates_grid=RAW_GATES_2X2,
    coeffs=RAW_COEFFS_2X2,
    b_specs=RAW_B_SPECS_2X2,
    data_wires=DATA_WIRES_2X2,
    b_weights=B_WEIGHTS_2X2,
    name="perturbed_cluster_state_12q_2x2_q1_index_qiskit",
    metadata={
        "n_total_qubits": N_TOTAL_QUBITS,
        "index_global_qubit": INDEX_GLOBAL_QUBIT,
        "local_global_qubits": list(LOCAL_GLOBAL_QUBITS),
        "global_term_count": len(GLOBAL_TERMS),
        "block_term_counts": BLOCK_TERM_COUNTS_2X2,
        "spectrum": dict(SPECTRUM_INFO),
        "real_problem": True,
        "recommended_ansatz_qiskit": "brickwall_ry_cz",
        "cluster_scaffold_edges_local": [list(edge) for edge in CLUSTER_SCAFFOLD_EDGES_LOCAL],
        "init_angle_fill": float(np.pi / 2.0),
        "init_sigma_target": EXACT_HALF_NORM,
        "agent_init_overrides": {
            0: {LOCAL_LAST_WIRE: EXACT_LAST_WIRE_ANGLE},
            1: {0: float(-np.pi / 2.0), LOCAL_LAST_WIRE: EXACT_LAST_WIRE_ANGLE},
        },
        "exact_alpha_coefficient": EXACT_ALPHA,
        "exact_beta_coefficient": EXACT_BETA,
        "exact_last_wire_angle": EXACT_LAST_WIRE_ANGLE,
        "design_note": (
            "weakly perturbed 1D cluster benchmark with exact shallow real solution and nonzero "
            "off-diagonal 2x2 blocks after splitting on qubit 1"
        ),
    },
)

SYSTEMS = {"2x2": SYSTEM_2X2}
SYSTEM = SYSTEM_2X2
DATA_WIRES = DATA_WIRES_2X2
N_DATA_QUBITS = N_DATA_QUBITS_2X2
N_AGENTS = N_AGENTS_2X2
RAW_GATES = RAW_GATES_2X2
RAW_COEFFS = RAW_COEFFS_2X2
RAW_B_SPECS = RAW_B_SPECS_2X2


def get_system(system_key: str = "2x2") -> LinearSystemDataQiskit:
    key = str(system_key)
    if key not in SYSTEMS:
        raise KeyError(f"Unknown system key {key!r}. Available: {sorted(SYSTEMS)}")
    return SYSTEMS[key]
