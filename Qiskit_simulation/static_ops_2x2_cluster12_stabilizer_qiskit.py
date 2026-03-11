"""Qiskit-native 12-qubit cluster benchmark with a small real X_2 perturbation.

We solve A|x> = |b> with
  |b> = |C_12>, the 12-qubit 1D cluster state,
and the real symmetric operator
  A = alpha I
      + c_2 K_2 + c_5 K_5 + c_8 K_8 + c_11 K_11
      + gamma X_2,
where K_j = Z_{j-1} X_j Z_{j+1}.

Design goals:
- |x> is real and close to |b>, but not proportional to |b>;
- the Pauli expansion stays sparse: 1 identity + 4 stabilizers + 1 perturbation;
- the spectrum remains exactly lambda_max = 1 and lambda_min = 1/50.

Choosing global qubit 2 as the block index yields a 2x2 system with 11 local qubits.
K_2 and X_2 become off-diagonal block terms; the other 3 stabilizers remain diagonal.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PauliWord:
    ops: Tuple[Tuple[int, str], ...]
    label: str

    @property
    def qubits(self) -> Tuple[int, ...]:
        return tuple(q for q, _ in self.ops)


@dataclass(frozen=True)
class BStateSpec:
    row_id: int
    z_after_prep: Tuple[int, ...]
    label: str


@dataclass
class LinearSystemDataQiskit:
    gates_grid: List[List[List[PauliWord]]]
    coeffs: List[List[np.ndarray]]
    b_specs: List[List[BStateSpec]]
    data_wires: Sequence[int]
    b_weights: List[List[float]]
    name: str
    metadata: Dict[str, object]

    def __post_init__(self):
        self.n = len(self.gates_grid)
        self.n_data_qubits = len(self.data_wires)

    def get_local_b_norms(self, sys_id: int) -> Tuple[float, ...]:
        return tuple(abs(float(x)) for x in self.b_weights[int(sys_id)])


N_TOTAL_QUBITS = 12
INDEX_GLOBAL_QUBIT = 2
N_DATA_QUBITS_2X2 = N_TOTAL_QUBITS - 1
N_AGENTS_2X2 = 2

DATA_WIRES_2X2 = list(range(N_DATA_QUBITS_2X2))
LOCAL_GLOBAL_QUBITS = [1] + list(range(3, N_TOTAL_QUBITS + 1))
GLOBAL_TO_LOCAL = {g: i for i, g in enumerate(LOCAL_GLOBAL_QUBITS)}

ALPHA = 0.51
COEFF_K2 = 0.1025
COEFF_K5 = 0.1225
COEFF_K8 = 0.1225
COEFF_K11 = 0.1225
COEFF_X2 = 0.02

STABILIZER_TRIPLES = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)]
GLOBAL_TERMS: List[Tuple[float, Dict[int, str], str]] = [
    (ALPHA, {}, "I"),
    (COEFF_K2, {1: "Z", 2: "X", 3: "Z"}, "K_2"),
    (COEFF_K5, {4: "Z", 5: "X", 6: "Z"}, "K_5"),
    (COEFF_K8, {7: "Z", 8: "X", 9: "Z"}, "K_8"),
    (COEFF_K11, {10: "Z", 11: "X", 12: "Z"}, "K_11"),
    (COEFF_X2, {2: "X"}, "X_2"),
]


COMMUTING_TERM_COEFFS = (COEFF_K2, COEFF_K5, COEFF_K8, COEFF_K11, COEFF_X2)
DISTINCT_EIGENVALUES = sorted(
    {
        ALPHA + sum(sign * coeff for sign, coeff in zip(sign_pattern, COMMUTING_TERM_COEFFS))
        for sign_pattern in product((-1.0, 1.0), repeat=len(COMMUTING_TERM_COEFFS))
    }
)
SPECTRUM_INFO = {
    "lambda_min": float(ALPHA - sum(COMMUTING_TERM_COEFFS)),
    "lambda_max": float(ALPHA + sum(COMMUTING_TERM_COEFFS)),
    "condition_number": float((ALPHA + sum(COMMUTING_TERM_COEFFS)) / (ALPHA - sum(COMMUTING_TERM_COEFFS))),
    "n_distinct_eigenvalues": len(DISTINCT_EIGENVALUES),
    "distinct_eigenvalues": DISTINCT_EIGENVALUES,
    "b_is_eigenvector": False,
}


def _make_local_pauli_word(local_ops: Dict[int, str], label: str) -> PauliWord:
    return PauliWord(tuple(sorted((int(k), str(v)) for k, v in local_ops.items())), label)


B_ROW0 = BStateSpec(row_id=0, z_after_prep=(), label="cluster12_removed_q2")
B_ROW1 = BStateSpec(row_id=1, z_after_prep=(GLOBAL_TO_LOCAL[1], GLOBAL_TO_LOCAL[3]), label="cluster12_removed_q2_z1z3")


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
        local_paulis = {
            GLOBAL_TO_LOCAL[g]: p for g, p in pauli_map.items() if g != INDEX_GLOBAL_QUBIT
        }
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
    name="cluster12_stabilizer_2x2_q2_index_qiskit",
    metadata={
        "n_total_qubits": N_TOTAL_QUBITS,
        "index_global_qubit": INDEX_GLOBAL_QUBIT,
        "local_global_qubits": list(LOCAL_GLOBAL_QUBITS),
        "stabilizer_triples": list(STABILIZER_TRIPLES),
        "global_term_count": len(GLOBAL_TERMS),
        "block_term_counts": BLOCK_TERM_COUNTS_2X2,
        "spectrum": dict(SPECTRUM_INFO),
        "real_problem": True,
        "exact_solution_relation": "x is real, close to b, but not proportional to b",
        "perturbation_term": "X_2",
        "term_coefficients": {
            "K_2": COEFF_K2,
            "K_5": COEFF_K5,
            "K_8": COEFF_K8,
            "K_11": COEFF_K11,
            "X_2": COEFF_X2,
        },
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


if __name__ == "__main__":
    print("=== cluster12 stabilizer 2x2 qiskit summary ===")
    print(f"system name: {SYSTEM.name}")
    print(f"n agents: {SYSTEM.n}")
    print(f"local data qubits: {SYSTEM.n_data_qubits}")
    print(f"block term counts: {BLOCK_TERM_COUNTS_2X2}")
    print(
        "coefficients:",
        {
            "K_2": COEFF_K2,
            "K_5": COEFF_K5,
            "K_8": COEFF_K8,
            "K_11": COEFF_K11,
            "X_2": COEFF_X2,
        },
    )
    print(f"lambda_min = {SPECTRUM_INFO['lambda_min']}")
    print(f"lambda_max = {SPECTRUM_INFO['lambda_max']}")
    print(f"condition number = {SPECTRUM_INFO['condition_number']}")
