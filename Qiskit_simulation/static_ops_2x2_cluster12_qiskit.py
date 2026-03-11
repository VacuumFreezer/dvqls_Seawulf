"""Qiskit-native 12-qubit cluster-state linear system in a 2x2 distributed block form.

We solve A|x> = |b> with
  |b> = |C_12>, the 12-qubit 1D cluster state,
and the real symmetric operator
  A = alpha I - J * (Z1 X2 Z3 + Z4 X5 Z6 + Z7 X8 Z9 + Z10 X11 Z12).

This gives exactly 5 Pauli terms total: 1 identity + 4 ZXZ terms.
Choosing global qubit 2 as the block index yields a 2x2 system with 11 local qubits.
Only Z1 X2 Z3 touches the index qubit, so the off-diagonal block contains a single Z1 Z3 term.
The remaining three ZXZ terms stay in the diagonal blocks.

All terms are real, so both A and |b> are real-valued in the computational basis.
The spectrum is analytic because the 4 ZXZ strings commute and square to identity.
We choose alpha=0.51 and J=0.1225 so that lambda_max = 1 and lambda_min = 1/50.
"""

from __future__ import annotations

from dataclasses import dataclass
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
J = 0.1225

ZXZ_TRIPLES = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)]
GLOBAL_TERMS: List[Tuple[float, Dict[int, str], str]] = [(ALPHA, {}, "I")]
for a, b, c in ZXZ_TRIPLES:
    GLOBAL_TERMS.append((-J, {a: "Z", b: "X", c: "Z"}, f"Z{a}X{b}Z{c}"))


DISTINCT_EIGENVALUES = sorted({ALPHA - J * s for s in (-4, -2, 0, 2, 4)})
SPECTRUM_INFO = {
    "lambda_min": float(ALPHA - 4.0 * J),
    "lambda_max": float(ALPHA + 4.0 * J),
    "condition_number": float((ALPHA + 4.0 * J) / (ALPHA - 4.0 * J)),
    "n_distinct_eigenvalues": len(DISTINCT_EIGENVALUES),
    "distinct_eigenvalues": DISTINCT_EIGENVALUES,
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
    name="cluster12_2x2_q2_index_qiskit",
    metadata={
        "n_total_qubits": N_TOTAL_QUBITS,
        "index_global_qubit": INDEX_GLOBAL_QUBIT,
        "local_global_qubits": list(LOCAL_GLOBAL_QUBITS),
        "zxz_triples": list(ZXZ_TRIPLES),
        "global_term_count": len(GLOBAL_TERMS),
        "block_term_counts": BLOCK_TERM_COUNTS_2X2,
        "spectrum": dict(SPECTRUM_INFO),
        "real_problem": True,
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
    print("=== cluster12 2x2 qiskit summary ===")
    print(f"system name: {SYSTEM.name}")
    print(f"n agents: {SYSTEM.n}")
    print(f"local data qubits: {SYSTEM.n_data_qubits}")
    print(f"block term counts: {BLOCK_TERM_COUNTS_2X2}")
    print(f"J = {J}, alpha = {ALPHA}")
    print(f"lambda_min = {SPECTRUM_INFO['lambda_min']}")
    print(f"lambda_max = {SPECTRUM_INFO['lambda_max']}")
    print(f"condition number = {SPECTRUM_INFO['condition_number']}")
