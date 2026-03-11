"""Qiskit-native 30-qubit cluster-state linear system in a 2x2 distributed block form."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PauliWord:
    """Local Pauli word on the 29-qubit agent register."""

    ops: Tuple[Tuple[int, str], ...]
    label: str

    @property
    def qubits(self) -> Tuple[int, ...]:
        return tuple(q for q, _ in self.ops)


@dataclass(frozen=True)
class BStateSpec:
    """Specification for the local right-hand-side preparation unitary."""

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


N_TOTAL_QUBITS = 30
INDEX_GLOBAL_QUBIT = 2
N_DATA_QUBITS_2X2 = N_TOTAL_QUBITS - 1
N_AGENTS_2X2 = 2

DATA_WIRES_2X2 = list(range(N_DATA_QUBITS_2X2))
LOCAL_GLOBAL_QUBITS = [1] + list(range(3, N_TOTAL_QUBITS + 1))
GLOBAL_TO_LOCAL = {g: i for i, g in enumerate(LOCAL_GLOBAL_QUBITS)}

ALPHA = 0.51
J = 0.06
h = 0.038

ZXZ_TRIPLES = [(1, 2, 3), (7, 8, 9), (13, 14, 15), (19, 20, 21), (25, 26, 27)]
ZYZ_TRIPLES = [(4, 5, 6), (10, 11, 12), (16, 17, 18), (22, 23, 24), (28, 29, 30)]

GLOBAL_TERMS: List[Tuple[float, Dict[int, str], str]] = [(ALPHA, {}, "I")]
for a, b, c in ZXZ_TRIPLES:
    GLOBAL_TERMS.append((-J, {a: "Z", b: "X", c: "Z"}, f"Z{a}X{b}Z{c}"))
for a, b, c in ZYZ_TRIPLES:
    GLOBAL_TERMS.append((h, {a: "Z", b: "Y", c: "Z"}, f"Z{a}Y{b}Z{c}"))


def _distinct_eigenvalues() -> List[float]:
    sums = [-5, -3, -1, 1, 3, 5]
    vals = sorted({ALPHA - J * sx + h * sy for sx in sums for sy in sums})
    return vals


DISTINCT_EIGENVALUES = _distinct_eigenvalues()
SPECTRUM_INFO = {
    "lambda_min": float(ALPHA - 5.0 * J - 5.0 * h),
    "lambda_max": float(ALPHA + 5.0 * J + 5.0 * h),
    "condition_number": float((ALPHA + 5.0 * J + 5.0 * h) / (ALPHA - 5.0 * J - 5.0 * h)),
    "n_distinct_eigenvalues": len(DISTINCT_EIGENVALUES),
    "distinct_eigenvalues": DISTINCT_EIGENVALUES,
}


def _make_local_pauli_word(local_ops: Dict[int, str], label: str) -> PauliWord:
    items = tuple(sorted((int(k), str(v)) for k, v in local_ops.items()))
    return PauliWord(items, label)


# Delete global qubit 2 to obtain the 29-qubit local register.
# The row-0 RHS is the chain graph state on local qubits 1..28 with an isolated local qubit 0.
B_ROW0 = BStateSpec(row_id=0, z_after_prep=(), label="cluster_removed_q2")
B_ROW1 = BStateSpec(row_id=1, z_after_prep=(GLOBAL_TO_LOCAL[1], GLOBAL_TO_LOCAL[3]), label="cluster_removed_q2_z1z3")


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
        elif index_pauli == "Z":
            diag00_gates.append(word)
            diag00_coeffs.append(float(coeff))
            diag11_gates.append(word)
            diag11_coeffs.append(float(-coeff))
        elif index_pauli == "X":
            off01_gates.append(word)
            off01_coeffs.append(float(coeff))
            off10_gates.append(word)
            off10_coeffs.append(float(coeff))
        elif index_pauli == "Y":
            raise ValueError("Index qubit 2 must not carry PauliY in this real block decomposition.")
        else:
            raise ValueError(f"Unsupported index Pauli: {index_pauli}")

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
    name="cluster30_2x2_q2_index_qiskit",
    metadata={
        "n_total_qubits": N_TOTAL_QUBITS,
        "index_global_qubit": INDEX_GLOBAL_QUBIT,
        "local_global_qubits": list(LOCAL_GLOBAL_QUBITS),
        "zxz_triples": list(ZXZ_TRIPLES),
        "zyz_triples": list(ZYZ_TRIPLES),
        "global_term_count": len(GLOBAL_TERMS),
        "block_term_counts": BLOCK_TERM_COUNTS_2X2,
        "spectrum": dict(SPECTRUM_INFO),
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
    print("=== cluster30 2x2 qiskit summary ===")
    print(f"system name: {SYSTEM.name}")
    print(f"n agents: {SYSTEM.n}")
    print(f"local data qubits: {SYSTEM.n_data_qubits}")
    print(f"block term counts: {BLOCK_TERM_COUNTS_2X2}")
    print(f"J = {J}, h = {h}, alpha = {ALPHA}")
    print(f"lambda_min = {SPECTRUM_INFO['lambda_min']}")
    print(f"lambda_max = {SPECTRUM_INFO['lambda_max']}")
    print(f"condition number = {SPECTRUM_INFO['condition_number']}")
