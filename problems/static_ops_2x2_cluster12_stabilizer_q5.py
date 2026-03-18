"""
12-qubit 2x2 distributed benchmark based on a truncated cluster Hamiltonian.

We solve
    A |x> = |b>
with
    |b> = |C_12>
    A = alpha I
        + c_2 K_2 + c_5 K_5 + c_8 K_8 + c_11 K_11
        + gamma X_2,
where K_j = Z_{j-1} X_j Z_{j+1} are 1D cluster-state stabilizers.

This variant chooses global qubit 5 as the 2x2 block index. That keeps the
problem nontrivial through the off-diagonal K_5 term while leaving the
perturbation X_2 inside the local subsystem instead of on the removed index
qubit.
"""

from __future__ import annotations

from itertools import product
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pennylane as qml


N_TOTAL_QUBITS = 12
INDEX_GLOBAL_QUBIT = 5
N_DATA_QUBITS_2X2 = N_TOTAL_QUBITS - 1
N_AGENTS_2X2 = 2

DATA_WIRES_2X2 = list(range(N_DATA_QUBITS_2X2))
LOCAL_GLOBAL_QUBITS = [g for g in range(1, N_TOTAL_QUBITS + 1) if g != INDEX_GLOBAL_QUBIT]
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

PERTURBATION_SUPPORT_GLOBALS = (1, 2, 3)


def _pauli_word_factory(local_ops: Dict[int, str], data_wires: Sequence[int]):
    local_ops = {int(k): str(v) for k, v in local_ops.items()}

    def gate():
        factors = []
        for local_wire in sorted(local_ops.keys()):
            label = local_ops[local_wire]
            wire = data_wires[local_wire]
            if label == "X":
                factors.append(qml.PauliX(wires=wire))
            elif label == "Y":
                factors.append(qml.PauliY(wires=wire))
            elif label == "Z":
                factors.append(qml.PauliZ(wires=wire))
            else:
                raise ValueError(f"Unsupported Pauli label: {label}")

        if not factors:
            return qml.Identity(wires=data_wires[0])
        if len(factors) == 1:
            return factors[0]
        return qml.prod(*factors)

    return gate


def _cluster_removed_index_prep(data_wires: Sequence[int]):
    for wire in data_wires:
        qml.Hadamard(wires=wire)

    for left_global, right_global in zip(LOCAL_GLOBAL_QUBITS[:-1], LOCAL_GLOBAL_QUBITS[1:]):
        if right_global - left_global != 1:
            continue
        qml.CZ(wires=[data_wires[GLOBAL_TO_LOCAL[left_global]], data_wires[GLOBAL_TO_LOCAL[right_global]]])


def _b_row0_factory(data_wires: Sequence[int]):
    def gate():
        _cluster_removed_index_prep(data_wires)

    return gate


def _removed_index_neighbors() -> tuple[int, ...]:
    neighbors = []
    if INDEX_GLOBAL_QUBIT > 1:
        neighbors.append(INDEX_GLOBAL_QUBIT - 1)
    if INDEX_GLOBAL_QUBIT < N_TOTAL_QUBITS:
        neighbors.append(INDEX_GLOBAL_QUBIT + 1)
    return tuple(neighbors)


CLUSTER_SCAFFOLD_EDGES_LOCAL = tuple(
    (GLOBAL_TO_LOCAL[left_global], GLOBAL_TO_LOCAL[right_global])
    for left_global, right_global in zip(LOCAL_GLOBAL_QUBITS[:-1], LOCAL_GLOBAL_QUBITS[1:])
    if right_global - left_global == 1
)


LOCAL_RY_SUPPORT_GLOBALS = tuple(sorted(set(PERTURBATION_SUPPORT_GLOBALS + _removed_index_neighbors())))
LOCAL_RY_SUPPORT_WIRES = tuple(GLOBAL_TO_LOCAL[g] for g in LOCAL_RY_SUPPORT_GLOBALS)


def _b_row1_factory(data_wires: Sequence[int]):
    neighbor_locals = tuple(GLOBAL_TO_LOCAL[g] for g in _removed_index_neighbors() if g in GLOBAL_TO_LOCAL)

    def gate():
        _cluster_removed_index_prep(data_wires)
        for local_wire in neighbor_locals:
            qml.PauliZ(wires=data_wires[local_wire])

    return gate


class LinearSystemData:
    def __init__(
        self,
        gates_grid,
        coeffs_grid,
        b_gates_grid,
        *,
        data_wires: Sequence[int],
        b_weights_grid=None,
        name: str = "",
        metadata: Dict[str, object] | None = None,
        max_dense_qubits: int = 16,
    ):
        self.n = len(gates_grid)
        self.gates_grid = gates_grid
        self.b_gates = b_gates_grid
        self.data_wires = list(data_wires)
        self.n_data_qubits = len(self.data_wires)
        self.name = str(name)
        self.max_dense_qubits = int(max_dense_qubits)
        self.supports_dense_validation = self.n_data_qubits <= self.max_dense_qubits
        self.metadata = dict(metadata or {})

        if b_weights_grid is None:
            b_weights_grid = [[1.0 for _ in range(self.n)] for __ in range(self.n)]
        self.b_weights = [list(map(float, row)) for row in b_weights_grid]

        self.coeffs = []
        self.ops = []
        for i in range(self.n):
            row_coeffs = []
            row_ops = []
            for j in range(self.n):
                g_list = list(gates_grid[i][j])
                c_list = list(coeffs_grid[i][j])
                if len(g_list) != len(c_list):
                    raise ValueError(
                        f"Coeff length mismatch at ({i},{j}): len(gates)={len(g_list)} vs len(coeffs)={len(c_list)}"
                    )
                row_coeffs.append(c_list)
                row_ops.append(self._make_wrapper(g_list))
            self.coeffs.append(row_coeffs)
            self.ops.append(row_ops)

        self._mat_cache: Dict[int, np.ndarray] = {}

    def _make_wrapper(self, gate_factories):
        def wrapper(l):
            return gate_factories[int(l)]()

        return wrapper

    def _dense_guard(self, purpose: str):
        if not self.supports_dense_validation:
            raise MemoryError(
                f"{purpose} is disabled for `{self.name}`: local dimension is 2^{self.n_data_qubits}, "
                "so dense matrix/vector construction is not practical. Use the distributed Hadamard-test "
                "representation directly."
            )

    def _as_matrix(self, gate_fn: Callable):
        key = id(gate_fn)
        if key in self._mat_cache:
            return self._mat_cache[key]

        def qfunc():
            gate_fn()

        mat = np.array(qml.matrix(qfunc, wire_order=self.data_wires)())
        self._mat_cache[key] = mat
        return mat

    def get_local_b_norms(self, sys_id: int) -> Tuple[float, ...]:
        return tuple(abs(float(w)) for w in self.b_weights[int(sys_id)])

    def get_global_matrix(self):
        self._dense_guard("get_global_matrix()")
        block_rows = []
        for i in range(self.n):
            block_cols = []
            for j in range(self.n):
                gates = self.gates_grid[i][j]
                coeffs = self.coeffs[i][j]
                mats = [self._as_matrix(g) for g in gates]
                combined = np.zeros_like(mats[0], dtype=complex)
                for coeff, mat in zip(coeffs, mats):
                    combined = combined + coeff * mat
                block_cols.append(combined)
            block_rows.append(block_cols)
        return np.block(block_rows)

    def get_b_vectors(self, sys_id: int):
        self._dense_guard("get_b_vectors()")
        u_gates = self.b_gates[int(sys_id)]
        u_mats = [self._as_matrix(u) for u in u_gates]

        dim = 2 ** self.n_data_qubits
        ket0 = np.zeros(dim, dtype=complex)
        ket0[0] = 1.0

        b_vecs = []
        for agent_id, mat in enumerate(u_mats):
            weight = float(self.b_weights[int(sys_id)][agent_id])
            b_vecs.append(weight * (mat @ ket0))

        b_total = sum(b_vecs)
        return (b_total, *b_vecs)

    def get_global_b_vector(self):
        self._dense_guard("get_global_b_vector()")
        return np.concatenate([self.get_b_vectors(sys_id)[0] for sys_id in range(self.n)])

    def get_b_op(self, sys_id: int, agent_id: int):
        return self.b_gates[int(sys_id)][int(agent_id)]


def _decompose_global_terms_to_2x2(data_wires: Sequence[int]):
    diag00_gates: List[Callable[[], qml.operation.Operator]] = []
    diag00_coeffs: List[float] = []
    diag11_gates: List[Callable[[], qml.operation.Operator]] = []
    diag11_coeffs: List[float] = []
    off01_gates: List[Callable[[], qml.operation.Operator]] = []
    off01_coeffs: List[float] = []
    off10_gates: List[Callable[[], qml.operation.Operator]] = []
    off10_coeffs: List[float] = []

    for coeff, pauli_map, _label in GLOBAL_TERMS:
        index_pauli = pauli_map.get(INDEX_GLOBAL_QUBIT, "I")
        local_paulis = {GLOBAL_TO_LOCAL[g]: p for g, p in pauli_map.items() if g != INDEX_GLOBAL_QUBIT}
        gate = _pauli_word_factory(local_paulis, data_wires)

        if index_pauli == "I":
            diag00_gates.append(gate)
            diag00_coeffs.append(float(coeff))
            diag11_gates.append(gate)
            diag11_coeffs.append(float(coeff))
        elif index_pauli == "X":
            off01_gates.append(gate)
            off01_coeffs.append(float(coeff))
            off10_gates.append(gate)
            off10_coeffs.append(float(coeff))
        else:
            raise ValueError(f"Unsupported Pauli on index qubit: {index_pauli}")

    return (
        [[diag00_gates, off01_gates], [off10_gates, diag11_gates]],
        [[diag00_coeffs, off01_coeffs], [off10_coeffs, diag11_coeffs]],
    )


RAW_GATES_2X2, RAW_COEFFS_2X2 = _decompose_global_terms_to_2x2(DATA_WIRES_2X2)

_b_row0 = _b_row0_factory(DATA_WIRES_2X2)
_b_row1 = _b_row1_factory(DATA_WIRES_2X2)
RAW_B_GATES_2X2 = [[_b_row0, _b_row0], [_b_row1, _b_row1]]

_PER_AGENT_B_WEIGHT = 1.0 / (2.0 * np.sqrt(2.0))
B_WEIGHTS_2X2 = [[_PER_AGENT_B_WEIGHT for _ in range(N_AGENTS_2X2)] for __ in range(N_AGENTS_2X2)]

BLOCK_TERM_COUNTS_2X2 = [[len(cell) for cell in row] for row in RAW_GATES_2X2]

SYSTEM_2X2 = LinearSystemData(
    RAW_GATES_2X2,
    RAW_COEFFS_2X2,
    RAW_B_GATES_2X2,
    data_wires=DATA_WIRES_2X2,
    b_weights_grid=B_WEIGHTS_2X2,
    name="cluster12_stabilizer_2x2_q5_index",
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
        "recommended_ansatz": "cluster_local_ry",
        "local_ry_support_wires": list(LOCAL_RY_SUPPORT_WIRES),
        "local_ry_support_globals": list(LOCAL_RY_SUPPORT_GLOBALS),
        "cluster_scaffold_edges_local": [list(edge) for edge in CLUSTER_SCAFFOLD_EDGES_LOCAL],
        "design_note": (
            "global index moved off the perturbation qubit so X_2 stays local while K_5 remains off-diagonal; "
            "the local RY ansatz support covers both the perturbation sector and the removed-index neighbors"
        ),
    },
    max_dense_qubits=16,
)

SYSTEMS = {"2x2": SYSTEM_2X2}
DATA_WIRES_BY_SYSTEM = {"2x2": DATA_WIRES_2X2}

SYSTEM = SYSTEM_2X2
DATA_WIRES = DATA_WIRES_2X2
N_DATA_QUBITS = N_DATA_QUBITS_2X2
N_AGENTS = N_AGENTS_2X2
RAW_GATES = RAW_GATES_2X2
RAW_COEFFS = RAW_COEFFS_2X2
RAW_B_GATES = RAW_B_GATES_2X2


def get_system(system_key: str = "2x2") -> LinearSystemData:
    key = str(system_key)
    if key not in SYSTEMS:
        raise KeyError(f"Unknown system key `{key}`. Available: {sorted(SYSTEMS.keys())}")
    return SYSTEMS[key]
