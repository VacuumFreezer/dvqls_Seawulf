"""
12-qubit real cluster-state linear system in a distributed 2x2 block form.

We solve
    A |x> = |b>
with
    |b> = |C_12>
    A = alpha I - J (Z1 X2 Z3 + Z4 X5 Z6 + Z7 X8 Z9 + Z10 X11 Z12).

This module exposes the same public interface used by the distributed scripts:
- LinearSystemData
- SYSTEM / SYSTEMS
- DATA_WIRES / DATA_WIRES_BY_SYSTEM
- RAW_GATES / RAW_COEFFS / RAW_B_GATES

Block decomposition
-------------------
We choose global qubit 2 as the 1-bit block index. The remaining 11 qubits are
kept locally on each agent, so the 12-qubit operator becomes a 2x2 block system
with 11 local qubits per agent.

With this choice:
- the term Z1 X2 Z3 becomes the real off-diagonal block term -J Z1 Z3,
- the remaining 3 ZXZ strings stay inside the diagonal blocks,
- the 1D cluster state row blocks are
    |C_12> = ( |0> |G_removed> + |1> Z1 Z3 |G_removed> ) / sqrt(2),
  where |G_removed> is the graph state on qubits {1,3,4,...,12}.

Spectrum
--------
All 4 non-identity Pauli strings commute and square to identity, so the spectrum is
available analytically. We choose
    alpha = 0.51,
    J = 0.1225,
which gives
    lambda_max = alpha + 4J = 1,
    lambda_min = alpha - 4J = 1/50,
and therefore cond(A) = 50.

Both A and |b> are real in the computational basis.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pennylane as qml


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


def _cluster_removed_q2_prep(data_wires: Sequence[int]):
    for w in data_wires:
        qml.Hadamard(wires=w)
    for left, right in zip(data_wires[1:-1], data_wires[2:]):
        qml.CZ(wires=[left, right])


def _b_row0_factory(data_wires: Sequence[int]):
    def gate():
        _cluster_removed_q2_prep(data_wires)

    return gate


def _b_row1_factory(data_wires: Sequence[int]):
    z_left = data_wires[GLOBAL_TO_LOCAL[1]]
    z_right = data_wires[GLOBAL_TO_LOCAL[3]]

    def gate():
        _cluster_removed_q2_prep(data_wires)
        qml.PauliZ(wires=z_left)
        qml.PauliZ(wires=z_right)

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
                for c, m in zip(coeffs, mats):
                    combined = combined + c * m
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
        all_rows = []
        for sys_id in range(self.n):
            all_rows.append(self.get_b_vectors(sys_id)[0])
        return np.concatenate(all_rows)

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
        local_paulis = {
            GLOBAL_TO_LOCAL[g]: p for g, p in pauli_map.items() if g != INDEX_GLOBAL_QUBIT
        }
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
    name="cluster12_2x2_q2_index",
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
