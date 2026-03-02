"""
Ising problem formulations for three algorithmic settings
--------------------------------------------------------
This module provides three formulations that solve the same linear system A x = b:
1) distributed 4x4 block partition (4 agents, 7 local qubits)
2) distributed 2x2 block partition (2 agents, 8 local qubits)
3) centralized direct Pauli-word formulation (9 total qubits)

Compatibility notes:
- Default exports SYSTEM / DATA_WIRES / N_DATA_QUBITS / N_AGENTS keep the 4x4 setup,
  so legacy scripts continue to work unchanged.
- Additional formulations are available through SYSTEMS and CENTRALIZED_PROBLEMS.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import pennylane as qml
from pennylane import numpy as np


# Shared Ising parameters
J = 0.1
eta = 10.43
zeta = 20.45

# Global qubit structure shared by all formulations
N_INDEX_QUBITS = 2
N_DATA_QUBITS_4X4 = 8
N_TOTAL_QUBITS = N_INDEX_QUBITS + N_DATA_QUBITS_4X4  # 9
N_DATA_QUBITS_2X2 = N_TOTAL_QUBITS - 1  # 8 (= 1 coarse index + 7 data)

DATA_WIRES_4X4 = list(range(N_DATA_QUBITS_4X4))
DATA_WIRES_2X2 = list(range(N_DATA_QUBITS_2X2))

N_AGENTS_4X4 = 4
N_AGENTS_2X2 = 2


class LinearSystemData:
    def __init__(
        self,
        gates_grid,
        coeffs_grid,
        b_gates_grid,
        *,
        data_wires: Sequence[int],
        b_weights_grid=None,
        zeta: float = 1.0,
        name: str = "",
    ):
        self.n = len(gates_grid)
        self.gates_grid = gates_grid
        self.b_gates = b_gates_grid
        self.data_wires = list(data_wires)
        self.n_data_qubits = len(self.data_wires)
        self.zeta = float(zeta)
        self.name = str(name)

        if b_weights_grid is None:
            b_weights_grid = [[1.0 for _ in range(self.n)] for __ in range(self.n)]
        self.b_weights = [list(map(float, row)) for row in b_weights_grid]

        self.coeffs = []
        self.ops = []
        for i in range(self.n):
            row_coeffs = []
            row_ops = []
            for j in range(self.n):
                g_list = gates_grid[i][j]
                c_list = list(coeffs_grid[i][j])
                if len(c_list) != len(g_list):
                    raise ValueError(
                        f"Coeff length mismatch at ({i},{j}): "
                        f"len(coeffs)={len(c_list)} vs len(gates)={len(g_list)}"
                    )
                if self.zeta != 1.0:
                    c_list = [c / self.zeta for c in c_list]
                row_coeffs.append(c_list)
                row_ops.append(self._make_wrapper(g_list))
            self.coeffs.append(row_coeffs)
            self.ops.append(row_ops)

        self._mat_cache: Dict[int, np.ndarray] = {}

    def _make_wrapper(self, gate_factories):
        def wrapper(l):
            return gate_factories[int(l)]()

        return wrapper

    def _as_matrix(self, gate_fn: Callable):
        key = id(gate_fn)
        if key in self._mat_cache:
            return self._mat_cache[key]

        def qfunc():
            gate_fn()

        m = np.array(qml.matrix(qfunc, wire_order=self.data_wires)())
        self._mat_cache[key] = m
        return m

    def get_global_matrix(self):
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
        u_gates = self.b_gates[sys_id]
        u_mats = [self._as_matrix(u) for u in u_gates]

        dim = 2 ** self.n_data_qubits
        ket0 = np.zeros(dim, dtype=float)
        ket0[0] = 1.0

        b_vecs = []
        for agent_id, m in enumerate(u_mats):
            w = self.b_weights[sys_id][agent_id]
            b_vecs.append(w * (m @ ket0))

        b_total = sum(b_vecs)
        return (b_total, *b_vecs)

    def get_global_b_vector(self):
        all_b_sums = []
        for sys_id in range(self.n):
            all_b_sums.append(self.get_b_vectors(sys_id)[0])
        return np.concatenate(all_b_sums)

    def get_b_op(self, sys_id: int, agent_id: int):
        return self.b_gates[sys_id][agent_id]


# -----------------------------
# 4x4 distributed formulation
# -----------------------------
def _build_4x4_system() -> Tuple[LinearSystemData, list, list, list]:
    data_wires = DATA_WIRES_4X4

    def i_op():
        return qml.Identity(wires=data_wires[0])

    def x_op(k: int):
        return lambda k=k: qml.PauliX(wires=data_wires[k])

    def z_op(k: int):
        return lambda k=k: qml.PauliZ(wires=data_wires[k])

    def zz_op(k: int):
        return lambda k=k: qml.prod(
            qml.PauliZ(wires=data_wires[k]), qml.PauliZ(wires=data_wires[k + 1])
        )

    z0 = z_op(0)

    def make_d_block(s01: float, s12: float):
        gates: List[Callable[[], qml.operation.Operator]] = []
        coeffs: List[float] = []

        gates.append(i_op)
        coeffs.append(eta + s01 * J)

        gates.append(z0)
        coeffs.append(s12 * J)

        for k in range(N_DATA_QUBITS_4X4):
            gates.append(x_op(k))
            coeffs.append(1.0)

        for k in range(N_DATA_QUBITS_4X4 - 1):
            gates.append(zz_op(k))
            coeffs.append(J)

        return gates, coeffs

    d00_g, d00_c = make_d_block(s01=+1.0, s12=+1.0)
    d01_g, d01_c = make_d_block(s01=-1.0, s12=-1.0)
    d10_g, d10_c = make_d_block(s01=-1.0, s12=+1.0)
    d11_g, d11_c = make_d_block(s01=+1.0, s12=-1.0)

    id_g, id_c = [i_op], [1.0]
    zr_g, zr_c = [i_op], [0.0]

    raw_gates = [
        [d00_g, id_g, id_g, zr_g],
        [id_g, d01_g, zr_g, id_g],
        [id_g, zr_g, d10_g, id_g],
        [zr_g, id_g, id_g, d11_g],
    ]

    raw_coeffs = [
        [d00_c, id_c, id_c, zr_c],
        [id_c, d01_c, zr_c, id_c],
        [id_c, zr_c, d10_c, id_c],
        [zr_c, id_c, id_c, d11_c],
    ]

    def h_all():
        return qml.prod(*[qml.Hadamard(wires=w) for w in data_wires])

    raw_b_gates = [[h_all for _ in range(N_AGENTS_4X4)] for __ in range(N_AGENTS_4X4)]

    system = LinearSystemData(
        raw_gates,
        raw_coeffs,
        raw_b_gates,
        data_wires=data_wires,
        b_weights_grid=[[1.0 for _ in range(N_AGENTS_4X4)] for __ in range(N_AGENTS_4X4)],
        zeta=zeta,
        name="4x4",
    )
    return system, raw_gates, raw_coeffs, raw_b_gates


# -----------------------------
# 2x2 distributed formulation
# -----------------------------
def _build_2x2_system() -> Tuple[LinearSystemData, list, list, list]:
    data_wires = DATA_WIRES_2X2
    coarse_wire = data_wires[0]
    local_data_wires = data_wires[1:]

    def i_op():
        return qml.Identity(wires=coarse_wire)

    def x_coarse():
        return qml.PauliX(wires=coarse_wire)

    def z_coarse():
        return qml.PauliZ(wires=coarse_wire)

    def x_data(k: int):
        return lambda k=k: qml.PauliX(wires=local_data_wires[k])

    def zz_data(k: int):
        return lambda k=k: qml.prod(
            qml.PauliZ(wires=local_data_wires[k]),
            qml.PauliZ(wires=local_data_wires[k + 1]),
        )

    def z_coarse_z_data0():
        return qml.prod(
            qml.PauliZ(wires=coarse_wire),
            qml.PauliZ(wires=local_data_wires[0]),
        )

    def make_diag_block(sign_zcoarse_i: float):
        gates: List[Callable[[], qml.operation.Operator]] = []
        coeffs: List[float] = []

        gates.append(i_op)
        coeffs.append(eta)

        gates.append(x_coarse)
        coeffs.append(1.0)

        for k in range(N_DATA_QUBITS_4X4):
            gates.append(x_data(k))
            coeffs.append(1.0)

        for k in range(N_DATA_QUBITS_4X4 - 1):
            gates.append(zz_data(k))
            coeffs.append(J)

        gates.append(z_coarse)
        coeffs.append(sign_zcoarse_i * J)

        gates.append(z_coarse_z_data0)
        coeffs.append(J)

        return gates, coeffs

    b00_g, b00_c = make_diag_block(sign_zcoarse_i=+1.0)
    b11_g, b11_c = make_diag_block(sign_zcoarse_i=-1.0)

    id_g, id_c = [i_op], [1.0]

    raw_gates = [
        [b00_g, id_g],
        [id_g, b11_g],
    ]

    raw_coeffs = [
        [b00_c, id_c],
        [id_c, b11_c],
    ]

    def h_all():
        return qml.prod(*[qml.Hadamard(wires=w) for w in data_wires])

    raw_b_gates = [[h_all for _ in range(N_AGENTS_2X2)] for __ in range(N_AGENTS_2X2)]

    # Scale chosen so global b exactly matches the 4x4 formulation.
    per_agent_b_weight = float(2.0 * np.sqrt(2.0))
    b_weights = [
        [per_agent_b_weight for _ in range(N_AGENTS_2X2)] for __ in range(N_AGENTS_2X2)
    ]

    system = LinearSystemData(
        raw_gates,
        raw_coeffs,
        raw_b_gates,
        data_wires=data_wires,
        b_weights_grid=b_weights,
        zeta=zeta,
        name="2x2",
    )
    return system, raw_gates, raw_coeffs, raw_b_gates


SYSTEM_4X4, RAW_GATES_4X4, RAW_COEFFS_4X4, RAW_B_GATES_4X4 = _build_4x4_system()
SYSTEM_2X2, RAW_GATES_2X2, RAW_COEFFS_2X2, RAW_B_GATES_2X2 = _build_2x2_system()


# -----------------------------
# Centralized direct formulation
# -----------------------------
_PAULI = {
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}


def _single_pauli_word(n: int, wire: int, pauli: str) -> Tuple[str, ...]:
    chars = ["I"] * n
    chars[int(wire)] = pauli
    return tuple(chars)


def _double_pauli_word(n: int, wire0: int, wire1: int, pauli0: str, pauli1: str) -> Tuple[str, ...]:
    chars = ["I"] * n
    chars[int(wire0)] = pauli0
    chars[int(wire1)] = pauli1
    return tuple(chars)


def _word_to_dense(word: Tuple[str, ...]) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    for p in word:
        out = np.kron(out, _PAULI[p])
    return out


def _dense_from_terms(n_qubits: int, terms: Sequence[Tuple[float, Tuple[str, ...]]]) -> np.ndarray:
    dim = 2 ** n_qubits
    out = np.zeros((dim, dim), dtype=np.complex128)
    for coeff, word in terms:
        out = out + float(coeff) * _word_to_dense(word)
    return out


def _build_centralized_terms() -> List[Tuple[float, Tuple[str, ...]]]:
    n = N_TOTAL_QUBITS
    inv_zeta = 1.0 / float(zeta)
    terms: List[Tuple[float, Tuple[str, ...]]] = []

    # eta * I
    terms.append((eta * inv_zeta, tuple("I" for _ in range(n))))

    # X on index qubits q0,q1
    terms.append((1.0 * inv_zeta, _single_pauli_word(n, 0, "X")))
    terms.append((1.0 * inv_zeta, _single_pauli_word(n, 1, "X")))

    # X on data qubits d0..d6 (global wires 2..8)
    for w in range(2, n):
        terms.append((1.0 * inv_zeta, _single_pauli_word(n, w, "X")))

    # J * Z_q0 Z_q1 and J * Z_q1 Z_d0
    terms.append((J * inv_zeta, _double_pauli_word(n, 0, 1, "Z", "Z")))
    terms.append((J * inv_zeta, _double_pauli_word(n, 1, 2, "Z", "Z")))

    # J * sum Z_dk Z_d(k+1)
    for w in range(2, n - 1):
        terms.append((J * inv_zeta, _double_pauli_word(n, w, w + 1, "Z", "Z")))

    return terms


CENTRALIZED_TERMS = _build_centralized_terms()
A_CENTRALIZED = _dense_from_terms(N_TOTAL_QUBITS, CENTRALIZED_TERMS)
B_CENTRALIZED = np.array(SYSTEM_4X4.get_global_b_vector(), dtype=np.complex128)


# -----------------------------
# Cross-formulation consistency
# -----------------------------
A_4X4 = np.array(SYSTEM_4X4.get_global_matrix(), dtype=np.complex128)
A_2X2 = np.array(SYSTEM_2X2.get_global_matrix(), dtype=np.complex128)
B_4X4 = np.array(SYSTEM_4X4.get_global_b_vector(), dtype=np.complex128)
B_2X2 = np.array(SYSTEM_2X2.get_global_b_vector(), dtype=np.complex128)

CONSISTENCY = {
    "a_4x4_vs_2x2_max_abs_diff": float(np.max(np.abs(A_4X4 - A_2X2))),
    "a_4x4_vs_2x2_fro_diff": float(np.linalg.norm(A_4X4 - A_2X2)),
    "a_4x4_vs_2x2_allclose": bool(np.allclose(A_4X4, A_2X2, atol=1.0e-12, rtol=0.0)),
    "a_4x4_vs_centralized_max_abs_diff": float(np.max(np.abs(A_4X4 - A_CENTRALIZED))),
    "a_4x4_vs_centralized_fro_diff": float(np.linalg.norm(A_4X4 - A_CENTRALIZED)),
    "a_4x4_vs_centralized_allclose": bool(np.allclose(A_4X4, A_CENTRALIZED, atol=1.0e-12, rtol=0.0)),
    "b_4x4_vs_2x2_max_abs_diff": float(np.max(np.abs(B_4X4 - B_2X2))),
    "b_4x4_vs_2x2_l2_diff": float(np.linalg.norm(B_4X4 - B_2X2)),
    "b_4x4_vs_2x2_allclose": bool(np.allclose(B_4X4, B_2X2, atol=1.0e-12, rtol=0.0)),
    "b_4x4_vs_centralized_max_abs_diff": float(np.max(np.abs(B_4X4 - B_CENTRALIZED))),
    "b_4x4_vs_centralized_l2_diff": float(np.linalg.norm(B_4X4 - B_CENTRALIZED)),
    "b_4x4_vs_centralized_allclose": bool(np.allclose(B_4X4, B_CENTRALIZED, atol=1.0e-12, rtol=0.0)),
}


# Public maps
SYSTEMS = {
    "4x4": SYSTEM_4X4,
    "2x2": SYSTEM_2X2,
}

DATA_WIRES_BY_SYSTEM = {
    "4x4": DATA_WIRES_4X4,
    "2x2": DATA_WIRES_2X2,
}

CENTRALIZED_PROBLEMS = {
    "centralized": {
        "name": "ising_q9_direct",
        "n_total_qubits": N_TOTAL_QUBITS,
        "n_index_qubits": N_INDEX_QUBITS,
        "n_data_qubits": N_DATA_QUBITS_4X4,
        "a_matrix": A_CENTRALIZED,
        "b_vector": B_CENTRALIZED,
        "terms": CENTRALIZED_TERMS,
        "reference_system_key": "4x4",
    }
}


def get_system(system_key: str = "4x4") -> LinearSystemData:
    key = str(system_key)
    if key not in SYSTEMS:
        raise KeyError(f"Unknown system key `{key}`. Available: {sorted(SYSTEMS.keys())}")
    return SYSTEMS[key]


def get_data_wires(system_key: str = "4x4") -> List[int]:
    key = str(system_key)
    if key not in DATA_WIRES_BY_SYSTEM:
        raise KeyError(
            f"Unknown system key `{key}` for DATA_WIRES. "
            f"Available: {sorted(DATA_WIRES_BY_SYSTEM.keys())}"
        )
    return list(DATA_WIRES_BY_SYSTEM[key])


# Backward-compatible default exports (4x4)
DEFAULT_SYSTEM_KEY = "4x4"
SYSTEM = SYSTEM_4X4
DATA_WIRES = DATA_WIRES_4X4
N_DATA_QUBITS = N_DATA_QUBITS_4X4
N_AGENTS = N_AGENTS_4X4

RAW_GATES = RAW_GATES_4X4
RAW_COEFFS = RAW_COEFFS_4X4
RAW_B_GATES = RAW_B_GATES_4X4


if __name__ == "__main__":
    print("=== static_ops_3net_Ising consistency ===")
    for k in sorted(CONSISTENCY.keys()):
        print(f"{k}: {CONSISTENCY[k]}")
    print("\nA_4x4 shape:", A_4X4.shape)
    print("A_2x2 shape:", A_2X2.shape)
    print("A_centralized shape:", A_CENTRALIZED.shape)
