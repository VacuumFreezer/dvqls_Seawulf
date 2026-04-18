from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import pennylane as qml
from pennylane import numpy as np


J = 0.1
TARGET_LAMBDA_MAX = 1.0
TARGET_LAMBDA_MIN = 1.0 / 50.0

N_INDEX_QUBITS = 2
N_DATA_QUBITS_4X4 = 2
N_TOTAL_QUBITS = N_INDEX_QUBITS + N_DATA_QUBITS_4X4  # 4
N_AGENTS_4X4 = 4

DATA_WIRES_4X4 = list(range(N_DATA_QUBITS_4X4))


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
        metadata: dict | None = None,
    ):
        self.n = len(gates_grid)
        self.gates_grid = gates_grid
        self.b_gates = b_gates_grid
        self.data_wires = list(data_wires)
        self.n_data_qubits = len(self.data_wires)
        self.zeta = float(zeta)
        self.name = str(name)
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
        self._global_matrix_cache: np.ndarray | None = None

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

    def get_block_matrix(self, row_id: int, col_id: int) -> np.ndarray:
        gates = self.gates_grid[int(row_id)][int(col_id)]
        coeffs = self.coeffs[int(row_id)][int(col_id)]
        mats = [self._as_matrix(g) for g in gates]
        combined = np.zeros_like(mats[0], dtype=complex)
        for coeff, mat in zip(coeffs, mats):
            combined = combined + coeff * mat
        return combined

    def apply_block_operator(self, row_id: int, col_id: int, vec: np.ndarray) -> np.ndarray:
        return self.get_block_matrix(row_id, col_id) @ np.asarray(vec, dtype=np.complex128)

    def get_global_matrix(self):
        if self._global_matrix_cache is not None:
            return self._global_matrix_cache

        block_rows = []
        for i in range(self.n):
            block_cols = []
            for j in range(self.n):
                block_cols.append(self.get_block_matrix(i, j))
            block_rows.append(block_cols)
        self._global_matrix_cache = np.block(block_rows)
        return self._global_matrix_cache

    def get_b_vectors(self, sys_id: int):
        u_gates = self.b_gates[sys_id]
        u_mats = [self._as_matrix(u) for u in u_gates]

        dim = 2 ** self.n_data_qubits
        ket0 = np.zeros(dim, dtype=float)
        ket0[0] = 1.0

        b_vecs = []
        for agent_id, mat in enumerate(u_mats):
            weight = self.b_weights[sys_id][agent_id]
            b_vecs.append(weight * (mat @ ket0))

        b_total = sum(b_vecs)
        return (b_total, *b_vecs)

    def get_global_b_vector(self):
        all_b_sums = []
        for sys_id in range(self.n):
            all_b_sums.append(self.get_b_vectors(sys_id)[0])
        return np.concatenate(all_b_sums)

    def get_local_b_norms(self, sys_id: int) -> list[float]:
        _, *b_vecs = self.get_b_vectors(sys_id)
        return [float(np.linalg.norm(np.asarray(vec, dtype=np.complex128))) for vec in b_vecs]

    def true_solution_vector(self) -> np.ndarray:
        return np.linalg.solve(
            np.asarray(self.get_global_matrix(), dtype=np.complex128),
            np.asarray(self.get_global_b_vector(), dtype=np.complex128),
        )


_PAULI = {
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
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
    for pauli in word:
        out = np.kron(out, _PAULI[pauli])
    return out


def _dense_from_terms(n_qubits: int, terms: Sequence[Tuple[float, Tuple[str, ...]]]) -> np.ndarray:
    dim = 2 ** n_qubits
    out = np.zeros((dim, dim), dtype=np.complex128)
    for coeff, word in terms:
        out = out + float(coeff) * _word_to_dense(word)
    return out


def _build_base_terms(*, eta: float, zeta: float) -> List[Tuple[float, Tuple[str, ...]]]:
    n = N_TOTAL_QUBITS
    inv_zeta = 1.0 / float(zeta)
    terms: List[Tuple[float, Tuple[str, ...]]] = []

    terms.append((eta * inv_zeta, tuple("I" for _ in range(n))))

    for wire in range(n):
        terms.append((1.0 * inv_zeta, _single_pauli_word(n, wire, "X")))

    for wire in range(n - 1):
        terms.append((J * inv_zeta, _double_pauli_word(n, wire, wire + 1, "Z", "Z")))

    return terms


def _compute_tuned_coefficients() -> tuple[float, float, dict]:
    unshifted_terms = _build_base_terms(eta=0.0, zeta=1.0)
    h_dense = _dense_from_terms(N_TOTAL_QUBITS, unshifted_terms)
    evals = np.linalg.eigvalsh(h_dense)
    h_min = float(np.min(evals).real)
    h_max = float(np.max(evals).real)

    zeta = (h_max - h_min) / (TARGET_LAMBDA_MAX - TARGET_LAMBDA_MIN)
    eta = TARGET_LAMBDA_MAX * zeta - h_max

    tuned_terms = _build_base_terms(eta=eta, zeta=zeta)
    a_dense = _dense_from_terms(N_TOTAL_QUBITS, tuned_terms)
    tuned_evals = np.linalg.eigvalsh(a_dense)
    lam_min = float(np.min(tuned_evals).real)
    lam_max = float(np.max(tuned_evals).real)

    return eta, zeta, {
        "h_min": h_min,
        "h_max": h_max,
        "eta": float(eta),
        "zeta": float(zeta),
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "condition_number": float(lam_max / lam_min),
    }


ETA, ZETA, SPECTRUM_INFO = _compute_tuned_coefficients()


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
            qml.PauliZ(wires=data_wires[k]),
            qml.PauliZ(wires=data_wires[k + 1]),
        )

    z0 = z_op(0)

    def make_diag_block(sign_q0q1: float, sign_q1d0: float):
        gates: List[Callable[[], qml.operation.Operator]] = []
        coeffs: List[float] = []

        gates.append(i_op)
        coeffs.append(ETA + sign_q0q1 * J)

        gates.append(z0)
        coeffs.append(sign_q1d0 * J)

        for k in range(N_DATA_QUBITS_4X4):
            gates.append(x_op(k))
            coeffs.append(1.0)

        for k in range(N_DATA_QUBITS_4X4 - 1):
            gates.append(zz_op(k))
            coeffs.append(J)

        return gates, coeffs

    d00_g, d00_c = make_diag_block(sign_q0q1=+1.0, sign_q1d0=+1.0)
    d01_g, d01_c = make_diag_block(sign_q0q1=-1.0, sign_q1d0=-1.0)
    d10_g, d10_c = make_diag_block(sign_q0q1=-1.0, sign_q1d0=+1.0)
    d11_g, d11_c = make_diag_block(sign_q0q1=+1.0, sign_q1d0=-1.0)

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
        return qml.prod(*[qml.Hadamard(wires=wire) for wire in data_wires])

    raw_b_gates = [[h_all for _ in range(N_AGENTS_4X4)] for __ in range(N_AGENTS_4X4)]

    metadata = {
        "benchmark_name": "ising_4x4_2q_cond50",
        "n_total_qubits": N_TOTAL_QUBITS,
        "n_local_data_qubits": N_DATA_QUBITS_4X4,
        "spectrum": SPECTRUM_INFO,
    }
    system = LinearSystemData(
        raw_gates,
        raw_coeffs,
        raw_b_gates,
        data_wires=data_wires,
        b_weights_grid=[[1.0 for _ in range(N_AGENTS_4X4)] for __ in range(N_AGENTS_4X4)],
        zeta=ZETA,
        name="4x4_2q_cond50",
        metadata=metadata,
    )
    return system, raw_gates, raw_coeffs, raw_b_gates


SYSTEM_4X4, RAW_GATES_4X4, RAW_COEFFS_4X4, RAW_B_GATES_4X4 = _build_4x4_system()

CENTRALIZED_TERMS = _build_base_terms(eta=ETA, zeta=ZETA)
A_CENTRALIZED = _dense_from_terms(N_TOTAL_QUBITS, CENTRALIZED_TERMS)
B_CENTRALIZED = np.array(SYSTEM_4X4.get_global_b_vector(), dtype=np.complex128)
A_4X4 = np.array(SYSTEM_4X4.get_global_matrix(), dtype=np.complex128)
B_4X4 = np.array(SYSTEM_4X4.get_global_b_vector(), dtype=np.complex128)

CONSISTENCY = {
    "a_4x4_vs_centralized_max_abs_diff": float(np.max(np.abs(A_4X4 - A_CENTRALIZED))),
    "a_4x4_vs_centralized_fro_diff": float(np.linalg.norm(A_4X4 - A_CENTRALIZED)),
    "a_4x4_vs_centralized_allclose": bool(np.allclose(A_4X4, A_CENTRALIZED, atol=1.0e-12, rtol=0.0)),
    "b_4x4_vs_centralized_max_abs_diff": float(np.max(np.abs(B_4X4 - B_CENTRALIZED))),
    "b_4x4_vs_centralized_l2_diff": float(np.linalg.norm(B_4X4 - B_CENTRALIZED)),
    "b_4x4_vs_centralized_allclose": bool(np.allclose(B_4X4, B_CENTRALIZED, atol=1.0e-12, rtol=0.0)),
}

SYSTEMS = {
    "4x4": SYSTEM_4X4,
}

DATA_WIRES_BY_SYSTEM = {
    "4x4": DATA_WIRES_4X4,
}

CENTRALIZED_PROBLEMS = {
    "centralized": {
        "name": "ising_q4_direct_cond50",
        "n_total_qubits": N_TOTAL_QUBITS,
        "n_index_qubits": N_INDEX_QUBITS,
        "n_data_qubits": N_DATA_QUBITS_4X4,
        "a_matrix": A_CENTRALIZED,
        "b_vector": B_CENTRALIZED,
        "terms": CENTRALIZED_TERMS,
        "reference_system_key": "4x4",
        "spectrum": SPECTRUM_INFO,
    }
}

DEFAULT_SYSTEM_KEY = "4x4"
SYSTEM = SYSTEM_4X4
DATA_WIRES = DATA_WIRES_4X4
N_DATA_QUBITS = N_DATA_QUBITS_4X4
N_AGENTS = N_AGENTS_4X4
RAW_GATES = RAW_GATES_4X4
RAW_COEFFS = RAW_COEFFS_4X4
RAW_B_GATES = RAW_B_GATES_4X4


if __name__ == "__main__":
    print("=== static_ops_ising_4x4_2q ===")
    print("SPECTRUM_INFO =", SPECTRUM_INFO)
    for key in sorted(CONSISTENCY.keys()):
        print(f"{key}: {CONSISTENCY[key]}")
    print("A_4X4 shape:", A_4X4.shape)
