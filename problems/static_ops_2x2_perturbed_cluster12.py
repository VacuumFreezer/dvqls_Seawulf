"""
12-qubit 2x2 "perturbed cluster state" benchmark with qubit 1 as the index qubit.

The global system is

    A(eps) = 0.51 I + (0.1225 - eps / 4) * sum_{j in S} K_j + eps * Z_12

with S = {1, 4, 7, 10}, K_1 = X_1 Z_2, and K_j = Z_{j-1} X_j Z_{j+1}
for j in S \ {1}.  The reference vector |b> is the 12-qubit open-chain
cluster state prepared by an RY(pi/2) layer followed by alternating CZ bonds.
Because CZ gates commute, the retained 11-qubit half is represented exactly by
one pre-CZ RY layer and one open-chain CZ scaffold.
"""

from __future__ import annotations

from itertools import product
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pennylane as qml

from objective.circuits_cluster_nodispatch import ANSATZ_BRICKWALL_RY_CZ, apply_selected_ansatz


N_TOTAL_QUBITS = 12
INDEX_GLOBAL_QUBIT = 1
N_DATA_QUBITS_2X2 = N_TOTAL_QUBITS - 1
N_AGENTS_2X2 = 2

DATA_WIRES_2X2 = list(range(N_DATA_QUBITS_2X2))
LOCAL_GLOBAL_QUBITS = [g for g in range(1, N_TOTAL_QUBITS + 1) if g != INDEX_GLOBAL_QUBIT]
GLOBAL_TO_LOCAL = {g: i for i, g in enumerate(LOCAL_GLOBAL_QUBITS)}
LOCAL_LAST_WIRE = GLOBAL_TO_LOCAL[N_TOTAL_QUBITS]

EPSILON = 0.01
STABILIZER_COEFF = 49.0 / 400.0 - EPSILON / 4.0

STABILIZER_INDEX_SET = (1, 4, 7, 10)
GLOBAL_TERMS: List[Tuple[float, Dict[int, str], str]] = [
    (0.51, {}, "I"),
    (STABILIZER_COEFF, {1: "X", 2: "Z"}, "K_1"),
    (STABILIZER_COEFF, {3: "Z", 4: "X", 5: "Z"}, "K_4"),
    (STABILIZER_COEFF, {6: "Z", 7: "X", 8: "Z"}, "K_7"),
    (STABILIZER_COEFF, {9: "Z", 10: "X", 11: "Z"}, "K_10"),
    (EPSILON, {12: "Z"}, "Z_12"),
]

STABILIZER_SUPPORTS = [(1, 2)] + [(j - 1, j, j + 1) for j in STABILIZER_INDEX_SET[1:]]

COMMUTING_TERM_COEFFS = (STABILIZER_COEFF,) * len(STABILIZER_INDEX_SET) + (EPSILON,)
DISTINCT_EIGENVALUES = sorted(
    {
        0.51 + sum(sign * coeff for sign, coeff in zip(sign_pattern, COMMUTING_TERM_COEFFS))
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

CLUSTER_SCAFFOLD_EDGES_LOCAL = tuple(
    (GLOBAL_TO_LOCAL[left_global], GLOBAL_TO_LOCAL[right_global])
    for left_global, right_global in zip(LOCAL_GLOBAL_QUBITS[:-1], LOCAL_GLOBAL_QUBITS[1:])
    if right_global - left_global == 1
)

EXACT_ALPHA = float((1.0 - EPSILON) / (1.0 - 2.0 * EPSILON))
EXACT_BETA = float(-EPSILON / (1.0 - 2.0 * EPSILON))
EXACT_HALF_NORM = float(np.sqrt((EXACT_ALPHA**2 + EXACT_BETA**2) / 2.0))
EXACT_LAST_WIRE_ANGLE = float(np.pi / 2.0 - 2.0 * np.arctan(EPSILON / (1.0 - EPSILON)))


def _exact_row_angles(row_index: int) -> np.ndarray:
    weights = np.full((1, N_DATA_QUBITS_2X2), np.pi / 2.0, dtype=np.float64)
    if int(row_index) == 1:
        weights[0, 0] = -np.pi / 2.0
    weights[0, LOCAL_LAST_WIRE] = EXACT_LAST_WIRE_ANGLE
    return weights


def _reference_row_angles(row_index: int) -> np.ndarray:
    weights = np.full((1, N_DATA_QUBITS_2X2), np.pi / 2.0, dtype=np.float64)
    if int(row_index) == 1:
        weights[0, 0] = -np.pi / 2.0
    return weights


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


def _prepare_reference_row_state(data_wires: Sequence[int], row_index: int):
    weights = _reference_row_angles(row_index)
    apply_selected_ansatz(
        weights,
        len(data_wires),
        ansatz_kind=ANSATZ_BRICKWALL_RY_CZ,
        repeat_cz_each_layer=False,
        local_ry_support=(),
        scaffold_edges=CLUSTER_SCAFFOLD_EDGES_LOCAL,
    )


def _b_row_factory(data_wires: Sequence[int], row_index: int):
    def gate():
        _prepare_reference_row_state(data_wires, row_index=row_index)

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
        block_terms=None,
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
        self.block_terms = block_terms

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


def _decompose_global_terms_to_2x2(data_wires: Sequence[int]):
    diag00_gates: List[Callable[[], qml.operation.Operator]] = []
    diag00_coeffs: List[float] = []
    diag11_gates: List[Callable[[], qml.operation.Operator]] = []
    diag11_coeffs: List[float] = []
    off01_gates: List[Callable[[], qml.operation.Operator]] = []
    off01_coeffs: List[float] = []
    off10_gates: List[Callable[[], qml.operation.Operator]] = []
    off10_coeffs: List[float] = []

    diag00_terms = []
    diag11_terms = []
    off01_terms = []
    off10_terms = []

    for coeff, pauli_map, label in GLOBAL_TERMS:
        index_pauli = pauli_map.get(INDEX_GLOBAL_QUBIT, "I")
        local_paulis = {GLOBAL_TO_LOCAL[g]: p for g, p in pauli_map.items() if g != INDEX_GLOBAL_QUBIT}
        gate = _pauli_word_factory(local_paulis, data_wires)
        term = {"coeff": float(coeff), "pauli_map": dict(local_paulis), "label": str(label)}

        if index_pauli == "I":
            diag00_gates.append(gate)
            diag00_coeffs.append(float(coeff))
            diag11_gates.append(gate)
            diag11_coeffs.append(float(coeff))
            diag00_terms.append(term)
            diag11_terms.append(term)
        elif index_pauli == "X":
            off01_gates.append(gate)
            off01_coeffs.append(float(coeff))
            off10_gates.append(gate)
            off10_coeffs.append(float(coeff))
            off01_terms.append(term)
            off10_terms.append(term)
        else:
            raise ValueError(f"Unsupported Pauli on index qubit: {index_pauli}")

    return (
        [[diag00_gates, off01_gates], [off10_gates, diag11_gates]],
        [[diag00_coeffs, off01_coeffs], [off10_coeffs, diag11_coeffs]],
        [[diag00_terms, off01_terms], [off10_terms, diag11_terms]],
    )


RAW_GATES_2X2, RAW_COEFFS_2X2, LOCAL_BLOCK_TERMS_2X2 = _decompose_global_terms_to_2x2(DATA_WIRES_2X2)

_b_row0 = _b_row_factory(DATA_WIRES_2X2, row_index=0)
_b_row1 = _b_row_factory(DATA_WIRES_2X2, row_index=1)
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
    name="perturbed_cluster_state_12q_2x2_q1_index",
    metadata={
        "n_total_qubits": N_TOTAL_QUBITS,
        "index_global_qubit": INDEX_GLOBAL_QUBIT,
        "local_global_qubits": list(LOCAL_GLOBAL_QUBITS),
        "stabilizer_supports": [list(support) for support in STABILIZER_SUPPORTS],
        "global_term_count": len(GLOBAL_TERMS),
        "block_term_counts": BLOCK_TERM_COUNTS_2X2,
        "spectrum": dict(SPECTRUM_INFO),
        "real_problem": True,
        "recommended_ansatz": ANSATZ_BRICKWALL_RY_CZ,
        "cluster_scaffold_edges_local": [list(edge) for edge in CLUSTER_SCAFFOLD_EDGES_LOCAL],
        "init_angle_fill": float(np.pi / 2.0),
        "init_sigma_target": EXACT_HALF_NORM,
        "agent_init_overrides": {
            0: {LOCAL_LAST_WIRE: EXACT_LAST_WIRE_ANGLE},
            1: {0: float(-np.pi / 2.0), LOCAL_LAST_WIRE: EXACT_LAST_WIRE_ANGLE},
        },
        "design_note": (
            "weakly perturbed 1D cluster benchmark with exact shallow real solution and nonzero "
            "off-diagonal 2x2 blocks after splitting on qubit 1"
        ),
    },
    max_dense_qubits=16,
    block_terms=LOCAL_BLOCK_TERMS_2X2,
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


def _to_real_state(vec: np.ndarray, *, tol: float = 1.0e-10) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.complex128).reshape(-1)
    imag_norm = float(np.linalg.norm(arr.imag))
    if imag_norm > tol:
        raise RuntimeError(f"Expected a real statevector, but ||Im(vec)||={imag_norm:.3e}.")
    return np.asarray(arr.real, dtype=np.float64)


def _format_preview(vec: np.ndarray, *, label: str, max_entries: int = 16) -> str:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    n_show = min(int(max_entries), arr.size)
    lines = [f"{label} (size={arr.size}, first {n_show} entries)"]
    for i in range(n_show):
        lines.append(f"[{i:4d}] {arr[i]:+.8e}")
    if n_show < arr.size:
        lines.append("...")
    return "\n".join(lines)


def _apply_local_pauli_word_real(state: np.ndarray, pauli_map: Dict[int, str], n_qubits: int) -> np.ndarray:
    out = np.asarray(state, dtype=np.float64).copy()
    for wire in sorted(pauli_map.keys()):
        label = str(pauli_map[wire])
        half = 1 << (int(n_qubits) - int(wire) - 1)
        step = half << 1
        if label == "Z":
            for start in range(0, out.size, step):
                out[start + half : start + step] *= -1.0
        elif label == "X":
            for start in range(0, out.size, step):
                left = out[start : start + half].copy()
                out[start : start + half] = out[start + half : start + step]
                out[start + half : start + step] = left
        else:
            raise ValueError(f"Only X/Z local Pauli maps are supported, got {label!r}.")
    return out


def _apply_block_operator_real(block_terms, state: np.ndarray, n_qubits: int) -> np.ndarray:
    out = np.zeros_like(np.asarray(state, dtype=np.float64))
    for term in block_terms:
        contrib = _apply_local_pauli_word_real(state, term["pauli_map"], n_qubits)
        out += float(term["coeff"]) * contrib
    return out


def _build_state_qnode(n_qubits: int, *, ansatz_kind: str, repeat_cz_each_layer: bool, local_ry_support, scaffold_edges):
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="jax")
    def get_state(weights):
        apply_selected_ansatz(
            weights,
            n_qubits,
            ansatz_kind=ansatz_kind,
            repeat_cz_each_layer=repeat_cz_each_layer,
            local_ry_support=local_ry_support,
            scaffold_edges=scaffold_edges,
        )
        return qml.state()

    return get_state


def write_structured_analysis(
    *,
    out,
    global_params,
    ansatz_kind: str,
    repeat_cz_each_layer: bool,
    local_ry_support,
    scaffold_edges,
):
    n_qubits = N_DATA_QUBITS_2X2
    get_state = _build_state_qnode(
        n_qubits,
        ansatz_kind=str(ansatz_kind),
        repeat_cz_each_layer=bool(repeat_cz_each_layer),
        local_ry_support=tuple(int(x) for x in local_ry_support),
        scaffold_edges=tuple((int(a), int(b)) for a, b in scaffold_edges),
    )
    get_exact_state = _build_state_qnode(
        n_qubits,
        ansatz_kind=ANSATZ_BRICKWALL_RY_CZ,
        repeat_cz_each_layer=False,
        local_ry_support=(),
        scaffold_edges=CLUSTER_SCAFFOLD_EDGES_LOCAL,
    )

    x_local = [[None, None], [None, None]]
    trained_state = [[None, None], [None, None]]
    for sys_id in range(2):
        for agent_id in range(2):
            alpha = global_params["alpha"][sys_id][agent_id]
            trained_state[sys_id][agent_id] = _to_real_state(np.array(get_state(alpha)))
            sigma = float(np.asarray(global_params["sigma"][sys_id][agent_id]))
            x_local[sys_id][agent_id] = sigma * trained_state[sys_id][agent_id]

    u_est = 0.5 * (x_local[0][0] + x_local[1][0])
    v_est = 0.5 * (x_local[0][1] + x_local[1][1])

    row0_norm = _to_real_state(np.array(get_exact_state(_exact_row_angles(0))))
    row1_norm = _to_real_state(np.array(get_exact_state(_exact_row_angles(1))))
    b0 = (1.0 / np.sqrt(2.0)) * _to_real_state(np.array(get_exact_state(_reference_row_angles(0))))
    b1 = (1.0 / np.sqrt(2.0)) * _to_real_state(np.array(get_exact_state(_reference_row_angles(1))))
    u_true = EXACT_HALF_NORM * row0_norm
    v_true = EXACT_HALF_NORM * row1_norm

    sol_diff_num = float(np.linalg.norm(u_est - u_true) ** 2 + np.linalg.norm(v_est - v_true) ** 2)
    sol_diff_den = float(np.linalg.norm(u_true) ** 2 + np.linalg.norm(v_true) ** 2)
    sol_diff = float(np.sqrt(sol_diff_num / max(sol_diff_den, 1.0e-30)))

    row0_residual = (
        _apply_block_operator_real(LOCAL_BLOCK_TERMS_2X2[0][0], u_est, n_qubits)
        + _apply_block_operator_real(LOCAL_BLOCK_TERMS_2X2[0][1], v_est, n_qubits)
        - b0
    )
    row1_residual = (
        _apply_block_operator_real(LOCAL_BLOCK_TERMS_2X2[1][0], u_est, n_qubits)
        + _apply_block_operator_real(LOCAL_BLOCK_TERMS_2X2[1][1], v_est, n_qubits)
        - b1
    )
    residual_norm = float(np.sqrt(np.linalg.norm(row0_residual) ** 2 + np.linalg.norm(row1_residual) ** 2))

    consensus_variances = []
    for agent_id in range(2):
        vec_stack = np.stack([x_local[0][agent_id], x_local[1][agent_id]], axis=0)
        mean_vec = np.mean(vec_stack, axis=0)
        diffs = vec_stack - mean_vec
        sq_dists = np.sum(diffs * diffs, axis=1)
        consensus_variances.append(float(np.mean(sq_dists)))
    avg_consensus_variance = float(np.mean(consensus_variances))

    print("\n" + "=" * 60, file=out)
    print("      STRUCTURED PERTURBED CLUSTER ANALYSIS", file=out)
    print("=" * 60, file=out)
    print(f"epsilon: {EPSILON:.8f}", file=out)
    print(f"exact alpha coefficient: {EXACT_ALPHA:.8f}", file=out)
    print(f"exact beta coefficient : {EXACT_BETA:.8f}", file=out)
    print(f"exact half norm        : {EXACT_HALF_NORM:.8e}", file=out)
    print(f"exact last-wire angle  : {EXACT_LAST_WIRE_ANGLE:.8f}", file=out)
    print(f"final sol_diff         : {sol_diff:.8e}", file=out)
    print(f"final ||Ax-b||         : {residual_norm:.8e}", file=out)
    print(f"consensus variance avg : {avg_consensus_variance:.8e}", file=out)

    for agent_id in range(2):
        print(f"consensus variance agent {agent_id}: {consensus_variances[agent_id]:.8e}", file=out)

    print("\n" + "=" * 60, file=out)
    print("      FINAL PARAMETERS", file=out)
    print("=" * 60, file=out)
    for sys_id in range(2):
        print(f"\n>>> SYSTEM {sys_id} <<<", file=out)
        for agent_id in range(2):
            print(f"  [Agent {agent_id}]", file=out)
            print(f"alpha: {np.array2string(np.asarray(global_params['alpha'][sys_id][agent_id]), precision=8)}", file=out)
            print(f"beta : {np.array2string(np.asarray(global_params['beta'][sys_id][agent_id]), precision=8)}", file=out)
            print(f"sigma: {float(np.asarray(global_params['sigma'][sys_id][agent_id])):.8e}", file=out)
            print(f"lambda: {float(np.asarray(global_params['lambda'][sys_id][agent_id])):.8e}", file=out)

    print("\n" + "=" * 60, file=out)
    print("      RECONSTRUCTED SOLUTION PREVIEW", file=out)
    print("=" * 60, file=out)
    print(_format_preview(u_est, label="u_est"), file=out)
    print(_format_preview(v_est, label="v_est"), file=out)
    print(_format_preview(u_true, label="u_true"), file=out)
    print(_format_preview(v_true, label="v_true"), file=out)
    print(_format_preview(b0, label="b_row0"), file=out)
    print(_format_preview(b1, label="b_row1"), file=out)
