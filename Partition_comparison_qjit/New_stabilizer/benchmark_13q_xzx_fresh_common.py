from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from itertools import product
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

try:
    import pennylane as qml
except Exception:  # pragma: no cover - verifier can run without PennyLane
    qml = None

try:
    from scipy import sparse as sp
    from scipy.sparse.linalg import spsolve

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - optional at verification/build time
    sp = None
    spsolve = None
    SCIPY_AVAILABLE = False


N_TOTAL_QUBITS = 13
ALPHA = 21.0 / 40.0
DEFAULT_EPSILON = 0.1
BASE_BLOCK_ANGLE = np.pi / 2.0

PLUS_STATE_QUBITS = frozenset((1, 3, 4, 6, 7, 9, 10, 12, 13))
ZERO_STATE_QUBITS = frozenset((2, 5, 8, 11))

B_STATE_ORIGINAL = "original"
B_STATE_ALL_PLUS = "all_plus"
VALID_B_STATE_KINDS = (B_STATE_ORIGINAL, B_STATE_ALL_PLUS)

B_PREP_RY = "ry"
B_PREP_HADAMARD = "hadamard"
VALID_B_PREP_KINDS = (B_PREP_RY, B_PREP_HADAMARD)

STABILIZER_TERMS: Tuple[Tuple[str, Dict[int, str]], ...] = (
    ("K_1", {1: "X", 2: "Z", 3: "X"}),
    ("K_2", {4: "X", 5: "Z", 6: "X"}),
    ("K_3", {7: "X", 8: "Z", 9: "X"}),
    ("K_4", {10: "X", 11: "Z", 12: "X"}),
)


def beta_from_epsilon(epsilon: float) -> float:
    return (19.0 / 160.0) - (float(epsilon) / 4.0)


def exact_final_angle(epsilon: float) -> float:
    eps = float(epsilon)
    return BASE_BLOCK_ANGLE + (2.0 * np.arctan(eps / (1.0 - eps)))


def exact_solution_scale(epsilon: float) -> float:
    eps = float(epsilon)
    numerator = np.sqrt(((1.0 - eps) ** 2) + (eps**2))
    denominator = 1.0 - (2.0 * eps)
    return float(numerator / denominator)


def bitstring_tuple(index: int, width: int) -> tuple[int, ...]:
    if width <= 0:
        return ()
    return tuple((int(index) >> shift) & 1 for shift in range(width - 1, -1, -1))


def bitstring_label(index: int, width: int) -> str:
    bits = bitstring_tuple(index, width)
    return "".join(str(bit) for bit in bits) if bits else "global"


def _normalize_b_state_kind(kind: str) -> str:
    value = str(kind).strip().lower()
    if value not in VALID_B_STATE_KINDS:
        raise ValueError(f"Unsupported b_state_kind {kind!r}; expected one of {VALID_B_STATE_KINDS}.")
    return value


def _normalize_b_prep_kind(kind: str) -> str:
    value = str(kind).strip().lower()
    if value not in VALID_B_PREP_KINDS:
        raise ValueError(f"Unsupported b_prep_kind {kind!r}; expected one of {VALID_B_PREP_KINDS}.")
    return value


def _global_reference_angle(global_qubit: int, *, state_kind: str, last_angle: float) -> float:
    state_kind = _normalize_b_state_kind(state_kind)
    qubit = int(global_qubit)
    if state_kind == B_STATE_ALL_PLUS:
        return BASE_BLOCK_ANGLE
    if qubit == N_TOTAL_QUBITS:
        return float(last_angle)
    if qubit in ZERO_STATE_QUBITS:
        return 0.0
    return BASE_BLOCK_ANGLE


def _single_qubit_amplitude(angle: float, bit: int) -> float:
    return float(np.cos(angle / 2.0) if int(bit) == 0 else np.sin(angle / 2.0))


def _product_state_angles_for_global_qubits(
    global_qubits: Sequence[int],
    *,
    state_kind: str,
    last_angle: float,
) -> np.ndarray:
    return np.asarray(
        [
            _global_reference_angle(global_qubit, state_kind=state_kind, last_angle=float(last_angle))
            for global_qubit in global_qubits
        ],
        dtype=np.float64,
    )


def _prefix_block_scale(
    prefix_bits: Sequence[int],
    *,
    state_kind: str,
    last_angle: float,
    include_solution_scale: float = 1.0,
) -> float:
    scale = 1.0
    for global_qubit, bit in enumerate(prefix_bits, start=1):
        angle = _global_reference_angle(global_qubit, state_kind=state_kind, last_angle=float(last_angle))
        scale *= _single_qubit_amplitude(angle, int(bit))
    return float(include_solution_scale) * float(scale)


def _matrix_element_index_register(
    pauli_map: Dict[int, str],
    row_bits: Sequence[int],
    col_bits: Sequence[int],
) -> complex:
    amp = 1.0 + 0.0j

    for qubit in range(1, len(row_bits) + 1):
        pauli = pauli_map.get(qubit, "I")
        row_bit = int(row_bits[qubit - 1])
        col_bit = int(col_bits[qubit - 1])

        if pauli == "I":
            if row_bit != col_bit:
                return 0.0 + 0.0j
            continue

        if pauli == "Z":
            if row_bit != col_bit:
                return 0.0 + 0.0j
            amp *= -1.0 if col_bit else 1.0
            continue

        if pauli == "X":
            if row_bit != (1 - col_bit):
                return 0.0 + 0.0j
            continue

        raise ValueError(f"Unsupported Pauli label on index register: {pauli!r}")

    return amp


def _pauli_word_factory(local_ops: Dict[int, str], data_wires: Sequence[int]):
    local_ops = {int(k): str(v) for k, v in local_ops.items()}

    def gate():
        if qml is None:
            raise RuntimeError("PennyLane is required to execute gate factories.")
        factors = []
        for local_wire in sorted(local_ops):
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


def _identity_gate_factory(data_wires: Sequence[int]):
    def gate():
        if qml is None:
            raise RuntimeError("PennyLane is required to execute gate factories.")
        return qml.Identity(wires=data_wires[0])

    return gate


def _apply_pauli_word_to_state(state: np.ndarray, pauli_map: Dict[int, str], n_qubits: int) -> np.ndarray:
    tensor = np.asarray(state, dtype=complex).reshape((2,) * int(n_qubits))
    out = tensor

    for wire, label in sorted(pauli_map.items()):
        axis = int(wire)
        if label == "I":
            continue
        if label == "X":
            out = np.flip(out, axis=axis)
            continue
        if label == "Z":
            phase = np.asarray([1.0, -1.0], dtype=complex).reshape(
                (1,) * axis + (2,) + (1,) * (int(n_qubits) - axis - 1)
            )
            out = out * phase
            continue
        raise ValueError(f"Unsupported local Pauli label: {label!r}")

    return np.asarray(out).reshape(-1)


def _product_state_vector(angles: Sequence[float]) -> np.ndarray:
    state = np.asarray([1.0 + 0.0j], dtype=complex)
    for angle in angles:
        qubit_state = np.asarray([np.cos(angle / 2.0), np.sin(angle / 2.0)], dtype=complex)
        state = np.kron(state, qubit_state)
    return state


def _product_ry_factory(data_wires: Sequence[int], angles: Sequence[float]):
    angles = tuple(float(angle) for angle in angles)

    def gate():
        if qml is None:
            raise RuntimeError("PennyLane is required to execute state-prep gate factories.")
        if not angles:
            return qml.Identity(wires=data_wires[0])
        for local_wire, angle in enumerate(angles):
            if abs(float(angle)) > 1e-15:
                qml.RY(float(angle), wires=data_wires[local_wire])

    return gate


def _product_hadamard_factory(data_wires: Sequence[int], angles: Sequence[float]):
    angles = tuple(float(angle) for angle in angles)

    def gate():
        if qml is None:
            raise RuntimeError("PennyLane is required to execute state-prep gate factories.")
        if not angles:
            return qml.Identity(wires=data_wires[0])
        for local_wire, angle in enumerate(angles):
            if abs(float(angle)) <= 1e-15:
                continue
            if abs(float(angle) - BASE_BLOCK_ANGLE) > 1e-12:
                raise ValueError(
                    "Hadamard-based b-preparation only supports local amplitudes prepared by 0 or pi/2 rotations."
                )
            qml.Hadamard(wires=data_wires[local_wire])

    return gate


def _local_chain_edge_layers(
    index_qubits: int,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    odd_edges: List[Tuple[int, int]] = []
    even_edges: List[Tuple[int, int]] = []

    for left_global in range(index_qubits + 1, N_TOTAL_QUBITS):
        left_local = left_global - (index_qubits + 1)
        right_local = left_local + 1
        edge = (left_local, right_local)
        if left_global % 2 == 1:
            odd_edges.append(edge)
        else:
            even_edges.append(edge)

    all_edges = tuple(odd_edges + even_edges)
    return tuple(odd_edges), tuple(even_edges), all_edges


@dataclass
class LinearSystemData:
    gates_grid: List[List[List[Callable[[], qml.operation.Operator]]]]
    coeffs_grid: List[List[List[float]]]
    b_gates_grid: List[List[Callable[[], None]]]
    data_wires: Sequence[int]
    b_weights_grid: List[List[float]]
    name: str
    metadata: Dict[str, object]
    local_pauli_maps_grid: List[List[List[Dict[int, str]]]]
    global_term_breakdown: List[List[List[Dict[str, object]]]]
    row_b_state: Sequence[np.ndarray]
    row_b_norms: Sequence[float]
    row_x_state: Sequence[np.ndarray]
    row_x_scales: Sequence[float]

    def __post_init__(self):
        self.n = len(self.gates_grid)
        self.data_wires = list(self.data_wires)
        self.n_data_qubits = len(self.data_wires)
        self.b_gates = self.b_gates_grid
        self.b_weights = [list(map(float, row)) for row in self.b_weights_grid]

        self.coeffs = []
        self.ops = []
        for i in range(self.n):
            row_coeffs = []
            row_ops = []
            for j in range(self.n):
                g_list = list(self.gates_grid[i][j])
                c_list = list(self.coeffs_grid[i][j])
                if len(g_list) != len(c_list):
                    raise ValueError(
                        f"Coeff length mismatch at ({i}, {j}): len(gates)={len(g_list)} vs len(coeffs)={len(c_list)}"
                    )
                row_coeffs.append(c_list)
                row_ops.append(self._make_wrapper(g_list))
            self.coeffs.append(row_coeffs)
            self.ops.append(row_ops)

        self._mat_cache: Dict[int, np.ndarray] = {}
        self._block_b_cache: Dict[int, np.ndarray] = {}
        self._true_solution_cache: np.ndarray | None = None

    def _local_masks(self, local_pauli_map: Dict[int, str]) -> tuple[int, int]:
        x_mask = 0
        z_mask = 0
        for wire in range(int(self.n_data_qubits)):
            bit = 1 << (int(self.n_data_qubits) - wire - 1)
            label = local_pauli_map.get(wire, "I")
            if label == "X":
                x_mask |= bit
            elif label == "Z":
                z_mask |= bit
        return x_mask, z_mask

    @staticmethod
    def _popcount(value: int) -> int:
        return bin(int(value)).count("1")

    def reconstruct_global_entries(self) -> dict[tuple[int, int], complex]:
        n_agents = int(self.n)
        n_local = int(self.n_data_qubits)
        local_dim = 1 << n_local
        entries: dict[tuple[int, int], complex] = {}

        for row_id in range(n_agents):
            for col_id in range(n_agents):
                for coeff, local_paulis in zip(self.coeffs[row_id][col_id], self.local_pauli_maps_grid[row_id][col_id]):
                    x_mask, z_mask = self._local_masks(local_paulis)
                    for local_col in range(local_dim):
                        local_row = local_col ^ x_mask
                        phase = -1.0 if self._popcount(local_col & z_mask) % 2 else 1.0
                        global_row = row_id * local_dim + local_row
                        global_col = col_id * local_dim + local_col
                        key = (global_row, global_col)
                        entries[key] = entries.get(key, 0.0 + 0.0j) + (complex(coeff) * phase)

        return {key: value for key, value in entries.items() if abs(value) > 1e-14}

    def _make_wrapper(self, gate_factories):
        def wrapper(term_id):
            return gate_factories[int(term_id)]()

        return wrapper

    def _as_matrix(self, gate_fn: Callable[[], None]) -> np.ndarray:
        key = id(gate_fn)
        if key in self._mat_cache:
            return self._mat_cache[key]

        def qfunc():
            gate_fn()

        matrix = np.asarray(qml.matrix(qfunc, wire_order=self.data_wires)(), dtype=complex)
        self._mat_cache[key] = matrix
        return matrix

    def get_local_b_norms(self, sys_id: int) -> Tuple[float, ...]:
        return tuple(abs(float(weight)) for weight in self.b_weights[int(sys_id)])

    def get_b_vectors(self, sys_id: int):
        row = int(sys_id)
        if row in self._block_b_cache:
            b_total = self._block_b_cache[row]
        else:
            b_total = float(self.row_b_norms[row]) * np.asarray(self.row_b_state[row], dtype=complex)
            self._block_b_cache[row] = b_total

        individual = [
            float(weight) * np.asarray(self.row_b_state[row], dtype=complex) for weight in self.b_weights[row]
        ]
        return (b_total, *individual)

    def get_global_b_vector(self) -> np.ndarray:
        return np.concatenate([self.get_b_vectors(sys_id)[0] for sys_id in range(self.n)], axis=0)

    def get_b_op(self, sys_id: int, agent_id: int):
        return self.b_gates[int(sys_id)][int(agent_id)]

    def get_global_matrix(self) -> np.ndarray:
        block_rows = []
        for i in range(self.n):
            block_cols = []
            for j in range(self.n):
                mats = [self._as_matrix(gate) for gate in self.gates_grid[i][j]]
                combined = np.zeros_like(mats[0], dtype=complex)
                for coeff, mat in zip(self.coeffs[i][j], mats):
                    combined = combined + float(coeff) * mat
                block_cols.append(combined)
            block_rows.append(block_cols)
        return np.block(block_rows)

    def apply_block_operator(self, row_id: int, col_id: int, local_state: np.ndarray) -> np.ndarray:
        row = int(row_id)
        col = int(col_id)
        out = np.zeros_like(np.asarray(local_state, dtype=complex))
        for coeff, pauli_map in zip(self.coeffs[row][col], self.local_pauli_maps_grid[row][col]):
            out = out + float(coeff) * _apply_pauli_word_to_state(local_state, pauli_map, self.n_data_qubits)
        return out

    def true_solution_vector(self) -> np.ndarray:
        if self._true_solution_cache is not None:
            return self._true_solution_cache

        if SCIPY_AVAILABLE:
            entries = self.reconstruct_global_entries()
            dim = int(self.n) * (1 << int(self.n_data_qubits))
            rows = []
            cols = []
            data = []
            for (row, col), value in entries.items():
                rows.append(int(row))
                cols.append(int(col))
                data.append(complex(value))
            matrix = sp.coo_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols))), shape=(dim, dim)).tocsr()
            global_b = np.asarray(self.get_global_b_vector(), dtype=np.complex128)
            self._true_solution_cache = np.asarray(spsolve(matrix, global_b), dtype=np.complex128)
            return self._true_solution_cache

        blocks = []
        for row in range(self.n):
            blocks.append(float(self.row_x_scales[row]) * np.asarray(self.row_x_state[row], dtype=complex))
        self._true_solution_cache = np.concatenate(blocks, axis=0)
        return self._true_solution_cache


def build_partition_namespace(
    index_qubits: int,
    *,
    epsilon: float = DEFAULT_EPSILON,
    b_state_kind: str = B_STATE_ORIGINAL,
    b_prep_kind: str = B_PREP_RY,
) -> dict:
    k = int(index_qubits)
    if k < 0 or k > 3:
        raise ValueError(f"Expected index_qubits in {{0, 1, 2, 3}}, got {k}.")

    epsilon = float(epsilon)
    b_state_kind = _normalize_b_state_kind(b_state_kind)
    b_prep_kind = _normalize_b_prep_kind(b_prep_kind)
    n_agents = 1 << k
    n_data_qubits = N_TOTAL_QUBITS - k
    data_wires = list(range(n_data_qubits))
    local_global_qubits = list(range(k + 1, N_TOTAL_QUBITS + 1))
    beta = beta_from_epsilon(epsilon)
    final_angle = exact_final_angle(epsilon)
    solution_scale = exact_solution_scale(epsilon)

    odd_edges_local, even_edges_local, scaffold_edges_local = _local_chain_edge_layers(k)

    global_terms: List[Tuple[float, Dict[int, str], str]] = [(ALPHA, {}, "I")]
    for label, pauli_map in STABILIZER_TERMS:
        global_terms.append((beta, dict(pauli_map), label))
    global_terms.append((epsilon, {13: "Z"}, "Z_13"))

    b_local_angles = _product_state_angles_for_global_qubits(
        local_global_qubits, state_kind=b_state_kind, last_angle=BASE_BLOCK_ANGLE
    )
    if b_state_kind == B_STATE_ORIGINAL:
        x_local_angles = _product_state_angles_for_global_qubits(
            local_global_qubits, state_kind=b_state_kind, last_angle=final_angle
        )
    else:
        x_local_angles = np.asarray(b_local_angles, dtype=np.float64)

    if b_prep_kind == B_PREP_RY:
        b_gate = _product_ry_factory(data_wires, b_local_angles)
    else:
        b_gate = _product_hadamard_factory(data_wires, b_local_angles)

    x_gate = _product_ry_factory(data_wires, x_local_angles)
    b_local_state = _product_state_vector(b_local_angles)
    x_local_state = _product_state_vector(x_local_angles)

    row_labels = [bitstring_label(index, k) for index in range(n_agents)]
    row_b_norms: List[float] = []
    row_x_scales: List[float] = []
    row_b_state = [b_local_state for _ in range(n_agents)]
    row_x_state = [x_local_state for _ in range(n_agents)]

    for index in range(n_agents):
        bits = bitstring_tuple(index, k)
        row_b_norms.append(_prefix_block_scale(bits, state_kind=b_state_kind, last_angle=BASE_BLOCK_ANGLE))
        if b_state_kind == B_STATE_ORIGINAL:
            row_x_scales.append(
                _prefix_block_scale(
                    bits,
                    state_kind=b_state_kind,
                    last_angle=BASE_BLOCK_ANGLE,
                    include_solution_scale=solution_scale,
                )
            )
        else:
            row_x_scales.append(_prefix_block_scale(bits, state_kind=b_state_kind, last_angle=BASE_BLOCK_ANGLE))

    b_weights = [
        [
            (float(row_b_norms[row_id]) / float(n_agents)) if abs(float(row_b_norms[row_id])) > 1e-15 else 0.0
            for _ in range(n_agents)
        ]
        for row_id in range(n_agents)
    ]
    raw_b_gates = [[b_gate for _ in range(n_agents)] for __ in range(n_agents)]

    gates_grid: List[List[List[Callable[[], qml.operation.Operator]]]] = [[[] for _ in range(n_agents)] for __ in range(n_agents)]
    coeffs_grid: List[List[List[float]]] = [[[] for _ in range(n_agents)] for __ in range(n_agents)]
    local_pauli_maps_grid: List[List[List[Dict[int, str]]]] = [[[] for _ in range(n_agents)] for __ in range(n_agents)]
    term_breakdown_grid: List[List[List[Dict[str, object]]]] = [[[] for _ in range(n_agents)] for __ in range(n_agents)]

    for row_id in range(n_agents):
        row_bits = bitstring_tuple(row_id, k)
        for col_id in range(n_agents):
            col_bits = bitstring_tuple(col_id, k)
            cell_entries: List[Tuple[float, Dict[int, str], str]] = []

            for coeff, pauli_map, label in global_terms:
                amp = _matrix_element_index_register(pauli_map, row_bits, col_bits)
                if abs(amp) < 1e-12:
                    continue

                local_paulis = {
                    int(global_qubit - (k + 1)): str(pauli)
                    for global_qubit, pauli in pauli_map.items()
                    if int(global_qubit) > k
                }
                cell_entries.append((float(np.real_if_close(coeff * amp)), local_paulis, label))

            if not cell_entries:
                local_pauli_maps_grid[row_id][col_id] = [{}]
                coeffs_grid[row_id][col_id] = [0.0]
                gates_grid[row_id][col_id] = [_identity_gate_factory(data_wires)]
                term_breakdown_grid[row_id][col_id] = [
                    {"label": "0", "coefficient": 0.0, "local_paulis": {}}
                ]
                continue

            merged: Dict[Tuple[Tuple[int, str], ...], Dict[str, object]] = {}
            for coeff, local_paulis, label in cell_entries:
                key = tuple(sorted((int(wire), str(pauli)) for wire, pauli in local_paulis.items()))
                if key not in merged:
                    merged[key] = {"coefficient": 0.0, "local_paulis": dict(local_paulis), "labels": []}
                merged[key]["coefficient"] = float(merged[key]["coefficient"]) + float(coeff)
                merged[key]["labels"].append(str(label))

            for merged_entry in merged.values():
                coeff = float(merged_entry["coefficient"])
                if abs(coeff) < 1e-12:
                    continue
                local_paulis = dict(merged_entry["local_paulis"])
                local_pauli_maps_grid[row_id][col_id].append(local_paulis)
                coeffs_grid[row_id][col_id].append(coeff)
                gates_grid[row_id][col_id].append(_pauli_word_factory(local_paulis, data_wires))
                term_breakdown_grid[row_id][col_id].append(
                    {
                        "label": " + ".join(merged_entry["labels"]),
                        "coefficient": coeff,
                        "local_paulis": local_paulis,
                    }
                )

            if not gates_grid[row_id][col_id]:
                local_pauli_maps_grid[row_id][col_id] = [{}]
                coeffs_grid[row_id][col_id] = [0.0]
                gates_grid[row_id][col_id] = [_identity_gate_factory(data_wires)]
                term_breakdown_grid[row_id][col_id] = [
                    {"label": "0", "coefficient": 0.0, "local_paulis": {}}
                ]

    distinct_eigenvalues = sorted(
        {
            ALPHA + (beta * sum(sign_pattern[:4])) + (epsilon * sign_pattern[4])
            for sign_pattern in product((-1.0, 1.0), repeat=5)
        }
    )
    spectrum_info = {
        "lambda_min": 1.0 / 20.0,
        "lambda_max": 1.0,
        "condition_number": 20.0,
        "n_distinct_eigenvalues": len(distinct_eigenvalues),
        "distinct_eigenvalues": distinct_eigenvalues,
        "b_is_eigenvector": bool(abs(epsilon) < 1e-15),
    }

    init_angle_fill = BASE_BLOCK_ANGLE
    init_overrides = {}
    for agent_id in range(n_agents):
        overrides = {}
        if b_state_kind == B_STATE_ORIGINAL:
            for local_wire, global_qubit in enumerate(local_global_qubits):
                if global_qubit in ZERO_STATE_QUBITS:
                    overrides[local_wire] = 0.0
        if overrides:
            init_overrides[agent_id] = overrides

    metadata = {
        "benchmark_name": "13q_xzx_uniform_stabilizer_prefix_partition",
        "n_total_qubits": N_TOTAL_QUBITS,
        "index_qubits": k,
        "index_global_qubits": list(range(1, k + 1)),
        "n_agents": n_agents,
        "local_global_qubits": local_global_qubits,
        "bitstring_labels": row_labels,
        "boundary_flip_by_row": [False for _ in range(n_agents)],
        "block_phase_by_row": [1.0 for _ in range(n_agents)],
        "row_b_norms": row_b_norms,
        "row_x_scales": row_x_scales,
        "b_state_kind": b_state_kind,
        "b_prep_kind": b_prep_kind,
        "epsilon": epsilon,
        "alpha": ALPHA,
        "beta": beta,
        "b_last_angle": BASE_BLOCK_ANGLE,
        "exact_solution_last_angle": final_angle,
        "exact_solution_scale": solution_scale,
        "odd_cz_edges_local": odd_edges_local,
        "even_cz_edges_local": even_edges_local,
        "cluster_scaffold_edges_local": scaffold_edges_local,
        "global_term_count": len(global_terms),
        "block_term_counts": [[len(cell) for cell in row] for row in gates_grid],
        "spectrum": spectrum_info,
        "real_problem": True,
        "recommended_ansatz": "brickwall_ry_cz",
        "recommended_layers": 4,
        "exact_solution_family": (
            "single_ry_layer_product_state" if b_state_kind == B_STATE_ORIGINAL else "computed_via_sparse_solve"
        ),
        "init_sigma_target": 1.0,
        "init_angle_fill": init_angle_fill,
        "agent_init_overrides": init_overrides,
        "reference_row_gates": [b_gate for _ in range(n_agents)],
        "exact_solution_gates_by_col": [x_gate for _ in range(n_agents)],
    }

    partition_label = f"{n_agents}x{n_agents}"
    system = LinearSystemData(
        gates_grid=gates_grid,
        coeffs_grid=coeffs_grid,
        b_gates_grid=raw_b_gates,
        data_wires=data_wires,
        b_weights_grid=b_weights,
        name=f"xzx13_{b_state_kind}_{b_prep_kind}_prefix_partition_{partition_label}",
        metadata=metadata,
        local_pauli_maps_grid=local_pauli_maps_grid,
        global_term_breakdown=term_breakdown_grid,
        row_b_state=row_b_state,
        row_b_norms=row_b_norms,
        row_x_state=row_x_state,
        row_x_scales=row_x_scales,
    )

    return {
        "N_TOTAL_QUBITS": N_TOTAL_QUBITS,
        "N_DATA_QUBITS": n_data_qubits,
        "N_AGENTS": n_agents,
        "INDEX_QUBITS": k,
        "EPSILON": epsilon,
        "ALPHA": ALPHA,
        "BETA": beta,
        "BLOCK_NORM": float(row_b_norms[0]) if row_b_norms else 1.0,
        "ROW_B_NORMS": tuple(float(x) for x in row_b_norms),
        "ROW_X_SCALES": tuple(float(x) for x in row_x_scales),
        "TRUE_SOLUTION_FINAL_ANGLE": final_angle,
        "TRUE_SOLUTION_SCALE": solution_scale,
        "B_STATE_KIND": b_state_kind,
        "B_PREP_KIND": b_prep_kind,
        "GLOBAL_TERMS": global_terms,
        "SPECTRUM_INFO": spectrum_info,
        "RAW_GATES": gates_grid,
        "RAW_COEFFS": coeffs_grid,
        "RAW_B_GATES": raw_b_gates,
        "RAW_LOCAL_PAULI_MAPS": local_pauli_maps_grid,
        "DATA_WIRES": data_wires,
        "CLUSTER_ODD_CZ_EDGES_LOCAL": odd_edges_local,
        "CLUSTER_EVEN_CZ_EDGES_LOCAL": even_edges_local,
        "CLUSTER_SCAFFOLD_EDGES_LOCAL": scaffold_edges_local,
        "SYSTEM": system,
        "SYSTEMS": {partition_label: system},
        "DATA_WIRES_BY_SYSTEM": {partition_label: data_wires},
        "get_system": lambda system_key=partition_label: {partition_label: system}[str(system_key)],
    }


def load_module_from_path(path_like: str | Path):
    import importlib.util
    import sys

    path = Path(path_like).resolve()
    digest = sha1(str(path).encode("utf-8")).hexdigest()[:12]
    module_name = f"partition_static_ops_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from path: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
