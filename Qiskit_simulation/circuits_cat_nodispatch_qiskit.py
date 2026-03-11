"""Qiskit circuit templates for the 2x2 cluster30 distributed VQLS workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import CCZGate, CHGate, CRYGate, CYGate
from qiskit.quantum_info import SparsePauliOp

from .static_ops_2x2_cluster30_qiskit import BStateSpec, PauliWord


_PAULI_PRODUCT = {
    ("I", "I"): (1.0 + 0.0j, "I"),
    ("I", "X"): (1.0 + 0.0j, "X"),
    ("I", "Y"): (1.0 + 0.0j, "Y"),
    ("I", "Z"): (1.0 + 0.0j, "Z"),
    ("X", "I"): (1.0 + 0.0j, "X"),
    ("Y", "I"): (1.0 + 0.0j, "Y"),
    ("Z", "I"): (1.0 + 0.0j, "Z"),
    ("X", "X"): (1.0 + 0.0j, "I"),
    ("Y", "Y"): (1.0 + 0.0j, "I"),
    ("Z", "Z"): (1.0 + 0.0j, "I"),
    ("X", "Y"): (0.0 + 1.0j, "Z"),
    ("Y", "X"): (0.0 - 1.0j, "Z"),
    ("Y", "Z"): (0.0 + 1.0j, "X"),
    ("Z", "Y"): (0.0 - 1.0j, "X"),
    ("Z", "X"): (0.0 + 1.0j, "Y"),
    ("X", "Z"): (0.0 - 1.0j, "Y"),
}


def _compile_small_gate(gate, n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    qc.append(gate, list(range(n_qubits)))
    return qc.decompose(reps=8)


_CH_TEMPLATE = _compile_small_gate(CHGate(), 2)
_CY_TEMPLATE = _compile_small_gate(CYGate(), 2)
_CCZ_TEMPLATE = _compile_small_gate(CCZGate(), 3)
_CRY_PARAM = Parameter("cry_theta")
_CRY_TEMPLATE = _compile_small_gate(CRYGate(_CRY_PARAM), 2)


def _append_ch(qc: QuantumCircuit, control: int, target: int):
    qc.compose(_CH_TEMPLATE, qubits=[control, target], inplace=True)


def _append_cy(qc: QuantumCircuit, control: int, target: int):
    qc.compose(_CY_TEMPLATE, qubits=[control, target], inplace=True)


def _append_cry(qc: QuantumCircuit, theta, control: int, target: int):
    qc.compose(_CRY_TEMPLATE.assign_parameters({_CRY_PARAM: theta}, inplace=False), qubits=[control, target], inplace=True)


def _append_ccz(qc: QuantumCircuit, control: int, target_a: int, target_b: int):
    qc.compose(_CCZ_TEMPLATE, qubits=[control, target_a, target_b], inplace=True)


@dataclass(frozen=True)
class CircuitTemplate:
    """Parameterized circuit + observable bundle for EstimatorV2."""

    name: str
    circuit: QuantumCircuit
    observables: Tuple[SparsePauliOp, ...]
    ordered_parameters: Tuple[Parameter, ...]
    blocks: Mapping[str, Tuple[Parameter, ...]]

    def pack(self, **named_values: np.ndarray) -> np.ndarray:
        if not self.ordered_parameters:
            return np.empty((0,), dtype=np.float32)
        lookup: Dict[Parameter, float] = {}
        for block_name, params in self.blocks.items():
            vals = np.asarray(named_values[block_name], dtype=np.float32).reshape(-1)
            if len(vals) != len(params):
                raise ValueError(
                    f"Template {self.name!r} expected {len(params)} values for block {block_name!r}, got {len(vals)}"
                )
            for param, val in zip(params, vals):
                lookup[param] = float(val)
        return np.asarray([lookup[p] for p in self.ordered_parameters], dtype=np.float32)

    @property
    def observable_arg(self):
        if len(self.observables) == 1:
            return self.observables[0]
        return list(self.observables)


@dataclass
class TermBundleQiskit:
    omega: CircuitTemplate
    delta: CircuitTemplate
    zeta: Dict[str, CircuitTemplate]
    tau: Dict[str, CircuitTemplate]
    beta: CircuitTemplate


def ring_edges(n_qubits: int) -> List[Tuple[int, int]]:
    if n_qubits < 2:
        return []
    edges = [(q, q + 1) for q in range(n_qubits - 1)]
    if n_qubits > 2:
        edges.append((n_qubits - 1, 0))
    return edges


def apply_basic_entangler_cz(qc: QuantumCircuit, params: Sequence, data_qubits: Sequence[int]):
    for theta, qubit in zip(params, data_qubits):
        qc.ry(theta, qubit)
    for a, b in ring_edges(len(data_qubits)):
        qc.cz(data_qubits[a], data_qubits[b])


def apply_basic_entangler_cz_inverse(qc: QuantumCircuit, params: Sequence, data_qubits: Sequence[int]):
    edges = ring_edges(len(data_qubits))
    for a, b in reversed(edges):
        qc.cz(data_qubits[a], data_qubits[b])
    for theta, qubit in reversed(list(zip(params, data_qubits))):
        qc.ry(-theta, qubit)


def apply_controlled_basic_entangler_cz(
    qc: QuantumCircuit,
    params: Sequence,
    data_qubits: Sequence[int],
    ancilla: int,
):
    for theta, qubit in zip(params, data_qubits):
        _append_cry(qc, theta, ancilla, qubit)
    for a, b in ring_edges(len(data_qubits)):
        _append_ccz(qc, ancilla, data_qubits[a], data_qubits[b])


def apply_controlled_basic_entangler_cz_inverse(
    qc: QuantumCircuit,
    params: Sequence,
    data_qubits: Sequence[int],
    ancilla: int,
):
    edges = ring_edges(len(data_qubits))
    for a, b in reversed(edges):
        _append_ccz(qc, ancilla, data_qubits[a], data_qubits[b])
    for theta, qubit in reversed(list(zip(params, data_qubits))):
        _append_cry(qc, -theta, ancilla, qubit)


def apply_pauli_word(qc: QuantumCircuit, word: PauliWord, data_qubits: Sequence[int]):
    for local_idx, label in word.ops:
        qubit = data_qubits[local_idx]
        if label == "X":
            qc.x(qubit)
        elif label == "Y":
            qc.y(qubit)
        elif label == "Z":
            qc.z(qubit)
        else:
            raise ValueError(f"Unsupported Pauli label {label!r}")


def apply_controlled_pauli_word(
    qc: QuantumCircuit,
    word: PauliWord,
    data_qubits: Sequence[int],
    ancilla: int,
):
    for local_idx, label in word.ops:
        qubit = data_qubits[local_idx]
        if label == "X":
            qc.cx(ancilla, qubit)
        elif label == "Y":
            _append_cy(qc, ancilla, qubit)
        elif label == "Z":
            qc.cz(ancilla, qubit)
        else:
            raise ValueError(f"Unsupported Pauli label {label!r}")


def apply_cluster_removed_q2_prep(qc: QuantumCircuit, data_qubits: Sequence[int], spec: BStateSpec):
    for qubit in data_qubits:
        qc.h(qubit)
    for left, right in zip(data_qubits[1:-1], data_qubits[2:]):
        qc.cz(left, right)
    for local_idx in spec.z_after_prep:
        qc.z(data_qubits[local_idx])


def apply_cluster_removed_q2_prep_inverse(qc: QuantumCircuit, data_qubits: Sequence[int], spec: BStateSpec):
    for local_idx in reversed(spec.z_after_prep):
        qc.z(data_qubits[local_idx])
    for left, right in reversed(list(zip(data_qubits[1:-1], data_qubits[2:]))):
        qc.cz(left, right)
    for qubit in reversed(list(data_qubits)):
        qc.h(qubit)


def apply_controlled_cluster_removed_q2_prep(
    qc: QuantumCircuit,
    data_qubits: Sequence[int],
    ancilla: int,
    spec: BStateSpec,
):
    for qubit in data_qubits:
        _append_ch(qc, ancilla, qubit)
    for left, right in zip(data_qubits[1:-1], data_qubits[2:]):
        _append_ccz(qc, ancilla, left, right)
    for local_idx in spec.z_after_prep:
        qc.cz(ancilla, data_qubits[local_idx])


def apply_controlled_cluster_removed_q2_prep_inverse(
    qc: QuantumCircuit,
    data_qubits: Sequence[int],
    ancilla: int,
    spec: BStateSpec,
):
    for local_idx in reversed(spec.z_after_prep):
        qc.cz(ancilla, data_qubits[local_idx])
    for left, right in reversed(list(zip(data_qubits[1:-1], data_qubits[2:]))):
        _append_ccz(qc, ancilla, left, right)
    for qubit in reversed(list(data_qubits)):
        _append_ch(qc, ancilla, qubit)


def ancilla_z_observable(n_qubits_total: int, ancilla: int = 0) -> SparsePauliOp:
    labels = ["I"] * n_qubits_total
    labels[n_qubits_total - 1 - ancilla] = "Z"
    return SparsePauliOp.from_list([("".join(labels), 1.0)])


def pauli_word_to_label(word: PauliWord, n_qubits: int) -> str:
    chars = ["I"] * n_qubits
    for local_idx, label in word.ops:
        chars[n_qubits - 1 - local_idx] = label
    return "".join(chars)


def pauli_word_to_observable(word: PauliWord, n_qubits: int, coeff: float = 1.0) -> SparsePauliOp:
    return SparsePauliOp.from_list([(pauli_word_to_label(word, n_qubits), complex(coeff))])


def multiply_pauli_words(left: PauliWord, right: PauliWord) -> Tuple[complex, PauliWord]:
    result: Dict[int, str] = {idx: label for idx, label in left.ops}
    phase = 1.0 + 0.0j
    for idx, label_r in right.ops:
        label_l = result.get(idx, "I")
        local_phase, out_label = _PAULI_PRODUCT[(label_l, label_r)]
        phase *= local_phase
        if out_label == "I":
            result.pop(idx, None)
        else:
            result[idx] = out_label
    label = "I" if not result else "".join(f"{p}{q}" for q, p in sorted(result.items()))
    return phase, PauliWord(tuple(sorted(result.items())), label)


def aggregate_pauli_operator(
    words_a: Sequence[PauliWord],
    coeffs_a: Sequence[float],
    words_b: Sequence[PauliWord] | None = None,
    coeffs_b: Sequence[float] | None = None,
) -> Tuple[List[PauliWord], np.ndarray]:
    if words_b is None:
        words_b = words_a
    if coeffs_b is None:
        coeffs_b = coeffs_a

    acc: Dict[Tuple[Tuple[int, str], ...], complex] = {}
    labels: Dict[Tuple[Tuple[int, str], ...], str] = {}
    for wa, ca in zip(words_a, coeffs_a):
        for wb, cb in zip(words_b, coeffs_b):
            phase, out = multiply_pauli_words(wa, wb)
            key = out.ops
            acc[key] = acc.get(key, 0.0 + 0.0j) + complex(ca) * complex(cb) * phase
            labels[key] = out.label

    merged_words: List[PauliWord] = []
    merged_coeffs: List[float] = []
    for key, coeff in acc.items():
        if abs(coeff) < 1e-12:
            continue
        if abs(float(np.imag(coeff))) > 1e-7:
            raise ValueError(f"Operator aggregation produced non-real coefficient {coeff} for {labels[key]}")
        merged_words.append(PauliWord(key, labels[key]))
        merged_coeffs.append(float(np.real(coeff)))

    return merged_words, np.asarray(merged_coeffs, dtype=np.float32)


def _finalize_template(
    name: str,
    circuit: QuantumCircuit,
    observables: Iterable[SparsePauliOp],
    blocks: Mapping[str, Sequence[Parameter]],
) -> CircuitTemplate:
    obs = tuple(observables)
    ordered = tuple(circuit.parameters)
    block_map = {k: tuple(v) for k, v in blocks.items()}
    return CircuitTemplate(name=name, circuit=circuit, observables=obs, ordered_parameters=ordered, blocks=block_map)


def build_expectation_template(
    *,
    n_data_qubits: int,
    observables: Sequence[SparsePauliOp],
    theta_name: str = "theta",
    template_name: str = "expectation",
) -> CircuitTemplate:
    params = ParameterVector(theta_name, n_data_qubits)
    qc = QuantumCircuit(n_data_qubits)
    data_qubits = list(range(n_data_qubits))
    apply_basic_entangler_cz(qc, params, data_qubits)
    return _finalize_template(template_name, qc, observables, {theta_name: params})


def build_overlap_template(
    *,
    n_data_qubits: int,
    left_kind: str,
    right_kind: str,
    pauli_word: PauliWord | None,
    left_name: str = "left",
    right_name: str = "right",
    left_bspec: BStateSpec | None = None,
    right_bspec: BStateSpec | None = None,
    template_name: str = "overlap",
) -> CircuitTemplate:
    n_total = n_data_qubits + 1
    anc = 0
    data_qubits = list(range(1, n_total))
    qc = QuantumCircuit(n_total)
    blocks: Dict[str, Sequence[Parameter]] = {}

    left_params = None
    right_params = None
    if left_kind == "ansatz":
        left_params = ParameterVector(left_name, n_data_qubits)
        blocks[left_name] = left_params
    elif left_kind != "bprep":
        raise ValueError(f"Unsupported left kind {left_kind!r}")

    if right_kind == "ansatz":
        right_params = ParameterVector(right_name, n_data_qubits)
        blocks[right_name] = right_params
    elif right_kind != "bprep":
        raise ValueError(f"Unsupported right kind {right_kind!r}")

    qc.h(anc)

    if right_kind == "ansatz":
        apply_controlled_basic_entangler_cz(qc, right_params, data_qubits, anc)
    else:
        if right_bspec is None:
            raise ValueError("right_bspec is required for right_kind='bprep'")
        apply_controlled_cluster_removed_q2_prep(qc, data_qubits, anc, right_bspec)

    if pauli_word is not None and pauli_word.ops:
        apply_controlled_pauli_word(qc, pauli_word, data_qubits, anc)

    if left_kind == "ansatz":
        apply_controlled_basic_entangler_cz_inverse(qc, left_params, data_qubits, anc)
    else:
        if left_bspec is None:
            raise ValueError("left_bspec is required for left_kind='bprep'")
        apply_controlled_cluster_removed_q2_prep_inverse(qc, data_qubits, anc, left_bspec)

    qc.h(anc)

    return _finalize_template(
        template_name,
        qc,
        [ancilla_z_observable(n_total, anc)],
        blocks,
    )


def build_beta_template(
    *,
    n_data_qubits: int,
    words: Sequence[PauliWord],
    coeffs: Sequence[float],
    template_name: str,
) -> CircuitTemplate:
    merged_words, merged_coeffs = aggregate_pauli_operator(words, coeffs)
    observables = [
        pauli_word_to_observable(word, n_data_qubits, coeff=float(coeff))
        for word, coeff in zip(merged_words, merged_coeffs)
    ]
    return build_expectation_template(
        n_data_qubits=n_data_qubits,
        observables=observables,
        theta_name="alpha",
        template_name=template_name,
    )


def make_term_bundle_qiskit(
    *,
    n_input_qubit: int,
    U_spec: BStateSpec,
    A_words: Sequence[PauliWord],
    coeffs: Sequence[float],
) -> TermBundleQiskit:
    zeta = {
        word.label: build_overlap_template(
            n_data_qubits=n_input_qubit,
            left_kind="ansatz",
            right_kind="ansatz",
            pauli_word=word,
            left_name="alpha",
            right_name="beta",
            template_name=f"zeta_{word.label}",
        )
        for word in A_words
    }
    tau = {
        word.label: build_overlap_template(
            n_data_qubits=n_input_qubit,
            left_kind="bprep",
            right_kind="ansatz",
            pauli_word=word,
            left_bspec=U_spec,
            right_name="alpha",
            template_name=f"tau_{word.label}_{U_spec.label}",
        )
        for word in A_words
    }
    return TermBundleQiskit(
        omega=build_overlap_template(
            n_data_qubits=n_input_qubit,
            left_kind="ansatz",
            right_kind="ansatz",
            pauli_word=None,
            left_name="left",
            right_name="right",
            template_name="omega",
        ),
        delta=build_overlap_template(
            n_data_qubits=n_input_qubit,
            left_kind="bprep",
            right_kind="ansatz",
            pauli_word=None,
            left_bspec=U_spec,
            right_name="beta",
            template_name=f"delta_{U_spec.label}",
        ),
        zeta=zeta,
        tau=tau,
        beta=build_beta_template(
            n_data_qubits=n_input_qubit,
            words=A_words,
            coeffs=coeffs,
            template_name="beta",
        ),
    )
