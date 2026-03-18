"""Qiskit circuit templates for the 2x2 distributed VQLS workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import CCZGate, CHGate, CRYGate, CYGate
from qiskit.quantum_info import SparsePauliOp

from .static_ops_2x2_cluster30_qiskit import BStateSpec, PauliWord


ANSATZ_CLUSTER_H_CZ_RY = "cluster_h_cz_ry"
ANSATZ_BRICKWALL_RY_CZ = "brickwall_ry_cz"


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


def _normalize_ansatz_kind(ansatz_kind: str | None) -> str:
    kind = str(ansatz_kind or ANSATZ_CLUSTER_H_CZ_RY).strip().lower()
    if kind not in (ANSATZ_CLUSTER_H_CZ_RY, ANSATZ_BRICKWALL_RY_CZ):
        raise ValueError(f"Unsupported ansatz kind {ansatz_kind!r}")
    return kind


def cluster_scaffold_edges(
    n_qubits: int,
    scaffold_edges: Sequence[Tuple[int, int]] | None = None,
) -> List[Tuple[int, int]]:
    if scaffold_edges is None:
        if n_qubits < 3:
            return []
        return [(q, q + 1) for q in range(1, n_qubits - 1)]

    edges: List[Tuple[int, int]] = []
    for pair in scaffold_edges:
        if len(pair) != 2:
            raise ValueError(f"Invalid scaffold edge {pair!r}")
        left, right = int(pair[0]), int(pair[1])
        if left < 0 or right < 0 or left >= n_qubits or right >= n_qubits or left == right:
            raise ValueError(f"Out-of-range scaffold edge {(left, right)} for n_qubits={n_qubits}")
        edges.append((left, right))
    return edges

def _split_layer_params(params: Sequence, n_layers: int, n_data_qubits: int) -> List[Sequence]:
    flat = list(params)
    expected = int(n_layers) * int(n_data_qubits)
    if len(flat) != expected:
        raise ValueError(f"Expected {expected} ansatz parameters, got {len(flat)}")
    return [flat[idx * n_data_qubits : (idx + 1) * n_data_qubits] for idx in range(n_layers)]

def apply_basic_entangler_cz(
    qc: QuantumCircuit,
    params: Sequence,
    data_qubits: Sequence[int],
    *,
    layers: int = 1,
    repeat_cz_each_layer: bool = False,
    ansatz_kind: str = ANSATZ_CLUSTER_H_CZ_RY,
    scaffold_edges: Sequence[Tuple[int, int]] | None = None,
):
    kind = _normalize_ansatz_kind(ansatz_kind)
    edges = cluster_scaffold_edges(len(data_qubits), scaffold_edges=scaffold_edges)
    layer_params = _split_layer_params(params, int(layers), len(data_qubits))

    if kind == ANSATZ_BRICKWALL_RY_CZ:
        for params_layer in layer_params:
            for theta, qubit in zip(params_layer, data_qubits):
                qc.ry(theta, qubit)
            for a, b in edges:
                qc.cz(data_qubits[a], data_qubits[b])
        return

    for qubit in data_qubits:
        qc.h(qubit)

    if not repeat_cz_each_layer:
        for a, b in edges:
            qc.cz(data_qubits[a], data_qubits[b])

    for params_layer in layer_params:
        if repeat_cz_each_layer:
            for a, b in edges:
                qc.cz(data_qubits[a], data_qubits[b])
        for theta, qubit in zip(params_layer, data_qubits):
            qc.ry(theta, qubit)


def apply_basic_entangler_cz_inverse(
    qc: QuantumCircuit,
    params: Sequence,
    data_qubits: Sequence[int],
    *,
    layers: int = 1,
    repeat_cz_each_layer: bool = False,
    ansatz_kind: str = ANSATZ_CLUSTER_H_CZ_RY,
    scaffold_edges: Sequence[Tuple[int, int]] | None = None,
):
    kind = _normalize_ansatz_kind(ansatz_kind)
    edges = cluster_scaffold_edges(len(data_qubits), scaffold_edges=scaffold_edges)
    layer_params = _split_layer_params(params, int(layers), len(data_qubits))

    if kind == ANSATZ_BRICKWALL_RY_CZ:
        for params_layer in reversed(layer_params):
            for a, b in reversed(edges):
                qc.cz(data_qubits[a], data_qubits[b])
            for theta, qubit in reversed(list(zip(params_layer, data_qubits))):
                qc.ry(-theta, qubit)
        return

    for params_layer in reversed(layer_params):
        for theta, qubit in reversed(list(zip(params_layer, data_qubits))):
            qc.ry(-theta, qubit)
        if repeat_cz_each_layer:
            for a, b in reversed(edges):
                qc.cz(data_qubits[a], data_qubits[b])

    if not repeat_cz_each_layer:
        for a, b in reversed(edges):
            qc.cz(data_qubits[a], data_qubits[b])

    for qubit in reversed(list(data_qubits)):
        qc.h(qubit)


def apply_controlled_basic_entangler_cz(
    qc: QuantumCircuit,
    params: Sequence,
    data_qubits: Sequence[int],
    ancilla: int,
    *,
    layers: int = 1,
    repeat_cz_each_layer: bool = False,
    ansatz_kind: str = ANSATZ_CLUSTER_H_CZ_RY,
    scaffold_edges: Sequence[Tuple[int, int]] | None = None,
):
    kind = _normalize_ansatz_kind(ansatz_kind)
    edges = cluster_scaffold_edges(len(data_qubits), scaffold_edges=scaffold_edges)
    layer_params = _split_layer_params(params, int(layers), len(data_qubits))

    if kind == ANSATZ_BRICKWALL_RY_CZ:
        for params_layer in layer_params:
            for theta, qubit in zip(params_layer, data_qubits):
                _append_cry(qc, theta, ancilla, qubit)
            for a, b in edges:
                _append_ccz(qc, ancilla, data_qubits[a], data_qubits[b])
        return

    for qubit in data_qubits:
        _append_ch(qc, ancilla, qubit)

    if not repeat_cz_each_layer:
        for a, b in edges:
            _append_ccz(qc, ancilla, data_qubits[a], data_qubits[b])

    for params_layer in layer_params:
        if repeat_cz_each_layer:
            for a, b in edges:
                _append_ccz(qc, ancilla, data_qubits[a], data_qubits[b])
        for theta, qubit in zip(params_layer, data_qubits):
            _append_cry(qc, theta, ancilla, qubit)


def apply_controlled_basic_entangler_cz_inverse(
    qc: QuantumCircuit,
    params: Sequence,
    data_qubits: Sequence[int],
    ancilla: int,
    *,
    layers: int = 1,
    repeat_cz_each_layer: bool = False,
    ansatz_kind: str = ANSATZ_CLUSTER_H_CZ_RY,
    scaffold_edges: Sequence[Tuple[int, int]] | None = None,
):
    kind = _normalize_ansatz_kind(ansatz_kind)
    edges = cluster_scaffold_edges(len(data_qubits), scaffold_edges=scaffold_edges)
    layer_params = _split_layer_params(params, int(layers), len(data_qubits))

    if kind == ANSATZ_BRICKWALL_RY_CZ:
        for params_layer in reversed(layer_params):
            for a, b in reversed(edges):
                _append_ccz(qc, ancilla, data_qubits[a], data_qubits[b])
            for theta, qubit in reversed(list(zip(params_layer, data_qubits))):
                _append_cry(qc, -theta, ancilla, qubit)
        return

    for params_layer in reversed(layer_params):
        for theta, qubit in reversed(list(zip(params_layer, data_qubits))):
            _append_cry(qc, -theta, ancilla, qubit)
        if repeat_cz_each_layer:
            for a, b in reversed(edges):
                _append_ccz(qc, ancilla, data_qubits[a], data_qubits[b])

    if not repeat_cz_each_layer:
        for a, b in reversed(edges):
            _append_ccz(qc, ancilla, data_qubits[a], data_qubits[b])

    for qubit in reversed(list(data_qubits)):
        _append_ch(qc, ancilla, qubit)


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


def _bprep_edges(data_qubits: Sequence[int], spec: BStateSpec) -> List[Tuple[int, int]]:
    label = str(spec.label)
    if "cluster_removed_q1" in label:
        return list(zip(data_qubits[:-1], data_qubits[1:]))
    if "cluster_removed_q2" in label:
        return list(zip(data_qubits[1:-1], data_qubits[2:]))
    raise ValueError(f"Unsupported B-state preparation label {label!r}")


def apply_cluster_bprep(qc: QuantumCircuit, data_qubits: Sequence[int], spec: BStateSpec):
    for qubit in data_qubits:
        qc.h(qubit)
    for left, right in _bprep_edges(data_qubits, spec):
        qc.cz(left, right)
    for local_idx in spec.z_after_prep:
        qc.z(data_qubits[local_idx])


def apply_cluster_bprep_inverse(qc: QuantumCircuit, data_qubits: Sequence[int], spec: BStateSpec):
    for local_idx in reversed(spec.z_after_prep):
        qc.z(data_qubits[local_idx])
    for left, right in reversed(_bprep_edges(data_qubits, spec)):
        qc.cz(left, right)
    for qubit in reversed(list(data_qubits)):
        qc.h(qubit)


def apply_controlled_cluster_bprep(
    qc: QuantumCircuit,
    data_qubits: Sequence[int],
    ancilla: int,
    spec: BStateSpec,
):
    for qubit in data_qubits:
        _append_ch(qc, ancilla, qubit)
    for left, right in _bprep_edges(data_qubits, spec):
        _append_ccz(qc, ancilla, left, right)
    for local_idx in spec.z_after_prep:
        qc.cz(ancilla, data_qubits[local_idx])


def apply_controlled_cluster_bprep_inverse(
    qc: QuantumCircuit,
    data_qubits: Sequence[int],
    ancilla: int,
    spec: BStateSpec,
):
    for local_idx in reversed(spec.z_after_prep):
        qc.cz(ancilla, data_qubits[local_idx])
    for left, right in reversed(_bprep_edges(data_qubits, spec)):
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
    layers: int = 1,
    repeat_cz_each_layer: bool = False,
    ansatz_kind: str = ANSATZ_CLUSTER_H_CZ_RY,
    scaffold_edges: Sequence[Tuple[int, int]] | None = None,
    theta_name: str = "theta",
    template_name: str = "expectation",
) -> CircuitTemplate:
    params = ParameterVector(theta_name, int(layers) * n_data_qubits)
    qc = QuantumCircuit(n_data_qubits)
    data_qubits = list(range(n_data_qubits))
    apply_basic_entangler_cz(
        qc,
        params,
        data_qubits,
        layers=layers,
        repeat_cz_each_layer=repeat_cz_each_layer,
        ansatz_kind=ansatz_kind,
        scaffold_edges=scaffold_edges,
    )
    return _finalize_template(template_name, qc, observables, {theta_name: params})


def build_overlap_template(
    *,
    n_data_qubits: int,
    layers: int = 1,
    repeat_cz_each_layer: bool = False,
    ansatz_kind: str = ANSATZ_CLUSTER_H_CZ_RY,
    scaffold_edges: Sequence[Tuple[int, int]] | None = None,
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
        left_params = ParameterVector(left_name, int(layers) * n_data_qubits)
        blocks[left_name] = left_params
    elif left_kind != "bprep":
        raise ValueError(f"Unsupported left kind {left_kind!r}")

    if right_kind == "ansatz":
        right_params = ParameterVector(right_name, int(layers) * n_data_qubits)
        blocks[right_name] = right_params
    elif right_kind != "bprep":
        raise ValueError(f"Unsupported right kind {right_kind!r}")

    qc.h(anc)

    if right_kind == "ansatz":
        apply_controlled_basic_entangler_cz(
            qc,
            right_params,
            data_qubits,
            anc,
            layers=layers,
            repeat_cz_each_layer=repeat_cz_each_layer,
            ansatz_kind=ansatz_kind,
            scaffold_edges=scaffold_edges,
        )
    else:
        if right_bspec is None:
            raise ValueError("right_bspec is required for right_kind='bprep'")
        apply_controlled_cluster_bprep(qc, data_qubits, anc, right_bspec)

    if pauli_word is not None and pauli_word.ops:
        apply_controlled_pauli_word(qc, pauli_word, data_qubits, anc)

    if left_kind == "ansatz":
        apply_controlled_basic_entangler_cz_inverse(
            qc,
            left_params,
            data_qubits,
            anc,
            layers=layers,
            repeat_cz_each_layer=repeat_cz_each_layer,
            ansatz_kind=ansatz_kind,
            scaffold_edges=scaffold_edges,
        )
    else:
        if left_bspec is None:
            raise ValueError("left_bspec is required for left_kind='bprep'")
        apply_controlled_cluster_bprep_inverse(qc, data_qubits, anc, left_bspec)

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
    layers: int = 1,
    repeat_cz_each_layer: bool = False,
    ansatz_kind: str = ANSATZ_CLUSTER_H_CZ_RY,
    scaffold_edges: Sequence[Tuple[int, int]] | None = None,
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
        layers=layers,
        repeat_cz_each_layer=repeat_cz_each_layer,
        ansatz_kind=ansatz_kind,
        scaffold_edges=scaffold_edges,
        theta_name="alpha",
        template_name=template_name,
    )


def make_term_bundle_qiskit(
    *,
    n_input_qubit: int,
    U_spec: BStateSpec,
    A_words: Sequence[PauliWord],
    coeffs: Sequence[float],
    layers: int = 1,
    repeat_cz_each_layer: bool = False,
    ansatz_kind: str = ANSATZ_CLUSTER_H_CZ_RY,
    scaffold_edges: Sequence[Tuple[int, int]] | None = None,
) -> TermBundleQiskit:
    zeta = {
        word.label: build_overlap_template(
            n_data_qubits=n_input_qubit,
            layers=layers,
            repeat_cz_each_layer=repeat_cz_each_layer,
            ansatz_kind=ansatz_kind,
            scaffold_edges=scaffold_edges,
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
            layers=layers,
            repeat_cz_each_layer=repeat_cz_each_layer,
            ansatz_kind=ansatz_kind,
            scaffold_edges=scaffold_edges,
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
            layers=layers,
            repeat_cz_each_layer=repeat_cz_each_layer,
            ansatz_kind=ansatz_kind,
            scaffold_edges=scaffold_edges,
            left_kind="ansatz",
            right_kind="ansatz",
            pauli_word=None,
            left_name="left",
            right_name="right",
            template_name="omega",
        ),
        delta=build_overlap_template(
            n_data_qubits=n_input_qubit,
            layers=layers,
            repeat_cz_each_layer=repeat_cz_each_layer,
            ansatz_kind=ansatz_kind,
            scaffold_edges=scaffold_edges,
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
            layers=layers,
            repeat_cz_each_layer=repeat_cz_each_layer,
            ansatz_kind=ansatz_kind,
            scaffold_edges=scaffold_edges,
            template_name="beta",
        ),
    )
