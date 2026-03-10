#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import yaml


PauliWord = Tuple[str, ...]


DEFAULT_CONFIG = {
    "problem": {
        "static_ops_path": "problems/static_ops_3net_Ising.py",
        "system_key": "4x4",
        "prefer_centralized_problem": True,
        "centralized_problem_key": "centralized",
        "consistency_system_key": "4x4",
        "consistency_atol": 1.0e-12,
        "b_consistency_atol": 1.0e-12,
        "b_state_tolerance": 1.0e-10,
    },
    "ansatz": {
        "layers": 5,
        "init_low": -3.141592653589793,
        "init_high": 3.141592653589793,
    },
    "optimization": {
        "steps": 200,
        "learning_rate": 0.03,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1.0e-8,
        "print_every": 20,
        "optimize_metric": "global_CG",
    },
    "runtime": {
        "seed": 0,
        "device": "default.qubit",
        "interface": "autograd",
        "diff_method": "backprop",
    },
    "report": {
        "out_dir": "cen_vqls/reports",
        "tag": "centralized_vqls_ising",
    },
}


_PAULI_MUL_TABLE: Dict[Tuple[str, str], Tuple[complex, str]] = {
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
    ("X", "Z"): (0.0 - 1.0j, "Y"),
    ("Z", "X"): (0.0 + 1.0j, "Y"),
    ("Y", "Z"): (0.0 + 1.0j, "X"),
    ("Z", "Y"): (0.0 - 1.0j, "X"),
}


def _deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: Path) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Top-level YAML config must be a mapping.")
    _deep_update(cfg, loaded)
    return cfg


def _multiply_pauli_words(lhs: PauliWord, rhs: PauliWord) -> Tuple[complex, PauliWord]:
    phase = 1.0 + 0.0j
    out = []
    for a, b in zip(lhs, rhs):
        local_phase, c = _PAULI_MUL_TABLE[(a, b)]
        phase = phase * local_phase
        out.append(c)
    return phase, tuple(out)


def _word_with_single_pauli(n_qubits: int, wire: int, pauli: str) -> PauliWord:
    chars = ["I"] * n_qubits
    chars[wire] = pauli
    return tuple(chars)


def _apply_controlled_pauli_word(word: PauliWord, control_wire: int, system_wires: Tuple[int, ...]) -> None:
    for p, w in zip(word, system_wires):
        if p == "X":
            qml.CNOT(wires=[control_wire, w])
        elif p == "Y":
            qml.CY(wires=[control_wire, w])
        elif p == "Z":
            qml.CZ(wires=[control_wire, w])


def _to_float(x) -> float:
    return float(np.asarray(x).real)


def _hadamard_tensor_unitary(n_qubits: int) -> np.ndarray:
    h1 = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
    out = h1
    for _ in range(n_qubits - 1):
        out = np.kron(out, h1)
    return out


def _householder_unitary_from_state(state: np.ndarray) -> np.ndarray:
    v = np.asarray(state, dtype=np.complex128)
    dim = v.shape[0]
    if dim == 0:
        raise ValueError("State vector cannot be empty.")

    v = v / np.linalg.norm(v)
    e1 = np.zeros(dim, dtype=np.complex128)
    e1[0] = 1.0

    if np.linalg.norm(v - e1) < 1.0e-14:
        return np.eye(dim, dtype=np.complex128)

    phi = np.angle(v[0])
    w = np.exp(-1.0j * phi) * v
    u = e1 - w
    denom = np.vdot(u, u)
    if np.abs(denom) < 1.0e-14:
        return np.eye(dim, dtype=np.complex128)

    h = np.eye(dim, dtype=np.complex128) - 2.0 * np.outer(u, np.conj(u)) / denom
    return np.exp(1.0j * phi) * h


def _align_global_phase(vec: np.ndarray, target: np.ndarray) -> np.ndarray:
    overlap = np.vdot(vec, target)
    if np.abs(overlap) < 1.0e-16:
        return vec
    return vec * np.exp(-1.0j * np.angle(overlap))


def _global_ansatz(weights: np.ndarray, wires: Tuple[int, ...] | range | List[int]) -> None:
    qml.BasicEntanglerLayers(weights, wires=wires, rotation=qml.RY)


@dataclass
class BUnitaryInfo:
    mode: str
    unitary: np.ndarray
    state_prep_error: float
    hadamard_match_error: float


@dataclass
class CentralizedIsingData:
    static_ops_path: Path
    problem_source: str
    reference_system_key: str
    n_agents: int
    n_index_qubits: int
    n_data_qubits: int
    n_total_qubits: int
    global_dim: int
    j: float
    h: float
    eta: float
    zeta: float
    a_block: np.ndarray
    condition_number: float
    a_formula: np.ndarray
    matrix_max_abs_diff: float
    matrix_fro_diff: float
    matrix_allclose: bool
    terms: List[Tuple[float, PauliWord]]
    b_unnorm: np.ndarray
    b_normed: np.ndarray
    b_unnorm_norm: float
    b_unitary_info: BUnitaryInfo
    reference_matrix_max_abs_diff: float
    reference_matrix_fro_diff: float
    reference_matrix_allclose: bool
    reference_b_max_abs_diff: float
    reference_b_l2_diff: float
    reference_b_allclose: bool


def load_static_system(static_ops_path: Path):
    spec = importlib.util.spec_from_file_location("static_ops_module", str(static_ops_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {static_ops_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_system_from_module(module, system_key: str):
    key = str(system_key)
    if hasattr(module, "SYSTEMS"):
        systems = getattr(module, "SYSTEMS")
        if key not in systems:
            raise RuntimeError(
                f"Unknown system key `{key}` in {module.__name__}. "
                f"Available: {sorted(systems.keys())}"
            )
        return systems[key]

    if not hasattr(module, "SYSTEM"):
        raise RuntimeError(f"{module.__name__} does not expose SYSTEM/SYSTEMS.")

    if key not in ("", "default", "4x4"):
        raise RuntimeError(
            f"{module.__name__} only exposes default SYSTEM; cannot select system key `{key}`."
        )
    return module.SYSTEM


def _resolve_data_wires_from_module(module, system_key: str, system, fallback_n_data_qubits: int):
    key = str(system_key)
    if hasattr(module, "DATA_WIRES_BY_SYSTEM"):
        wires_map = getattr(module, "DATA_WIRES_BY_SYSTEM")
        if key in wires_map:
            return list(wires_map[key])

    if hasattr(system, "data_wires"):
        return list(system.data_wires)

    if hasattr(module, "DATA_WIRES"):
        return list(getattr(module, "DATA_WIRES"))

    return list(range(int(fallback_n_data_qubits)))


def _resolve_centralized_problem(module, centralized_problem_key: str):
    key = str(centralized_problem_key)
    if hasattr(module, "CENTRALIZED_PROBLEMS"):
        mapping = getattr(module, "CENTRALIZED_PROBLEMS")
        if key in mapping:
            return mapping[key]
        return None

    if hasattr(module, "CENTRALIZED_PROBLEM") and key in ("centralized", "default", ""):
        return getattr(module, "CENTRALIZED_PROBLEM")

    return None


def _normalize_pauli_word(word) -> PauliWord:
    if isinstance(word, tuple):
        chars = tuple(str(x) for x in word)
    elif isinstance(word, list):
        chars = tuple(str(x) for x in word)
    elif isinstance(word, str):
        chars = tuple(word)
    else:
        raise RuntimeError(f"Unsupported Pauli word format: {type(word)}")

    for c in chars:
        if c not in {"I", "X", "Y", "Z"}:
            raise RuntimeError(f"Invalid Pauli label `{c}` in word `{word}`")
    return chars


def _normalize_terms(terms_obj) -> List[Tuple[float, PauliWord]]:
    if not isinstance(terms_obj, (list, tuple)):
        raise RuntimeError("`terms` must be a list/tuple of (coeff, pauli_word).")
    out: List[Tuple[float, PauliWord]] = []
    for item in terms_obj:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise RuntimeError(f"Invalid term entry: {item}")
        coeff, word = item
        out.append((float(coeff), _normalize_pauli_word(word)))
    return out


def build_formula_terms(n_total_qubits: int, eta: float, j: float, h: float, zeta: float) -> List[Tuple[float, PauliWord]]:
    terms: List[Tuple[float, PauliWord]] = []
    terms.append((float(eta / zeta), tuple("I" for _ in range(n_total_qubits))))

    for wire in range(n_total_qubits):
        chars = ["I"] * n_total_qubits
        chars[wire] = "X"
        terms.append((float(1.0 / zeta), tuple(chars)))

    if n_total_qubits < 3:
        raise ValueError("Need at least 3 qubits for the Z0Z1 and Z1Z2 terms.")

    chars = ["I"] * n_total_qubits
    chars[0] = "Z"
    chars[1] = "Z"
    terms.append((float(j / zeta), tuple(chars)))

    chars = ["I"] * n_total_qubits
    chars[1] = "Z"
    chars[2] = "Z"
    terms.append((float(h / zeta), tuple(chars)))

    return terms


def _pauli_word_to_dense(word: PauliWord) -> np.ndarray:
    pauli = {
        "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
        "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
        "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
    }
    out = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    for p in word:
        out = np.kron(out, pauli[p])
    return out


def _data_gate_to_pauli_word(gate_fn, data_wires: List[int]) -> PauliWord:
    wire_to_pos = {int(w): i for i, w in enumerate(data_wires)}
    word = ["I"] * len(data_wires)

    def visit(node):
        if hasattr(node, "operands"):
            for child in node.operands:
                visit(child)
            return

        name = node.name
        if name == "Identity":
            return
        if name in ("PauliX", "PauliY", "PauliZ"):
            wire = int(node.wires[0])
            if wire not in wire_to_pos:
                raise ValueError(f"Data gate wire {wire} not in DATA_WIRES.")
            pos = wire_to_pos[wire]
            lbl = name[-1]  # X / Y / Z
            if word[pos] != "I" and word[pos] != lbl:
                raise ValueError(f"Non-Pauli-compressible product on data wire {wire}.")
            word[pos] = lbl
            return
        raise ValueError(f"Unsupported gate in distributed block decomposition: {name}")

    op = gate_fn()
    visit(op)
    return tuple(word)


def build_terms_from_distributed_system(
    system,
    data_wires: List[int],
    *,
    atol: float = 1.0e-12,
) -> List[Tuple[float, PauliWord]]:
    """
    Convert distributed block decomposition A_ij = sum_m c_ijm G_ijm
    into centralized Pauli-word LCU terms:
      A = sum_l c_l P_l
    where each P_l acts on (index_qubits + data_qubits).
    """
    n_agents = int(system.n)
    n_index_qubits = int(round(np.log2(n_agents)))
    if 2**n_index_qubits != n_agents:
        raise RuntimeError(f"Expected number of agents to be power of 2, got {n_agents}.")

    coeff_mats_by_data_word: Dict[PauliWord, np.ndarray] = {}
    for i in range(n_agents):
        for j in range(n_agents):
            gates = system.gates_grid[i][j]
            coeffs = system.coeffs[i][j]
            if len(gates) != len(coeffs):
                raise RuntimeError(f"Mismatch len(gates) != len(coeffs) at block ({i},{j}).")
            for gate_fn, coeff in zip(gates, coeffs):
                dword = _data_gate_to_pauli_word(gate_fn, data_wires=data_wires)
                if dword not in coeff_mats_by_data_word:
                    coeff_mats_by_data_word[dword] = np.zeros((n_agents, n_agents), dtype=np.complex128)
                coeff_mats_by_data_word[dword][i, j] += np.complex128(coeff)

    index_basis = [tuple(x) for x in itertools.product("IXYZ", repeat=n_index_qubits)]
    index_dense = {w: _pauli_word_to_dense(w) for w in index_basis}

    coeff_by_full_word: Dict[PauliWord, np.complex128] = {}
    for dword, cmat in coeff_mats_by_data_word.items():
        for iword, pmat in index_dense.items():
            alpha = np.trace(pmat.conj().T @ cmat) / float(n_agents)
            if np.abs(alpha) <= atol:
                continue
            fword = iword + dword
            coeff_by_full_word[fword] = coeff_by_full_word.get(fword, 0.0 + 0.0j) + alpha

    terms: List[Tuple[float, PauliWord]] = []
    for word in sorted(coeff_by_full_word.keys(), key=lambda w: "".join(w)):
        coeff = coeff_by_full_word[word]
        if np.abs(coeff) <= atol:
            continue
        if np.abs(np.imag(coeff)) > max(1.0e-11, 10.0 * atol):
            raise RuntimeError(f"Complex term coefficient found for word {word}: {coeff}")
        terms.append((float(np.real(coeff)), word))
    return terms


def pauli_word_to_operator(word: PauliWord):
    factors = []
    for wire, p in enumerate(word):
        if p == "I":
            continue
        if p == "X":
            factors.append(qml.PauliX(wires=wire))
        elif p == "Y":
            factors.append(qml.PauliY(wires=wire))
        elif p == "Z":
            factors.append(qml.PauliZ(wires=wire))
        else:
            raise ValueError(f"Unsupported Pauli op: {p}")

    if not factors:
        return qml.Identity(wires=0)
    if len(factors) == 1:
        return factors[0]
    return qml.prod(*factors)


def build_formula_matrix(n_total_qubits: int, terms: List[Tuple[float, PauliWord]]) -> np.ndarray:
    op_terms = [qml.s_prod(coeff, pauli_word_to_operator(word)) for coeff, word in terms]
    op = qml.sum(*op_terms)
    return np.array(qml.matrix(op, wire_order=list(range(n_total_qubits))), dtype=np.complex128)


def build_b_unitary_from_distributed(b_normed: np.ndarray, n_total_qubits: int, tol: float) -> BUnitaryInfo:
    hadamard_target = np.ones(2**n_total_qubits, dtype=np.complex128) / np.sqrt(2**n_total_qubits)
    hadamard_match_error = float(np.linalg.norm(b_normed - hadamard_target))

    if hadamard_match_error <= tol:
        u = _hadamard_tensor_unitary(n_total_qubits)
        mode = "hadamard_tensor"
    else:
        u = _householder_unitary_from_state(b_normed)
        mode = "householder"

    prepared = u[:, 0]
    prepared_aligned = _align_global_phase(prepared, b_normed)
    prep_error = float(np.linalg.norm(prepared_aligned - b_normed))
    return BUnitaryInfo(mode=mode, unitary=u, state_prep_error=prep_error, hadamard_match_error=hadamard_match_error)


def load_centralized_data(cfg: dict, repo_root: Path) -> CentralizedIsingData:
    problem_cfg = cfg["problem"]
    static_ops_path = Path(problem_cfg["static_ops_path"])
    if not static_ops_path.is_absolute():
        static_ops_path = repo_root / static_ops_path
    static_ops_path = static_ops_path.resolve()

    module = load_static_system(static_ops_path)
    atol = float(problem_cfg.get("consistency_atol", 1.0e-12))
    b_atol = float(problem_cfg.get("b_consistency_atol", atol))
    system_key = str(problem_cfg.get("system_key", "4x4"))
    reference_system_key = str(problem_cfg.get("consistency_system_key", system_key))
    prefer_centralized_problem = bool(problem_cfg.get("prefer_centralized_problem", False))
    centralized_problem_key = str(problem_cfg.get("centralized_problem_key", "centralized"))

    reference_system = _resolve_system_from_module(module, reference_system_key)
    reference_n_agents = int(reference_system.n)
    reference_n_index = int(round(np.log2(reference_n_agents)))
    if 2**reference_n_index != reference_n_agents:
        raise RuntimeError(
            f"Expected reference system agent count to be power of 2, got {reference_n_agents}."
        )
    reference_n_data = int(
        getattr(reference_system, "n_data_qubits", getattr(module, "N_DATA_QUBITS", 0))
    )
    reference_data_wires = _resolve_data_wires_from_module(
        module,
        reference_system_key,
        reference_system,
        fallback_n_data_qubits=reference_n_data,
    )
    reference_a = np.array(reference_system.get_global_matrix(), dtype=np.complex128)
    reference_b = np.array(reference_system.get_global_b_vector(), dtype=np.complex128)

    problem_source = f"distributed_system[{system_key}]"
    centralized_problem = None
    if prefer_centralized_problem:
        centralized_problem = _resolve_centralized_problem(module, centralized_problem_key)

    if centralized_problem is not None:
        if "a_matrix" not in centralized_problem or "b_vector" not in centralized_problem:
            raise RuntimeError(
                "Centralized problem must contain `a_matrix` and `b_vector`."
            )
        problem_source = f"module.CENTRALIZED_PROBLEMS[{centralized_problem_key}]"
        a_block = np.array(centralized_problem["a_matrix"], dtype=np.complex128)
        b_unnorm = np.array(centralized_problem["b_vector"], dtype=np.complex128)

        if "terms" in centralized_problem and centralized_problem["terms"] is not None:
            terms = _normalize_terms(centralized_problem["terms"])
        else:
            terms = build_terms_from_distributed_system(
                system=reference_system,
                data_wires=reference_data_wires,
                atol=atol,
            )

        n_total_qubits = int(
            centralized_problem.get(
                "n_total_qubits", int(round(np.log2(a_block.shape[0])))
            )
        )
        n_index_qubits = int(
            centralized_problem.get("n_index_qubits", reference_n_index)
        )
        n_data_qubits = int(
            centralized_problem.get("n_data_qubits", n_total_qubits - n_index_qubits)
        )
        n_agents = int(centralized_problem.get("n_agents", 2**n_index_qubits))
    else:
        system = _resolve_system_from_module(module, system_key)
        n_agents = int(system.n)
        n_index_qubits = int(round(np.log2(n_agents)))
        if 2**n_index_qubits != n_agents:
            raise RuntimeError(f"Expected number of agents to be power of 2, got {n_agents}.")
        n_data_qubits = int(
            getattr(system, "n_data_qubits", getattr(module, "N_DATA_QUBITS", 0))
        )
        n_total_qubits = n_index_qubits + n_data_qubits

        data_wires = _resolve_data_wires_from_module(
            module,
            system_key,
            system,
            fallback_n_data_qubits=n_data_qubits,
        )
        a_block = np.array(system.get_global_matrix(), dtype=np.complex128)
        b_unnorm = np.array(system.get_global_b_vector(), dtype=np.complex128)
        terms = build_terms_from_distributed_system(
            system=system,
            data_wires=data_wires,
            atol=atol,
        )

    global_dim = int(a_block.shape[0])

    j = float(getattr(module, "J", 0.0))
    h = float(getattr(module, "h", 0.0))
    eta = float(getattr(module, "eta", 0.0))
    zeta = float(getattr(module, "zeta", 1.0))

    condition_number = float(np.linalg.cond(a_block))
    a_formula = build_formula_matrix(n_total_qubits=n_total_qubits, terms=terms)

    diff = a_formula - a_block
    matrix_max_abs_diff = float(np.max(np.abs(diff)))
    matrix_fro_diff = float(np.linalg.norm(diff))
    matrix_allclose = bool(np.allclose(a_formula, a_block, atol=atol, rtol=0.0))

    if reference_a.shape != a_block.shape:
        raise RuntimeError(
            f"Reference/system A shape mismatch: target {a_block.shape} vs reference {reference_a.shape}"
        )
    if reference_b.shape != b_unnorm.shape:
        raise RuntimeError(
            f"Reference/system b shape mismatch: target {b_unnorm.shape} vs reference {reference_b.shape}"
        )

    ref_a_diff = a_block - reference_a
    reference_matrix_max_abs_diff = float(np.max(np.abs(ref_a_diff)))
    reference_matrix_fro_diff = float(np.linalg.norm(ref_a_diff))
    reference_matrix_allclose = bool(
        np.allclose(a_block, reference_a, atol=atol, rtol=0.0)
    )

    ref_b_diff = b_unnorm - reference_b
    reference_b_max_abs_diff = float(np.max(np.abs(ref_b_diff)))
    reference_b_l2_diff = float(np.linalg.norm(ref_b_diff))
    reference_b_allclose = bool(
        np.allclose(b_unnorm, reference_b, atol=b_atol, rtol=0.0)
    )

    b_unnorm_norm = float(np.linalg.norm(b_unnorm))
    if b_unnorm_norm <= 0.0:
        raise RuntimeError("Unnormalized b has zero norm.")
    b_normed = b_unnorm / b_unnorm_norm

    b_unitary_info = build_b_unitary_from_distributed(
        b_normed=b_normed,
        n_total_qubits=n_total_qubits,
        tol=float(problem_cfg["b_state_tolerance"]),
    )

    return CentralizedIsingData(
        static_ops_path=static_ops_path,
        problem_source=problem_source,
        reference_system_key=reference_system_key,
        n_agents=n_agents,
        n_index_qubits=n_index_qubits,
        n_data_qubits=n_data_qubits,
        n_total_qubits=n_total_qubits,
        global_dim=global_dim,
        j=j,
        h=h,
        eta=eta,
        zeta=zeta,
        a_block=a_block,
        condition_number=condition_number,
        a_formula=a_formula,
        matrix_max_abs_diff=matrix_max_abs_diff,
        matrix_fro_diff=matrix_fro_diff,
        matrix_allclose=matrix_allclose,
        terms=terms,
        b_unnorm=b_unnorm,
        b_normed=b_normed,
        b_unnorm_norm=b_unnorm_norm,
        b_unitary_info=b_unitary_info,
        reference_matrix_max_abs_diff=reference_matrix_max_abs_diff,
        reference_matrix_fro_diff=reference_matrix_fro_diff,
        reference_matrix_allclose=reference_matrix_allclose,
        reference_b_max_abs_diff=reference_b_max_abs_diff,
        reference_b_l2_diff=reference_b_l2_diff,
        reference_b_allclose=reference_b_allclose,
    )


class HadamardCentralizedVQLS:
    def __init__(
        self,
        data: CentralizedIsingData,
        device_name: str,
        interface: str,
        diff_method: str,
    ):
        self.data = data
        self.n = data.n_total_qubits
        self.terms = data.terms
        self.identity_word = tuple("I" for _ in range(self.n))

        self.control_wire = 0
        self.system_wires = tuple(range(1, self.n + 1))
        self.dev_h = qml.device(device_name, wires=self.n + 1)
        self.dev_state = qml.device(device_name, wires=self.n)

        self.b_mode = data.b_unitary_info.mode
        self.b_unitary_dag = np.conjugate(data.b_unitary_info.unitary.T)

        self._build_linear_combinations()

        @qml.qnode(self.dev_h, interface=interface, diff_method=diff_method)
        def hadamard_expect_word(weights: np.ndarray, pauli_word: PauliWord):
            qml.Hadamard(wires=self.control_wire)
            _global_ansatz(weights, wires=self.system_wires)
            _apply_controlled_pauli_word(pauli_word, self.control_wire, self.system_wires)
            qml.Hadamard(wires=self.control_wire)
            return qml.expval(qml.PauliZ(self.control_wire))

        @qml.qnode(self.dev_h, interface=interface, diff_method=diff_method)
        def hadamard_gamma_real(weights: np.ndarray, term_word: PauliWord):
            qml.Hadamard(wires=self.control_wire)
            qml.ctrl(_global_ansatz, control=self.control_wire)(weights, self.system_wires)
            _apply_controlled_pauli_word(term_word, self.control_wire, self.system_wires)
            self._apply_b_dagger_controlled()
            qml.Hadamard(wires=self.control_wire)
            return qml.expval(qml.PauliZ(self.control_wire))

        @qml.qnode(self.dev_h, interface=interface, diff_method=diff_method)
        def hadamard_gamma_imag(weights: np.ndarray, term_word: PauliWord):
            qml.Hadamard(wires=self.control_wire)
            qml.ctrl(_global_ansatz, control=self.control_wire)(weights, self.system_wires)
            _apply_controlled_pauli_word(term_word, self.control_wire, self.system_wires)
            self._apply_b_dagger_controlled()
            qml.adjoint(qml.S)(wires=self.control_wire)
            qml.Hadamard(wires=self.control_wire)
            return qml.expval(qml.PauliZ(self.control_wire))

        # State readout is only used for post-update diagnostics (no gradients).
        # Keep diff_method disabled so adjoint mode does not fail on qml.state().
        @qml.qnode(self.dev_state, interface=interface, diff_method=None)
        def state_qnode(weights: np.ndarray):
            _global_ansatz(weights, wires=range(self.n))
            return qml.state()

        self._hadamard_expect_word = hadamard_expect_word
        self._hadamard_gamma_real = hadamard_gamma_real
        self._hadamard_gamma_imag = hadamard_gamma_imag
        self._state_qnode = state_qnode

    def _apply_b_dagger_controlled(self) -> None:
        if self.b_mode == "hadamard_tensor":
            for wire in self.system_wires:
                qml.ctrl(qml.Hadamard, control=self.control_wire)(wires=wire)
            return

        def _b_dag():
            qml.QubitUnitary(self.b_unitary_dag, wires=self.system_wires)

        qml.ctrl(_b_dag, control=self.control_wire)()

    def _build_linear_combinations(self) -> None:
        self.beta_word_weights: Dict[PauliWord, complex] = {}
        self.mu_word_weights_by_wire: List[Dict[PauliWord, complex]] = [
            {} for _ in range(self.n)
        ]

        for c_l, w_l in self.terms:
            for c_lp, w_lp in self.terms:
                pair_coeff = complex(c_l) * complex(c_lp)

                phase_beta, w_beta = _multiply_pauli_words(w_l, w_lp)
                self.beta_word_weights[w_beta] = (
                    self.beta_word_weights.get(w_beta, 0.0 + 0.0j) + pair_coeff * phase_beta
                )

                for wire in range(self.n):
                    x_word = _word_with_single_pauli(self.n, wire, "X")
                    phase_1, tmp = _multiply_pauli_words(w_l, x_word)
                    phase_2, w_full = _multiply_pauli_words(tmp, w_lp)
                    row = self.mu_word_weights_by_wire[wire]
                    row[w_full] = row.get(w_full, 0.0 + 0.0j) + pair_coeff * phase_1 * phase_2

        all_words = set(self.beta_word_weights.keys())
        for row in self.mu_word_weights_by_wire:
            all_words.update(row.keys())
        self.all_words = sorted(all_words)

    def state(self, weights: np.ndarray):
        return self._state_qnode(weights)

    def _expect_word_on_x(self, weights: np.ndarray, word: PauliWord):
        if word == self.identity_word:
            return pnp.array(1.0)
        return self._hadamard_expect_word(weights, word)

    def metrics(self, weights: np.ndarray) -> Dict[str, pnp.tensor]:
        mu_cache: Dict[PauliWord, pnp.tensor] = {}
        for word in self.all_words:
            mu_cache[word] = self._expect_word_on_x(weights, word)

        beta = pnp.array(0.0)
        for word, alpha in self.beta_word_weights.items():
            beta = beta + float(np.real(alpha)) * mu_cache[word]

        plus_sum = pnp.array(0.0)
        for wire in range(self.n):
            mu_j = pnp.array(0.0)
            for word, alpha in self.mu_word_weights_by_wire[wire].items():
                mu_j = mu_j + float(np.real(alpha)) * mu_cache[word]
            plus_sum = plus_sum + 0.5 * (beta + mu_j)

        gamma_re = pnp.array(0.0)
        gamma_im = pnp.array(0.0)
        for coeff, word in self.terms:
            gamma_re = gamma_re + coeff * self._hadamard_gamma_real(weights, word)
            gamma_im = gamma_im + coeff * self._hadamard_gamma_imag(weights, word)

        overlap_sq = gamma_re * gamma_re + gamma_im * gamma_im

        cghat = beta - overlap_sq
        clhat = 0.5 * beta - plus_sum / (2.0 * self.n)

        beta_safe = beta + 1.0e-14
        cg = 1.0 - overlap_sq / beta_safe
        cl = 0.5 - plus_sum / (2.0 * self.n * beta_safe)

        return {
            "beta": beta,
            "plus_sum": plus_sum,
            "gamma_re": gamma_re,
            "gamma_im": gamma_im,
            "overlap_sq": overlap_sq,
            "global_CG": cg,
            "global_CL": cl,
            "global_CG_hat": cghat,
            "global_CL_hat": clhat,
        }

    def objective(self, weights: np.ndarray, metric_name: str):
        vals = self.metrics(weights)
        if metric_name not in vals:
            raise ValueError(f"Unknown optimize metric: {metric_name}")
        return vals[metric_name]


@dataclass
class L2Info:
    abs_error_raw: float
    rel_error_raw: float
    abs_error_aligned: float
    rel_error_aligned: float
    phase_angle_rad: float
    ax_norm: float
    scale_ratio: float
    residual_ax_minus_b_abs_raw: float
    residual_ax_minus_b_rel_raw: float
    residual_ax_minus_b_abs_aligned: float
    residual_ax_minus_b_rel_aligned: float
    residual_ax_minus_b_abs: float
    residual_ax_minus_b_rel: float


def compute_l2_error_unnormalized(
    evaluator: HadamardCentralizedVQLS,
    weights: np.ndarray,
    a_dense: np.ndarray,
    b_unnorm: np.ndarray,
    b_unnorm_norm: float,
    x_true: np.ndarray,
) -> L2Info:
    x_norm = np.array(evaluator.state(weights), dtype=np.complex128)
    ax = a_dense @ x_norm
    ax_norm = float(np.linalg.norm(ax))
    scale = float(b_unnorm_norm / (ax_norm + 1.0e-14))
    x_est_unnorm = scale * x_norm
    residual_ax_minus_b_abs_raw = float(np.linalg.norm((a_dense @ x_est_unnorm) - b_unnorm))
    residual_ax_minus_b_rel_raw = float(residual_ax_minus_b_abs_raw / (b_unnorm_norm + 1.0e-14))

    x_true_norm = float(np.linalg.norm(x_true))
    abs_err_raw = float(np.linalg.norm(x_est_unnorm - x_true))
    rel_err_raw = float(abs_err_raw / (x_true_norm + 1.0e-14))

    overlap = np.vdot(x_est_unnorm, x_true)
    phase_angle = float(np.angle(overlap)) if np.abs(overlap) > 1.0e-16 else 0.0
    x_est_aligned = x_est_unnorm * np.exp(-1.0j * phase_angle)
    residual_ax_minus_b_abs_aligned = float(np.linalg.norm((a_dense @ x_est_aligned) - b_unnorm))
    residual_ax_minus_b_rel_aligned = float(
        residual_ax_minus_b_abs_aligned / (b_unnorm_norm + 1.0e-14)
    )

    # Evaluation-only trick: remove global phase/sign ambiguity before L2.
    abs_err_aligned = float(np.linalg.norm(x_est_aligned - x_true))
    rel_err_aligned = float(abs_err_aligned / (x_true_norm + 1.0e-14))

    return L2Info(
        abs_error_raw=abs_err_raw,
        rel_error_raw=rel_err_raw,
        abs_error_aligned=abs_err_aligned,
        rel_error_aligned=rel_err_aligned,
        phase_angle_rad=phase_angle,
        ax_norm=ax_norm,
        scale_ratio=scale,
        residual_ax_minus_b_abs_raw=residual_ax_minus_b_abs_raw,
        residual_ax_minus_b_rel_raw=residual_ax_minus_b_rel_raw,
        residual_ax_minus_b_abs_aligned=residual_ax_minus_b_abs_aligned,
        residual_ax_minus_b_rel_aligned=residual_ax_minus_b_rel_aligned,
        # Keep these aliases for backward compatibility; use aligned sign convention.
        residual_ax_minus_b_abs=residual_ax_minus_b_abs_aligned,
        residual_ax_minus_b_rel=residual_ax_minus_b_rel_aligned,
    )


def _write_complex_vector_table(path: Path, vec: np.ndarray) -> None:
    arr = np.asarray(vec, dtype=np.complex128).reshape(-1)
    with path.open("w", encoding="utf-8") as f:
        f.write("# idx real imag\n")
        for i, v in enumerate(arr):
            f.write(f"{i} {v.real:.18e} {v.imag:.18e}\n")


def _format_complex_vector_preview(vec: np.ndarray, max_entries: int = 16) -> str:
    arr = np.asarray(vec, dtype=np.complex128).reshape(-1)
    n_show = min(max_entries, arr.size)
    lines = [f"size={arr.size}, showing first {n_show} entries"]
    for i in range(n_show):
        v = arr[i]
        lines.append(f"[{i:4d}] {v.real:+.12e} {v.imag:+.12e}j")
    if n_show < arr.size:
        lines.append("...")
    return "\n".join(lines)


def write_report(
    run_dir: Path,
    cfg: dict,
    data: CentralizedIsingData,
    history: List[dict],
    checkpoints: List[dict],
    best: dict,
    final_metrics: Dict[str, float],
    solution_artifacts: Dict[str, Path] | None = None,
    solution_previews: Dict[str, str] | None = None,
) -> None:
    history_path = run_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    checkpoints_path = run_dir / "checkpoints.json"
    with checkpoints_path.open("w", encoding="utf-8") as f:
        json.dump(checkpoints, f, indent=2)

    report_lines = []
    report_lines.append("# Centralized VQLS Report")
    report_lines.append("")
    report_lines.append(f"- Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    report_lines.append(f"- Static ops path: `{data.static_ops_path}`")
    report_lines.append(f"- Loaded problem source: `{data.problem_source}`")
    report_lines.append(f"- Total qubits: {data.n_total_qubits}")
    report_lines.append(f"- Agents: {data.n_agents} (index qubits={data.n_index_qubits}, local qubits={data.n_data_qubits})")
    report_lines.append("")

    report_lines.append("## Matrix Consistency")
    report_lines.append(f"- Formula vs block max abs diff: {data.matrix_max_abs_diff:.6e}")
    report_lines.append(f"- Formula vs block Frobenius diff: {data.matrix_fro_diff:.6e}")
    report_lines.append(f"- allclose(atol={cfg['problem']['consistency_atol']}, rtol=0): {data.matrix_allclose}")
    report_lines.append(f"- Condition number of A (2-norm): {data.condition_number:.12e}")
    report_lines.append("")

    report_lines.append("## Equation Match vs Distributed Reference")
    report_lines.append(f"- Reference system key: `{data.reference_system_key}`")
    report_lines.append(f"- A_target vs A_ref max abs diff: {data.reference_matrix_max_abs_diff:.6e}")
    report_lines.append(f"- A_target vs A_ref Frobenius diff: {data.reference_matrix_fro_diff:.6e}")
    report_lines.append(
        f"- A_target vs A_ref allclose(atol={cfg['problem']['consistency_atol']}, rtol=0): "
        f"{data.reference_matrix_allclose}"
    )
    report_lines.append(f"- b_target vs b_ref max abs diff: {data.reference_b_max_abs_diff:.6e}")
    report_lines.append(f"- b_target vs b_ref L2 diff: {data.reference_b_l2_diff:.6e}")
    report_lines.append(
        f"- b_target vs b_ref allclose(atol={cfg['problem'].get('b_consistency_atol', cfg['problem']['consistency_atol'])}, "
        f"rtol=0): {data.reference_b_allclose}"
    )
    report_lines.append("")

    report_lines.append("## b-Vector and Unitary")
    report_lines.append(f"- ||b_unnorm||: {data.b_unnorm_norm:.12f}")
    report_lines.append(f"- b unitary mode: {data.b_unitary_info.mode}")
    report_lines.append(f"- ||b_normed - |+>^n||: {data.b_unitary_info.hadamard_match_error:.6e}")
    report_lines.append(f"- ||U_b|0> - b_normed|| (phase-aligned): {data.b_unitary_info.state_prep_error:.6e}")
    report_lines.append("")

    report_lines.append("## Optimization Summary")
    report_lines.append(f"- Optimize metric: {cfg['optimization']['optimize_metric']}")
    report_lines.append(f"- Steps: {cfg['optimization']['steps']}")
    report_lines.append(f"- Learning rate: {cfg['optimization']['learning_rate']}")
    report_lines.append(f"- Adam betas: ({cfg['optimization']['beta1']}, {cfg['optimization']['beta2']})")
    report_lines.append(f"- Adam eps: {cfg['optimization']['eps']}")
    report_lines.append(f"- Best iteration: {best['iteration']}")
    report_lines.append(f"- Best {cfg['optimization']['optimize_metric']}: {best['loss']:.12e}")
    report_lines.append("")

    report_lines.append("## Final Metrics")
    for key in ["global_CG", "global_CL", "global_CG_hat", "global_CL_hat", "beta", "overlap_sq"]:
        report_lines.append(f"- {key}: {final_metrics[key]:.12e}")
    report_lines.append("- Residual on linear system (using reconstructed global x):")
    report_lines.append(f"  - ||A x_est - b||_2 (raw x_est): {final_metrics['residual_ax_minus_b_abs_raw']:.12e}")
    report_lines.append(
        f"  - ||A x_est - b||_2 / ||b||_2 (raw x_est): {final_metrics['residual_ax_minus_b_rel_raw']:.12e}"
    )
    report_lines.append(
        f"  - ||A x_est - b||_2 (phase-aligned x_est): {final_metrics['residual_ax_minus_b_abs_aligned']:.12e}"
    )
    report_lines.append(
        "  - ||A x_est - b||_2 / ||b||_2 (phase-aligned x_est): "
        f"{final_metrics['residual_ax_minus_b_rel_aligned']:.12e}"
    )
    report_lines.append("- Residual to exact solution (using reconstructed global x):")
    report_lines.append(f"  - ||x_est - x_true||_2 (raw): {final_metrics['l2_abs_raw']:.12e}")
    report_lines.append(f"  - ||x_est - x_true||_2 / ||x_true||_2 (raw): {final_metrics['l2_rel_raw']:.12e}")
    report_lines.append(f"  - ||x_est - x_true||_2 (phase-aligned): {final_metrics['l2_abs_aligned']:.12e}")
    report_lines.append(f"  - ||x_est - x_true||_2 / ||x_true||_2 (phase-aligned): {final_metrics['l2_rel_aligned']:.12e}")
    report_lines.append(f"- phase alignment angle (rad): {final_metrics['phase_angle_rad']:.12e}")
    report_lines.append("")

    report_lines.append("## Timing")
    if history:
        report_lines.append(f"- First output time: {history[0]['time_iso']}")
        report_lines.append(f"- Last output time: {history[-1]['time_iso']}")
        report_lines.append(f"- Total elapsed seconds: {history[-1]['elapsed_s']:.6f}")
    else:
        report_lines.append("- No optimization history captured.")
    report_lines.append("")

    report_lines.append("## Files")
    report_lines.append(f"- History: `{history_path}`")
    report_lines.append(f"- Checkpoints: `{checkpoints_path}`")
    report_lines.append(f"- Config used: `{run_dir / 'config_used.yaml'}`")
    if solution_artifacts:
        for key in sorted(solution_artifacts.keys()):
            report_lines.append(f"- {key}: `{solution_artifacts[key]}`")
    report_lines.append("")

    if solution_previews:
        report_lines.append("## Solution Preview")
        for key in ["x_true", "x_est_unnorm_raw", "x_est_unnorm_aligned", "final_theta"]:
            if key not in solution_previews:
                continue
            report_lines.append(f"### {key}")
            report_lines.append("```text")
            report_lines.extend(solution_previews[key].splitlines())
            report_lines.append("```")
    report_lines.append("")

    report_lines.append("## Hyperparameters")
    report_lines.append("```yaml")
    cfg_yaml = yaml.safe_dump(cfg, sort_keys=False).strip("\n")
    report_lines.extend(cfg_yaml.splitlines())
    report_lines.append("```")

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Centralized VQLS for Ising block-partitioned system")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().with_name("config.yaml")),
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(Path(args.config).resolve())

    data = load_centralized_data(cfg, repo_root=repo_root)

    print("[problem] source:", data.problem_source)
    print("[consistency] formula/block max abs diff:", f"{data.matrix_max_abs_diff:.6e}")
    print("[consistency] formula/block fro diff:", f"{data.matrix_fro_diff:.6e}")
    print("[consistency] allclose:", data.matrix_allclose)
    print("[matrix] cond(A):", f"{data.condition_number:.12e}")
    if not data.matrix_allclose:
        raise RuntimeError(
            "Matrix from loaded terms is not consistent with the loaded centralized target matrix. "
            "Please check static-ops centralized terms/matrix definition and wire ordering."
        )
    print("[reference] system key:", data.reference_system_key)
    print("[reference] A_target vs A_ref max abs diff:", f"{data.reference_matrix_max_abs_diff:.6e}")
    print("[reference] A_target vs A_ref fro diff:", f"{data.reference_matrix_fro_diff:.6e}")
    print("[reference] A_target vs A_ref allclose:", data.reference_matrix_allclose)
    print("[reference] b_target vs b_ref max abs diff:", f"{data.reference_b_max_abs_diff:.6e}")
    print("[reference] b_target vs b_ref l2 diff:", f"{data.reference_b_l2_diff:.6e}")
    print("[reference] b_target vs b_ref allclose:", data.reference_b_allclose)
    print("[b] ||b_unnorm||:", f"{data.b_unnorm_norm:.12f}")
    print("[b] b unitary mode:", data.b_unitary_info.mode)
    print("[b] ||b_normed - |+>^n||:", f"{data.b_unitary_info.hadamard_match_error:.6e}")

    evaluator = HadamardCentralizedVQLS(
        data=data,
        device_name=str(cfg["runtime"]["device"]),
        interface=str(cfg["runtime"]["interface"]),
        diff_method=str(cfg["runtime"]["diff_method"]),
    )

    x_true = np.linalg.solve(data.a_block, data.b_unnorm)

    rng = np.random.default_rng(int(cfg["runtime"]["seed"]))
    layers = int(cfg["ansatz"]["layers"])
    theta = pnp.array(
        rng.uniform(
            float(cfg["ansatz"]["init_low"]),
            float(cfg["ansatz"]["init_high"]),
            size=(layers, data.n_total_qubits),
        ),
        requires_grad=True,
    )

    optimize_metric = str(cfg["optimization"]["optimize_metric"])

    def cost_fn(w):
        return evaluator.objective(w, optimize_metric)

    opt = qml.AdamOptimizer(
        stepsize=float(cfg["optimization"]["learning_rate"]),
        beta1=float(cfg["optimization"]["beta1"]),
        beta2=float(cfg["optimization"]["beta2"]),
        eps=float(cfg["optimization"]["eps"]),
    )

    steps = int(cfg["optimization"]["steps"])
    print_every = int(cfg["optimization"]["print_every"])

    history: List[dict] = []
    checkpoints: List[dict] = []
    t0 = time.time()

    best = {"iteration": 0, "loss": float("inf")}

    for it in range(1, steps + 1):
        theta, loss_val = opt.step_and_cost(cost_fn, theta)
        loss_float = _to_float(loss_val)
        elapsed_s = float(time.time() - t0)
        now_iso = datetime.now().isoformat(timespec="milliseconds")

        l2 = compute_l2_error_unnormalized(
            evaluator=evaluator,
            weights=theta,
            a_dense=data.a_block,
            b_unnorm=data.b_unnorm,
            b_unnorm_norm=data.b_unnorm_norm,
            x_true=x_true,
        )

        row = {
            "iteration": it,
            "time_iso": now_iso,
            "elapsed_s": elapsed_s,
            optimize_metric: loss_float,
            "l2_abs_raw": l2.abs_error_raw,
            "l2_rel_raw": l2.rel_error_raw,
            "l2_abs_aligned": l2.abs_error_aligned,
            "l2_rel_aligned": l2.rel_error_aligned,
            "phase_angle_rad": l2.phase_angle_rad,
            "ax_norm": l2.ax_norm,
            "scale_ratio": l2.scale_ratio,
            "residual_ax_minus_b_abs_raw": l2.residual_ax_minus_b_abs_raw,
            "residual_ax_minus_b_rel_raw": l2.residual_ax_minus_b_rel_raw,
            "residual_ax_minus_b_abs_aligned": l2.residual_ax_minus_b_abs_aligned,
            "residual_ax_minus_b_rel_aligned": l2.residual_ax_minus_b_rel_aligned,
            "residual_ax_minus_b_abs": l2.residual_ax_minus_b_abs,
            "residual_ax_minus_b_rel": l2.residual_ax_minus_b_rel,
        }
        history.append(row)

        if loss_float < best["loss"]:
            best = {"iteration": it, "loss": loss_float}

        if (it % print_every == 0) or (it == 1) or (it == steps):
            m = evaluator.metrics(theta)
            check = {
                "iteration": it,
                "time_iso": now_iso,
                "elapsed_s": elapsed_s,
                optimize_metric: loss_float,
                "global_CG": _to_float(m["global_CG"]),
                "global_CL": _to_float(m["global_CL"]),
                "global_CG_hat": _to_float(m["global_CG_hat"]),
                "global_CL_hat": _to_float(m["global_CL_hat"]),
                "beta": _to_float(m["beta"]),
                "overlap_sq": _to_float(m["overlap_sq"]),
                "l2_abs_raw": l2.abs_error_raw,
                "l2_rel_raw": l2.rel_error_raw,
                "l2_abs_aligned": l2.abs_error_aligned,
                "l2_rel_aligned": l2.rel_error_aligned,
                "phase_angle_rad": l2.phase_angle_rad,
                "ax_norm": l2.ax_norm,
                "scale_ratio": l2.scale_ratio,
                "residual_ax_minus_b_abs_raw": l2.residual_ax_minus_b_abs_raw,
                "residual_ax_minus_b_rel_raw": l2.residual_ax_minus_b_rel_raw,
                "residual_ax_minus_b_abs_aligned": l2.residual_ax_minus_b_abs_aligned,
                "residual_ax_minus_b_rel_aligned": l2.residual_ax_minus_b_rel_aligned,
                "residual_ax_minus_b_abs": l2.residual_ax_minus_b_abs,
                "residual_ax_minus_b_rel": l2.residual_ax_minus_b_rel,
            }
            checkpoints.append(check)
            print(
                f"[{now_iso}] [iter {it:4d}] {optimize_metric}={loss_float:.8e} "
                f"CG={check['global_CG']:.8e} CL={check['global_CL']:.8e} "
                f"||Ax_est-b||(aligned)={l2.residual_ax_minus_b_abs_aligned:.8e} "
                f"L2_rel(aligned)={l2.rel_error_aligned:.8e} "
                f"L2_rel(raw)={l2.rel_error_raw:.8e}"
            )

    final_raw = evaluator.metrics(theta)
    final_metrics = {
        "global_CG": _to_float(final_raw["global_CG"]),
        "global_CL": _to_float(final_raw["global_CL"]),
        "global_CG_hat": _to_float(final_raw["global_CG_hat"]),
        "global_CL_hat": _to_float(final_raw["global_CL_hat"]),
        "beta": _to_float(final_raw["beta"]),
        "overlap_sq": _to_float(final_raw["overlap_sq"]),
    }
    final_l2 = compute_l2_error_unnormalized(
        evaluator=evaluator,
        weights=theta,
        a_dense=data.a_block,
        b_unnorm=data.b_unnorm,
        b_unnorm_norm=data.b_unnorm_norm,
        x_true=x_true,
    )
    final_metrics["l2_abs_raw"] = final_l2.abs_error_raw
    final_metrics["l2_rel_raw"] = final_l2.rel_error_raw
    final_metrics["l2_abs_aligned"] = final_l2.abs_error_aligned
    final_metrics["l2_rel_aligned"] = final_l2.rel_error_aligned
    final_metrics["phase_angle_rad"] = final_l2.phase_angle_rad
    final_metrics["residual_ax_minus_b_abs_raw"] = final_l2.residual_ax_minus_b_abs_raw
    final_metrics["residual_ax_minus_b_rel_raw"] = final_l2.residual_ax_minus_b_rel_raw
    final_metrics["residual_ax_minus_b_abs_aligned"] = final_l2.residual_ax_minus_b_abs_aligned
    final_metrics["residual_ax_minus_b_rel_aligned"] = final_l2.residual_ax_minus_b_rel_aligned
    final_metrics["residual_ax_minus_b_abs"] = final_l2.residual_ax_minus_b_abs
    final_metrics["residual_ax_minus_b_rel"] = final_l2.residual_ax_minus_b_rel

    report_cfg = cfg["report"]
    out_dir = Path(report_cfg["out_dir"])
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"{report_cfg['tag']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    theta_final = np.array(theta, dtype=np.float64)
    x_norm_final = np.array(evaluator.state(theta), dtype=np.complex128)
    ax_final = data.a_block @ x_norm_final
    scale_final = float(data.b_unnorm_norm / (np.linalg.norm(ax_final) + 1.0e-14))
    x_est_unnorm_raw = scale_final * x_norm_final
    x_est_unnorm_aligned = x_est_unnorm_raw * np.exp(-1.0j * final_l2.phase_angle_rad)

    solution_txt_path = run_dir / "solution_comparison.txt"
    with solution_txt_path.open("w", encoding="utf-8") as f:
        f.write("# Final variational parameter theta\n")
        f.write(np.array2string(theta_final, precision=16))
        f.write("\n\n")
        for name, vec in [
            ("x_true", x_true),
            ("x_est_unnorm_raw", x_est_unnorm_raw),
            ("x_est_unnorm_aligned", x_est_unnorm_aligned),
        ]:
            arr = np.asarray(vec, dtype=np.complex128).reshape(-1)
            f.write(f"# {name}: idx real imag\n")
            for i, v in enumerate(arr):
                f.write(f"{i} {v.real:.18e} {v.imag:.18e}\n")
            f.write("\n")

    solution_artifacts = {
        "solution_txt": solution_txt_path,
    }
    solution_previews = {
        "x_true": _format_complex_vector_preview(x_true),
        "x_est_unnorm_raw": _format_complex_vector_preview(x_est_unnorm_raw),
        "x_est_unnorm_aligned": _format_complex_vector_preview(x_est_unnorm_aligned),
        "final_theta": np.array2string(theta_final, precision=10),
    }

    write_report(
        run_dir=run_dir,
        cfg=cfg,
        data=data,
        history=history,
        checkpoints=checkpoints,
        best=best,
        final_metrics=final_metrics,
        solution_artifacts=solution_artifacts,
        solution_previews=solution_previews,
    )

    print("[done] report directory:", run_dir)


if __name__ == "__main__":
    main()
