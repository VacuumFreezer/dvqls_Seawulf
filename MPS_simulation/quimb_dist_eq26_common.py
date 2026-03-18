#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_PARAM_PATH = THIS_DIR / "param.yaml"


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML config must be a mapping.")
    return data


def dump_yaml_config(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def merge_section_config(
    defaults: dict[str, Any],
    config_path: str | Path | None,
    section: str,
    case: str | None = None,
) -> dict[str, Any]:
    merged = copy.deepcopy(defaults)
    loaded = load_yaml_config(config_path)

    shared = loaded.get("shared", {})
    if isinstance(shared, dict):
        deep_update(merged, shared)

    section_data = loaded.get(section, {})
    if isinstance(section_data, dict):
        deep_update(merged, section_data)

    if case is not None:
        cases = loaded.get("cases", {})
        if case not in cases:
            raise KeyError(f"Unknown case `{case}` in config `{config_path}`.")
        case_data = cases[case]
        if not isinstance(case_data, dict):
            raise ValueError(f"Case `{case}` must map to a dictionary.")
        plain_case_updates = {k: v for k, v in case_data.items() if k != section}
        deep_update(merged, plain_case_updates)
        nested_section = case_data.get(section, {})
        if isinstance(nested_section, dict):
            deep_update(merged, nested_section)

    return merged


def resolve_qubit_layout(
    cfg_dict: dict[str, Any],
    *,
    qubits_per_agent: int | None = None,
    global_qubits: int | None = None,
    local_qubits: int | None = None,
) -> dict[str, Any]:
    resolved = copy.deepcopy(cfg_dict)

    if qubits_per_agent is not None:
        resolved["local_qubits"] = int(qubits_per_agent)
        resolved["global_qubits"] = int(qubits_per_agent) + 1

    if local_qubits is not None:
        resolved["local_qubits"] = int(local_qubits)

    if global_qubits is not None:
        resolved["global_qubits"] = int(global_qubits)

    if "local_qubits" not in resolved and "qubits_per_agent" in resolved:
        resolved["local_qubits"] = int(resolved["qubits_per_agent"])
    if "global_qubits" not in resolved and "local_qubits" in resolved:
        resolved["global_qubits"] = int(resolved["local_qubits"]) + 1

    if int(resolved["global_qubits"]) != int(resolved["local_qubits"]) + 1:
        raise ValueError(
            "For the current 2x2 Eq. (26) MPS workflow, global_qubits must equal local_qubits + 1."
        )

    return resolved


def parse_int_sequence(raw_value: str | None, *, default: list[int] | None = None) -> list[int]:
    if raw_value is None:
        return [] if default is None else list(default)
    if isinstance(raw_value, (list, tuple)):
        return [int(item) for item in raw_value]
    text = str(raw_value).strip()
    if not text:
        return [] if default is None else list(default)
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def time_callable(fn, *, repeats: int, warmup: int = 0) -> dict[str, Any]:
    last_value = None
    for _ in range(max(int(warmup), 0)):
        last_value = fn()

    durations = []
    for _ in range(max(int(repeats), 1)):
        start = time.perf_counter()
        last_value = fn()
        durations.append(time.perf_counter() - start)

    values = np.asarray(durations, dtype=np.float64)
    return {
        "last_value": last_value,
        "timing": {
            "repeats": int(max(int(repeats), 1)),
            "warmup": int(max(int(warmup), 0)),
            "mean_s": float(np.mean(values)),
            "std_s": float(np.std(values)),
            "min_s": float(np.min(values)),
            "max_s": float(np.max(values)),
        },
    }


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def write(self, payload: dict[str, Any]) -> None:
        self._handle.write(json.dumps(sanitize_jsonable(payload), indent=None) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(sanitize_jsonable(payload), indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(target)


def sanitize_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        float_value = float(value)
        if math.isnan(float_value) or math.isinf(float_value):
            return None
        return float_value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def encode_array(array: np.ndarray) -> dict[str, Any]:
    array_np = np.asarray(array)
    if np.iscomplexobj(array_np):
        return {
            "real": np.asarray(array_np.real, dtype=np.float64).tolist(),
            "imag": np.asarray(array_np.imag, dtype=np.float64).tolist(),
        }
    return {"real": np.asarray(array_np, dtype=np.float64).tolist()}


def decode_array(payload: Any) -> np.ndarray:
    if isinstance(payload, dict) and "real" in payload:
        real = np.asarray(payload["real"])
        imag = np.asarray(payload.get("imag", 0.0))
        if np.any(imag):
            return real + 1.0j * imag
        return real
    return np.asarray(payload)


def format_array_preview(array: np.ndarray, *, max_elements: int = 200) -> str:
    array_np = np.real_if_close(np.asarray(array), tol=1000)
    formatter = None
    if np.iscomplexobj(array_np):
        formatter = {
            "complex_kind": lambda z: f"{complex(z).real:.12g}{complex(z).imag:+.12g}j"
        }
    elif np.issubdtype(array_np.dtype, np.floating):
        formatter = {"float_kind": lambda x: f"{float(x):.12g}"}

    if array_np.size <= max_elements:
        return np.array2string(
            array_np,
            separator=", ",
            max_line_width=160,
            precision=12,
            suppress_small=False,
            threshold=array_np.size,
            formatter=formatter,
        )

    preview = np.asarray(array_np).reshape(-1)[:max_elements]
    preview_text = np.array2string(
        preview,
        separator=", ",
        max_line_width=160,
        precision=12,
        suppress_small=False,
        threshold=preview.size,
        formatter=formatter,
    )
    return (
        f"shape={array_np.shape}, showing first {max_elements} flattened elements:\n"
        f"{preview_text}"
    )


def reshape_brickwall_angles(flat_angles, layers: int, n_qubits: int):
    expected = 2 * int(layers) * int(n_qubits)
    if tuple(np.shape(flat_angles)) != (expected,):
        raise ValueError(
            f"Expected flattened angle vector of shape ({expected},), got {np.shape(flat_angles)}."
        )
    return np.reshape(flat_angles, (int(layers), 2, int(n_qubits)))


def _apply_brickwall_ry_cz(circuit, angles, n_qubits: int) -> None:
    for even_angles, odd_angles in angles:
        for wire in range(n_qubits):
            circuit.apply_gate("RY", even_angles[wire], wire)
        for left in range(0, n_qubits - 1, 2):
            circuit.apply_gate("CZ", left, left + 1)

        for wire in range(n_qubits):
            circuit.apply_gate("RY", odd_angles[wire], wire)
        for left in range(1, n_qubits - 1, 2):
            circuit.apply_gate("CZ", left, left + 1)


def build_circuit_numpy(n_qubits: int, flat_angles, cfg) -> Any:
    import quimb.tensor as qtn

    angles = reshape_brickwall_angles(np.asarray(flat_angles, dtype=np.float64), cfg.layers, n_qubits)
    circuit = qtn.CircuitMPS(
        n_qubits,
        max_bond=cfg.gate_max_bond,
        cutoff=cfg.gate_cutoff,
        convert_eager=True,
    )
    _apply_brickwall_ry_cz(circuit, angles, n_qubits)
    return circuit


def build_circuit_jax(n_qubits: int, flat_angles, cfg) -> Any:
    import jax.numpy as jnp
    import quimb.tensor as qtn

    angles = reshape_brickwall_angles(flat_angles, cfg.layers, n_qubits)
    circuit = qtn.CircuitMPS(
        n_qubits,
        max_bond=cfg.gate_max_bond,
        cutoff=cfg.gate_cutoff,
        to_backend=jnp.asarray,
        convert_eager=True,
    )
    _apply_brickwall_ry_cz(circuit, angles, n_qubits)
    return circuit


def make_plus_state_mps(n_qubits: int):
    import quimb.tensor as qtn

    plus = np.asarray([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
    return qtn.MPS_product_state([plus] * int(n_qubits))


def scale_mps(state, scalar: complex | float):
    return state.copy().multiply(scalar, inplace=True, spread_over=1)


def add_mps(lhs, rhs, *, max_bond: int, cutoff: float):
    summed = lhs.add_MPS(rhs, inplace=False, compress=True, max_bond=max_bond, cutoff=cutoff)
    summed.compress(max_bond=max_bond, cutoff=cutoff)
    return summed


def apply_mpo_to_mps(mpo, state, *, max_bond: int, cutoff: float):
    applied = mpo.apply(
        state,
        contract=True,
        compress=True,
        max_bond=max_bond,
        cutoff=cutoff,
    )
    applied.compress(max_bond=max_bond, cutoff=cutoff)
    return applied


def dmrg_extremal_eigenvalue(mpo, *, which: str, bond_dims: list[int], cutoffs: float, tol: float, max_sweeps: int) -> float:
    import quimb.tensor as qtn

    dmrg = qtn.DMRG2(
        mpo,
        which=which,
        bond_dims=list(bond_dims),
        cutoffs=cutoffs,
    )
    dmrg.solve(
        tol=tol,
        bond_dims=list(bond_dims),
        cutoffs=cutoffs,
        max_sweeps=max_sweeps,
        verbosity=0,
        suppress_warnings=True,
    )
    return float(dmrg.energy)
