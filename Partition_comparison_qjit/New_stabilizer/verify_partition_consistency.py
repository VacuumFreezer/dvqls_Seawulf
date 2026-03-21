from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Partition_comparison_qjit.New_stabilizer.benchmark_13q_xzx_fresh_common import load_module_from_path

DEFAULT_STATIC_OPS = (
    "Partition_comparison_qjit/New_stabilizer/1d1/static_ops_13q_xzx_fresh_1x1.py",
    "Partition_comparison_qjit/New_stabilizer/2d2/static_ops_13q_xzx_fresh_2x2.py",
    "Partition_comparison_qjit/New_stabilizer/4d4/static_ops_13q_xzx_fresh_4x4.py",
    "Partition_comparison_qjit/New_stabilizer/8d8/static_ops_13q_xzx_fresh_8x8.py",
)


def _local_masks(local_pauli_map: dict[int, str], n_qubits: int) -> tuple[int, int]:
    x_mask = 0
    z_mask = 0
    for wire in range(int(n_qubits)):
        bit = 1 << (int(n_qubits) - wire - 1)
        label = local_pauli_map.get(wire, "I")
        if label == "X":
            x_mask |= bit
        elif label == "Z":
            z_mask |= bit
    return x_mask, z_mask


def _popcount(value: int) -> int:
    return bin(int(value)).count("1")


def reconstruct_global_entries(system) -> dict[tuple[int, int], complex]:
    n_agents = int(system.n)
    n_local = int(system.n_data_qubits)
    local_dim = 1 << n_local
    entries: dict[tuple[int, int], complex] = {}

    for row_id in range(n_agents):
        for col_id in range(n_agents):
            for coeff, local_paulis in zip(system.coeffs[row_id][col_id], system.local_pauli_maps_grid[row_id][col_id]):
                x_mask, z_mask = _local_masks(local_paulis, n_local)
                for local_col in range(local_dim):
                    local_row = local_col ^ x_mask
                    phase = -1.0 if _popcount(local_col & z_mask) % 2 else 1.0
                    global_row = row_id * local_dim + local_row
                    global_col = col_id * local_dim + local_col
                    key = (global_row, global_col)
                    entries[key] = entries.get(key, 0.0 + 0.0j) + (complex(coeff) * phase)

    return {key: value for key, value in entries.items() if abs(value) > 1e-14}


def compare_entries(
    reference_entries: dict[tuple[int, int], complex],
    candidate_entries: dict[tuple[int, int], complex],
    *,
    dim: int,
) -> dict[str, float | int]:
    all_keys = set(reference_entries) | set(candidate_entries)
    max_abs = 0.0
    data_sq_norm = 0.0
    nnz_diff = 0
    for key in all_keys:
        diff = candidate_entries.get(key, 0.0 + 0.0j) - reference_entries.get(key, 0.0 + 0.0j)
        abs_diff = abs(diff)
        if abs_diff > 1e-14:
            nnz_diff += 1
            data_sq_norm += abs_diff * abs_diff
            if abs_diff > max_abs:
                max_abs = float(abs_diff)
    return {
        "shape_0": int(dim),
        "shape_1": int(dim),
        "nnz_diff": int(nnz_diff),
        "max_abs_diff": max_abs,
        "data_fro_norm": float(np.sqrt(data_sq_norm)),
    }


def apply_entries(entries: dict[tuple[int, int], complex], vector: np.ndarray, dim: int) -> np.ndarray:
    out = np.zeros((int(dim),), dtype=np.complex128)
    for (row, col), value in entries.items():
        out[int(row)] += value * vector[int(col)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Partition_comparison_qjit/New_stabilizer/verification_report.json")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--static_ops", nargs="*", default=list(DEFAULT_STATIC_OPS))
    args = ap.parse_args(argv)

    records = []
    for static_ops in args.static_ops:
        module = load_module_from_path((ROOT / static_ops).resolve())
        system = module.SYSTEM
        entries = reconstruct_global_entries(system)
        dim = int(system.n) * (1 << int(system.n_data_qubits))
        global_b = np.asarray(system.get_global_b_vector())
        true_solution = np.asarray(system.true_solution_vector())
        residual = apply_entries(entries, true_solution, dim) - global_b
        records.append(
            {
                "name": getattr(system, "name", static_ops),
                "static_ops": static_ops,
                "n_agents": int(system.n),
                "n_local_qubits": int(system.n_data_qubits),
                "shape": [dim, dim],
                "nnz": int(len(entries)),
                "A_entries": entries,
                "b": global_b,
                "x_star": true_solution,
                "x_star_residual_norm": float(np.linalg.norm(residual)),
            }
        )

    reference = records[0]
    comparisons = []
    consistent = True
    for record in records[1:]:
        a_cmp = compare_entries(reference["A_entries"], record["A_entries"], dim=reference["shape"][0])
        b_diff = np.asarray(record["b"]) - np.asarray(reference["b"])
        x_diff = np.asarray(record["x_star"]) - np.asarray(reference["x_star"])
        item = {
            "reference": reference["name"],
            "candidate": record["name"],
            "A": a_cmp,
            "b_max_abs_diff": float(np.max(np.abs(b_diff))),
            "x_star_max_abs_diff": float(np.max(np.abs(x_diff))),
            "candidate_x_star_residual_norm": float(record["x_star_residual_norm"]),
        }
        item["same_problem"] = (
            a_cmp["max_abs_diff"] <= args.tol
            and item["b_max_abs_diff"] <= args.tol
            and item["x_star_max_abs_diff"] <= args.tol
        )
        consistent = consistent and bool(item["same_problem"])
        comparisons.append(item)

    payload = {
        "tolerance": float(args.tol),
        "consistent": bool(consistent),
        "reference": reference["name"],
        "systems": [
            {
                "name": record["name"],
                "static_ops": record["static_ops"],
                "n_agents": record["n_agents"],
                "n_local_qubits": record["n_local_qubits"],
                "shape": record["shape"],
                "nnz": record["nnz"],
                "x_star_residual_norm": record["x_star_residual_norm"],
            }
            for record in records
        ],
        "comparisons": comparisons,
    }

    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2))
    if not consistent:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
