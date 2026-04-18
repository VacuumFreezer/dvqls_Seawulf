from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOCAL_ROOT = Path(__file__).resolve().parent
if str(LOCAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ROOT))

from benchmark_13q_full_cluster_stabilizer_common import load_module_from_path


DEFAULT_STATIC_OPS = (
    "Partition_comparison_qjit/13stabilizer/1d1/static_ops_13q_full_cluster_stabilizer_1x1.py",
    "Partition_comparison_qjit/13stabilizer/2d2/static_ops_13q_full_cluster_stabilizer_2x2.py",
    "Partition_comparison_qjit/13stabilizer/4d4/static_ops_13q_full_cluster_stabilizer_4x4.py",
    "Partition_comparison_qjit/13stabilizer/8d8/static_ops_13q_full_cluster_stabilizer_8x8.py",
)


def compare_entries(reference_entries, candidate_entries, *, dim: int):
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
        "max_abs_diff": float(max_abs),
        "data_fro_norm": float(np.sqrt(data_sq_norm)),
    }


def apply_entries(entries, vector: np.ndarray, dim: int) -> np.ndarray:
    out = np.zeros((int(dim),), dtype=np.complex128)
    for (row, col), value in entries.items():
        out[int(row)] += value * vector[int(col)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Partition_comparison_qjit/13stabilizer/verification_report.json")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--static_ops", nargs="*", default=list(DEFAULT_STATIC_OPS))
    args = ap.parse_args(argv)

    records = []
    for static_ops in args.static_ops:
        module = load_module_from_path((ROOT / static_ops).resolve())
        system = module.SYSTEM
        entries = system.reconstruct_global_entries()
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
                "projection_block_check_max_abs_diff": float(system.metadata["b_projection_verification_max_abs_diff"]),
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
            "candidate_projection_block_check_max_abs_diff": float(record["projection_block_check_max_abs_diff"]),
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
                "projection_block_check_max_abs_diff": record["projection_block_check_max_abs_diff"],
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
