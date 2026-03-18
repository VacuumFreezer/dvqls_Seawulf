#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from MPS_simulation.cen.quimb_vqls_eq26_benchmark import build_circuit_numpy  # noqa: E402


def bytes_to_human(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.3f} {unit}"
        value /= 1024.0
    return f"{value:.3f} EiB"


def get_total_ram_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None
    if page_size <= 0 or phys_pages <= 0:
        return None
    return int(page_size) * int(phys_pages)


def estimate_storage(global_qubits: int) -> dict[str, object]:
    local_qubits = global_qubits - 1
    global_dim = 2**global_qubits
    local_dim = 2**local_qubits

    complex128_bytes = 16
    float64_bytes = 8
    int32_bytes = 4
    int64_bytes = 8

    dense_global_matrix = global_dim * global_dim * complex128_bytes
    dense_local_block = local_dim * local_dim * complex128_bytes
    dense_global_vector = global_dim * complex128_bytes
    dense_local_vector = local_dim * complex128_bytes

    nnz_lower_bound = (global_qubits + 1) * global_dim
    csr_float64_lower_bound = (
        nnz_lower_bound * (float64_bytes + int32_bytes)
        + (global_dim + 1) * int64_bytes
    )

    return {
        "global_qubits": global_qubits,
        "local_qubits": local_qubits,
        "global_dim": global_dim,
        "local_dim": local_dim,
        "dense_global_matrix_bytes": dense_global_matrix,
        "dense_local_block_bytes": dense_local_block,
        "dense_global_vector_bytes": dense_global_vector,
        "dense_local_vector_bytes": dense_local_vector,
        "sparse_nnz_lower_bound": nnz_lower_bound,
        "csr_float64_lower_bound_bytes": csr_float64_lower_bound,
    }


def run_circuit_smoke_test(local_qubits: int) -> dict[str, object]:
    cfg = SimpleNamespace(
        layers=4,
        gate_cutoff=1.0e-10,
        gate_max_bond=32,
    )
    angles = np.linspace(
        0.01,
        0.2,
        2 * cfg.layers * local_qubits,
        dtype=np.float64,
    )
    t0 = time.perf_counter()
    circ = build_circuit_numpy(local_qubits, angles, cfg)
    elapsed = time.perf_counter() - t0
    psi = circ.psi
    bond_sizes = tuple(int(b) for b in psi.bond_sizes())
    return {
        "local_qubits": local_qubits,
        "build_time_s": elapsed,
        "state_norm": float(np.real(psi.overlap(psi))),
        "max_bond": max(bond_sizes) if bond_sizes else 1,
        "bond_sizes": bond_sizes,
        "num_tensors": int(psi.num_tensors),
    }


def make_report(result: dict[str, object]) -> str:
    est = result["storage_estimates"]
    smoke = result["circuit_smoke_test"]
    ram_bytes = result["system"]["total_ram_bytes"]
    lines = [
        "# 30-Qubit Feasibility Check",
        "",
        "## Conclusion",
        "The current distributed workflow is not feasible at 30 global qubits as written.",
        "The MPS circuit ansatz itself is fine at 29 local qubits, but the current operator and RHS construction path is not scalable.",
        "",
        "## Quick Check",
        f"- Pure 29-qubit MPS circuit build time: `{smoke['build_time_s']:.6f} s`.",
        f"- Circuit state norm: `{smoke['state_norm']:.12g}`.",
        f"- Max MPS bond during the smoke test: `{smoke['max_bond']}`.",
        f"- Number of MPS tensors: `{smoke['num_tensors']}`.",
        "",
        "## Current Workflow Bottlenecks",
        "- The distributed code builds a full sparse global matrix with `build_sparse(global_qubits)`.",
        "  File: `MPS_simulation/dist/quimb_dist_eq26_2x2_benchmark.py:168` and `MPS_simulation/dist/quimb_dist_eq26_2x2_optimize.py:165`.",
        "- It then converts the full global matrix to dense with `a_sparse.toarray()`.",
        "  File: `MPS_simulation/dist/quimb_dist_eq26_2x2_benchmark.py:180` and `MPS_simulation/dist/quimb_dist_eq26_2x2_optimize.py:591`.",
        "- Local MPOs are constructed from dense blocks using `MatrixProductOperator.from_dense(...)`.",
        "  File: `MPS_simulation/dist/quimb_dist_eq26_2x2_benchmark.py:148`.",
        "- The RHS `b` is also materialized as a dense global vector and then split into dense local vectors.",
        "  File: `MPS_simulation/dist/quimb_dist_eq26_2x2_benchmark.py:191-216`.",
        "",
        "## Size Estimates For 30 Global Qubits",
        f"- Global dimension: `{est['global_dim']}`.",
        f"- Local block dimension: `{est['local_dim']}`.",
        f"- Dense global matrix size: `{bytes_to_human(est['dense_global_matrix_bytes'])}`.",
        f"- Dense local block size: `{bytes_to_human(est['dense_local_block_bytes'])}`.",
        f"- Dense global `b` vector size: `{bytes_to_human(est['dense_global_vector_bytes'])}`.",
        f"- Dense local `b_ij` vector size: `{bytes_to_human(est['dense_local_vector_bytes'])}`.",
        f"- Sparse nnz lower bound for the full matrix: `{est['sparse_nnz_lower_bound']}`.",
        f"- Float64 CSR storage lower bound for the full matrix: `{bytes_to_human(est['csr_float64_lower_bound_bytes'])}`.",
    ]

    if ram_bytes is not None:
        lines.extend(
            [
                f"- Detected system RAM: `{bytes_to_human(ram_bytes)}`.",
                f"- Dense global matrix / RAM ratio: `{est['dense_global_matrix_bytes'] / ram_bytes:.3e}`.",
                f"- Dense local block / RAM ratio: `{est['dense_local_block_bytes'] / ram_bytes:.3e}`.",
                f"- Dense global `b` vector / RAM ratio: `{est['dense_global_vector_bytes'] / ram_bytes:.3e}`.",
            ]
        )

    lines.extend(
        [
            "",
            "## Practical Reading",
            "A 30-qubit run is not blocked by the MPS ansatz circuit. It is blocked by the dense and sparse matrix construction used to get `A_ij` and `b_ij` in the current workflow.",
            "To make 30 global qubits feasible, we would need to replace the current matrix extraction path with direct MPO/MPS constructions for the local operators and RHS blocks.",
            "",
            "## Artifacts",
            f"- JSON: `{result['artifacts']['json']}`",
            f"- Report: `{result['artifacts']['report']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    global_qubits = 30
    estimates = estimate_storage(global_qubits)
    smoke = run_circuit_smoke_test(estimates["local_qubits"])
    total_ram = get_total_ram_bytes()

    json_path = THIS_DIR / "quimb_dist_eq26_2x2_feasibility_30q.json"
    report_path = THIS_DIR / "quimb_dist_eq26_2x2_feasibility_30q_report.md"

    result = {
        "storage_estimates": estimates,
        "circuit_smoke_test": smoke,
        "system": {
            "total_ram_bytes": total_ram,
            "total_ram_human": bytes_to_human(total_ram) if total_ram is not None else None,
        },
        "artifacts": {
            "json": str(json_path.resolve()),
            "report": str(report_path.resolve()),
        },
    }

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path.write_text(make_report(result), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"Wrote JSON to {json_path}")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
