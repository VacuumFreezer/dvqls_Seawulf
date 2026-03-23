from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from Partition_comparison_qjit.New_stabilizer.benchmark_13q_xzx_fresh_common import (
    BASE_BLOCK_ANGLE,
    B_PREP_HADAMARD,
    B_STATE_ALL_PLUS,
    LinearSystemData,
    N_TOTAL_QUBITS,
    _identity_gate_factory,
    _local_chain_edge_layers,
    _matrix_element_index_register,
    _normalize_b_prep_kind,
    _pauli_word_factory,
    _prefix_block_scale,
    _product_hadamard_factory,
    _product_ry_factory,
    _product_state_angles_for_global_qubits,
    _product_state_vector,
    bitstring_label,
    bitstring_tuple,
)


J_PERTURB = 0.1
LAMBDA_MIN_TARGET = 1.0 / 20.0
LAMBDA_MAX_TARGET = 1.0
RAW_IDENTITY_COEFF = 3.5
RAW_X_BLOCK_COEFF = -0.5
RAW_X13_COEFF = -0.5

XXX_TERMS: Tuple[Tuple[str, Dict[int, str]], ...] = (
    ("S_1", {1: "X", 2: "X", 3: "X"}),
    ("S_2", {4: "X", 5: "X", 6: "X"}),
    ("S_3", {7: "X", 8: "X", 9: "X"}),
    ("S_4", {10: "X", 11: "X", 12: "X"}),
)

ZZZ_TERMS: Tuple[Tuple[str, Dict[int, str]], ...] = (
    ("T_1", {1: "Z", 2: "Z", 3: "Z"}),
    ("T_2", {4: "Z", 5: "Z", 6: "Z"}),
    ("T_3", {7: "Z", 8: "Z", 9: "Z"}),
    ("T_4", {10: "Z", 11: "Z", 12: "Z"}),
)


def raw_block_radius(j_perturb: float) -> float:
    return float(np.sqrt(0.25 + float(j_perturb) ** 2))


def raw_spectrum_extrema(j_perturb: float) -> tuple[float, float]:
    radius = raw_block_radius(j_perturb)
    lambda_min = 3.0 - (4.0 * radius)
    lambda_max = 4.0 + (4.0 * radius)
    return float(lambda_min), float(lambda_max)


def affine_normalization(j_perturb: float) -> tuple[float, float]:
    raw_min, raw_max = raw_spectrum_extrema(j_perturb)
    scale = (float(LAMBDA_MAX_TARGET) - float(LAMBDA_MIN_TARGET)) / (raw_max - raw_min)
    shift = float(LAMBDA_MAX_TARGET) - (scale * raw_max)
    return float(scale), float(shift)


def build_zzz_perturb_namespace(
    index_qubits: int,
    *,
    j_perturb: float = J_PERTURB,
    b_prep_kind: str = B_PREP_HADAMARD,
) -> dict:
    k = int(index_qubits)
    if k < 0 or k > 3:
        raise ValueError(f"Expected index_qubits in {{0, 1, 2, 3}}, got {k}.")

    b_prep_kind = _normalize_b_prep_kind(b_prep_kind)
    n_agents = 1 << k
    n_data_qubits = N_TOTAL_QUBITS - k
    data_wires = list(range(n_data_qubits))
    local_global_qubits = list(range(k + 1, N_TOTAL_QUBITS + 1))
    odd_edges_local, even_edges_local, scaffold_edges_local = _local_chain_edge_layers(k)

    scale, shift = affine_normalization(j_perturb)
    identity_coeff = (scale * RAW_IDENTITY_COEFF) + shift
    xxx_coeff = scale * RAW_X_BLOCK_COEFF
    x13_coeff = scale * RAW_X13_COEFF
    zzz_coeff = scale * float(j_perturb)

    global_terms: List[Tuple[float, Dict[int, str], str]] = [(identity_coeff, {}, "I_scaled")]
    for label, pauli_map in XXX_TERMS:
        global_terms.append((xxx_coeff, dict(pauli_map), label))
    global_terms.append((x13_coeff, {13: "X"}, "X_13"))
    for label, pauli_map in ZZZ_TERMS:
        global_terms.append((zzz_coeff, dict(pauli_map), label))

    b_local_angles = _product_state_angles_for_global_qubits(
        local_global_qubits,
        state_kind=B_STATE_ALL_PLUS,
        last_angle=BASE_BLOCK_ANGLE,
    )
    if b_prep_kind == B_PREP_HADAMARD:
        b_gate = _product_hadamard_factory(data_wires, b_local_angles)
    else:
        b_gate = _product_ry_factory(data_wires, b_local_angles)
    x_gate = _product_ry_factory(data_wires, b_local_angles)
    b_local_state = _product_state_vector(b_local_angles)

    row_labels = [bitstring_label(index, k) for index in range(n_agents)]
    row_b_norms = []
    row_x_scales = []
    row_b_state = [b_local_state for _ in range(n_agents)]
    row_x_state = [b_local_state for _ in range(n_agents)]
    for index in range(n_agents):
        bits = bitstring_tuple(index, k)
        block_scale = _prefix_block_scale(bits, state_kind=B_STATE_ALL_PLUS, last_angle=BASE_BLOCK_ANGLE)
        row_b_norms.append(block_scale)
        row_x_scales.append(block_scale)

    b_weights = [
        [
            float(row_b_norms[row_id]) / float(n_agents)
            for _ in range(n_agents)
        ]
        for row_id in range(n_agents)
    ]
    raw_b_gates = [[b_gate for _ in range(n_agents)] for __ in range(n_agents)]

    gates_grid: List[List[List[Callable[[], object]]]] = [[[] for _ in range(n_agents)] for __ in range(n_agents)]
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
                term_breakdown_grid[row_id][col_id] = [{"label": "0", "coefficient": 0.0, "local_paulis": {}}]
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
                term_breakdown_grid[row_id][col_id] = [{"label": "0", "coefficient": 0.0, "local_paulis": {}}]

    radius = raw_block_radius(j_perturb)
    raw_eigs = sorted(
        {
            RAW_IDENTITY_COEFF + (float(sign_x13) * 0.5) + (radius * sum(float(sign) for sign in block_signs))
            for sign_x13 in (-1.0, 1.0)
            for block_signs in (
                (s1, s2, s3, s4)
                for s1 in (-1.0, 1.0)
                for s2 in (-1.0, 1.0)
                for s3 in (-1.0, 1.0)
                for s4 in (-1.0, 1.0)
            )
        }
    )
    distinct_eigenvalues = [float((scale * eig) + shift) for eig in raw_eigs]
    spectrum_info = {
        "lambda_min": float(LAMBDA_MIN_TARGET),
        "lambda_max": float(LAMBDA_MAX_TARGET),
        "condition_number": float(LAMBDA_MAX_TARGET / LAMBDA_MIN_TARGET),
        "n_distinct_eigenvalues": len(distinct_eigenvalues),
        "distinct_eigenvalues": distinct_eigenvalues,
        "raw_lambda_min": float(raw_spectrum_extrema(j_perturb)[0]),
        "raw_lambda_max": float(raw_spectrum_extrema(j_perturb)[1]),
        "affine_scale": float(scale),
        "affine_shift": float(shift),
        "b_is_eigenvector": False,
    }

    metadata = {
        "benchmark_name": "13q_zzz_perturb_prefix_partition",
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
        "b_state_kind": B_STATE_ALL_PLUS,
        "b_prep_kind": b_prep_kind,
        "j_perturb": float(j_perturb),
        "target_lambda_min": float(LAMBDA_MIN_TARGET),
        "target_lambda_max": float(LAMBDA_MAX_TARGET),
        "identity_coeff": float(identity_coeff),
        "xxx_coeff": float(xxx_coeff),
        "x13_coeff": float(x13_coeff),
        "zzz_coeff": float(zzz_coeff),
        "odd_cz_edges_local": odd_edges_local,
        "even_cz_edges_local": even_edges_local,
        "cluster_scaffold_edges_local": scaffold_edges_local,
        "global_term_count": len(global_terms),
        "block_term_counts": [[len(cell) for cell in row] for row in gates_grid],
        "spectrum": spectrum_info,
        "real_problem": True,
        "recommended_ansatz": "hadamard_brickwall_ry_cz",
        "recommended_layers": 2,
        "exact_solution_family": "computed_via_sparse_solve",
        "init_sigma_target": 1.0,
        "init_angle_fill": BASE_BLOCK_ANGLE,
        "agent_init_overrides": {},
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
        name=f"zzz13_all_plus_{b_prep_kind}_prefix_partition_{partition_label}",
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
        "J_PERTURB": float(j_perturb),
        "LAMBDA_MIN_TARGET": float(LAMBDA_MIN_TARGET),
        "LAMBDA_MAX_TARGET": float(LAMBDA_MAX_TARGET),
        "ROW_B_NORMS": tuple(float(x) for x in row_b_norms),
        "ROW_X_SCALES": tuple(float(x) for x in row_x_scales),
        "B_STATE_KIND": B_STATE_ALL_PLUS,
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


globals().update(build_zzz_perturb_namespace(2, j_perturb=J_PERTURB, b_prep_kind=B_PREP_HADAMARD))
