from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Graph_comparison.Ising_5qubits_9benchmark import seawulf_graph_comparison_ising_qjit as base


def write_analysis_report(
    path: Path,
    *,
    args,
    benchmark,
    row_graph: str,
    column_graph: str,
    row_topology,
    column_topology,
    row_metropolis,
    column_metropolis,
    row_laplacian,
    column_laplacian,
    ops_module_name: str,
    system,
    data_wires,
    scaffold_edges,
    global_params,
    row_b_totals,
    true_solution,
    final_metrics,
    a_global,
    global_b,
):
    even_edges, odd_edges = base._split_alternating_edges(scaffold_edges)
    with path.open("w", encoding="utf-8") as out:
        metadata = getattr(system, "metadata", {})
        out.write("2-local-qubit row-sparsity / zero-block diagnostic run\n")
        out.write(f"static_ops: {ops_module_name}\n")
        out.write(f"system key: {str(args.system_key).strip() or 'default'}\n")
        out.write(f"system name: {getattr(system, 'name', 'unknown')}\n")
        out.write(f"benchmark: {benchmark.benchmark_id if benchmark else 'custom'}\n")
        out.write(f"n agents: {system.n}\n")
        out.write(f"local data qubits: {len(data_wires)}\n")
        out.write(f"global total qubits: {int(base.np.log2(a_global.shape[0]))}\n")
        out.write(
            f"ansatz: {base.ANSATZ_PAPER_FIG3_RY_CZ} "
            f"({base.describe_ansatz(base.ANSATZ_PAPER_FIG3_RY_CZ)})\n"
        )
        out.write("ansatz block: RY(theta_1) -> CZ(even bonds) -> RY(theta_2) -> CZ(odd bonds)\n")
        out.write(f"layers: {int(args.layers)}\n")
        out.write(f"learning rate: {float(args.lr):.8f}\n")
        out.write(f"decay: {float(args.decay):.8f}\n")
        out.write(f"seed: {int(args.seed)}\n")
        out.write(f"row graph: {row_graph} ({base.graph_label(row_graph)})\n")
        out.write(f"column graph: {column_graph} ({base.graph_label(column_graph)})\n")
        out.write(f"row topology: {row_topology}\n")
        out.write(f"column topology: {column_topology}\n")
        out.write(f"first-agent cost formula: {base.first_agent_cost_formula(row_graph)}\n")
        out.write(f"scaffold_edges: {tuple((int(a), int(b)) for a, b in scaffold_edges)}\n")
        out.write(f"even_bond_cz_edges: {tuple((int(a), int(b)) for a, b in even_edges)}\n")
        out.write(f"odd_bond_cz_edges: {tuple((int(a), int(b)) for a, b in odd_edges)}\n")
        out.write(f"row Metropolis matrix:\n{base._format_array_for_report(row_metropolis)}\n")
        out.write(f"column Metropolis matrix:\n{base._format_array_for_report(column_metropolis)}\n")
        out.write(f"row Laplacian:\n{base._format_array_for_report(row_laplacian)}\n")
        out.write(f"column Laplacian:\n{base._format_array_for_report(column_laplacian)}\n")
        if metadata:
            out.write(f"metadata: {base._to_jsonable(metadata)}\n")
        out.write(f"condition number of A: {float(base.np.linalg.cond(a_global)):.8e}\n")
        out.write(f"global matrix shape: {a_global.shape}\n")
        out.write(f"global b shape: {global_b.shape}\n")
        out.write(f"final global cost: {float(final_metrics['global_cost']):.8e}\n")
        out.write(f"final ||Ax-b||: {float(final_metrics['residual_norm']):.8e}\n")
        out.write(f"final relative L2 error: {float(final_metrics['l2_error']):.8e}\n")
        out.write(f"final consensus error: {float(final_metrics['consensus_error']):.8e}\n")
        out.write(f"final row disagreement energy: {float(final_metrics['row_disagreement_energy']):.8e}\n")
        out.write(f"final row disagreement ratio: {float(final_metrics['row_disagreement_ratio']):.8e}\n")
        out.write(
            f"final row disagreement per row: "
            f"{base._format_array_for_report(final_metrics['row_disagreement_energy_per_row'])}\n"
        )
        out.write(
            f"final row disagreement ratio per row: "
            f"{base._format_array_for_report(final_metrics['row_disagreement_ratio_per_row'])}\n"
        )
        out.write(f"final recovered solution:\n{base._format_array_for_report(final_metrics['recovered_solution'])}\n")
        out.write(f"true solution:\n{base._format_array_for_report(true_solution)}\n")
        out.write(f"global b:\n{base._format_array_for_report(global_b)}\n")
        out.write(f"final params summary: {base._to_jsonable(global_params)}\n")
        out.write("\nROW RIGHT-HAND SIDES\n")
        for row_id, row_b in enumerate(row_b_totals):
            out.write(f"[row={row_id}] b_i:\n{base._format_array_for_report(row_b)}\n")


if __name__ == "__main__":
    base.write_analysis_report = write_analysis_report
    base.main()
