from __future__ import annotations

from dataclasses import dataclass

from Graph_comparison.topology_registry import (
    build_neighbor_map,
    first_agent_cost_formula,
    graph_label,
    laplacian_matrix_from_topology,
    metropolis_matrix_from_topology,
    normalize_graph_name,
)


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    name: str
    row_graph: str
    column_graph: str
    keywords: tuple[str, ...]


BENCHMARK_SPECS = (
    BenchmarkSpec("B1", "R-Path_C-Path", "P4", "P4", ("Minimal connectivity", "Maximum diameter")),
    BenchmarkSpec("B2", "R-Path_C-Ring", "P4", "C4", ("Row bottleneck", "Balanced column")),
    BenchmarkSpec("B3", "R-Path_C-Comp", "P4", "K4", ("Row bottleneck", "Ideal column")),
    BenchmarkSpec("B4", "R-Ring_C-Path", "C4", "P4", ("Balanced row", "Column bottleneck")),
    BenchmarkSpec("B5", "R-Ring_C-Ring", "C4", "C4", ("Sparse regularity", "Decentralized balance")),
    BenchmarkSpec("B6", "R-Ring_C-Comp", "C4", "K4", ("Balanced row", "Ideal column")),
    BenchmarkSpec("B7", "R-Comp_C-Path", "K4", "P4", ("Ideal row", "Column bottleneck")),
    BenchmarkSpec("B8", "R-Comp_C-Ring", "K4", "C4", ("Ideal row", "Balanced column")),
    BenchmarkSpec("B9", "R-Comp_C-Comp", "K4", "K4", ("Maximal connectivity", "Ideal baseline")),
)

BENCHMARK_BY_ID = {spec.benchmark_id: spec for spec in BENCHMARK_SPECS}


def iter_benchmarks() -> tuple[BenchmarkSpec, ...]:
    return BENCHMARK_SPECS


def get_benchmark_spec(benchmark_id: str) -> BenchmarkSpec:
    key = str(benchmark_id).strip().upper()
    if key not in BENCHMARK_BY_ID:
        valid = ", ".join(spec.benchmark_id for spec in BENCHMARK_SPECS)
        raise ValueError(f"Unknown benchmark {benchmark_id!r}. Expected one of: {valid}.")
    return BENCHMARK_BY_ID[key]


def resolve_benchmark_and_graphs(
    *,
    benchmark_id: str = "",
    topology: str = "",
    row_graph: str = "",
    column_graph: str = "",
) -> tuple[BenchmarkSpec | None, str, str]:
    benchmark = None
    if str(benchmark_id).strip():
        benchmark = get_benchmark_spec(benchmark_id)
        resolved_row = benchmark.row_graph
        resolved_col = benchmark.column_graph

        if str(topology).strip():
            topo = normalize_graph_name(topology)
            if topo != resolved_row or topo != resolved_col:
                raise ValueError(
                    f"--benchmark {benchmark.benchmark_id} fixes row/column graphs to "
                    f"{resolved_row}/{resolved_col}; it conflicts with --topology {topology!r}."
                )
        if str(row_graph).strip() and normalize_graph_name(row_graph) != resolved_row:
            raise ValueError(
                f"--benchmark {benchmark.benchmark_id} fixes row_graph={resolved_row}; "
                f"it conflicts with --row_graph {row_graph!r}."
            )
        if str(column_graph).strip() and normalize_graph_name(column_graph) != resolved_col:
            raise ValueError(
                f"--benchmark {benchmark.benchmark_id} fixes column_graph={resolved_col}; "
                f"it conflicts with --column_graph {column_graph!r}."
            )
        return benchmark, resolved_row, resolved_col

    topo_text = str(topology).strip()
    row_text = str(row_graph).strip() or topo_text
    col_text = str(column_graph).strip() or topo_text
    if not row_text or not col_text:
        raise ValueError(
            "Specify either --benchmark or both --row_graph/--column_graph. "
            "--topology may be used as a shorthand for the symmetric case."
        )
    return None, normalize_graph_name(row_text), normalize_graph_name(col_text)


__all__ = [
    "BenchmarkSpec",
    "build_neighbor_map",
    "first_agent_cost_formula",
    "graph_label",
    "iter_benchmarks",
    "laplacian_matrix_from_topology",
    "metropolis_matrix_from_topology",
    "resolve_benchmark_and_graphs",
]
