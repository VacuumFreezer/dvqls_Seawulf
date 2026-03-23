from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


N_GRAPH_NODES = 4

GRAPH_ALIASES = {
    "p4": "P4",
    "path": "P4",
    "path4": "P4",
    "line": "P4",
    "line4": "P4",
    "k4": "K4",
    "complete": "K4",
    "complete4": "K4",
    "s4": "S4",
    "star": "S4",
    "star4": "S4",
    "s4h1": "S4H1",
    "star_hub1": "S4H1",
    "starhub1": "S4H1",
    "s4h2": "S4H2",
    "star_hub2": "S4H2",
    "starhub2": "S4H2",
    "s4h3": "S4H3",
    "star_hub3": "S4H3",
    "starhub3": "S4H3",
    "c4": "C4",
    "cycle": "C4",
    "cycle4": "C4",
    "ring": "C4",
    "ring4": "C4",
}

GRAPH_LABELS = {
    "P4": "Path graph on four nodes",
    "K4": "Complete graph on four nodes",
    "S4": "Star graph on four nodes (hub = row 0)",
    "S4H1": "Star graph on four nodes (hub = row 1)",
    "S4H2": "Star graph on four nodes (hub = row 2)",
    "S4H3": "Star graph on four nodes (hub = row 3)",
    "C4": "Cycle graph on four nodes",
}

GRAPH_EDGES = {
    "P4": ((0, 1), (1, 2), (2, 3)),
    "K4": ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
    "S4": ((0, 1), (0, 2), (0, 3)),
    "S4H1": ((1, 0), (1, 2), (1, 3)),
    "S4H2": ((2, 0), (2, 1), (2, 3)),
    "S4H3": ((3, 0), (3, 1), (3, 2)),
    "C4": ((0, 1), (1, 2), (2, 3), (3, 0)),
}


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    name: str
    row_graph: str
    column_graph: str
    keywords: tuple[str, ...]


BENCHMARK_SPECS = (
    BenchmarkSpec("B1", "Min-Min", "P4", "P4", ("Least connectivity", "Max diameter")),
    BenchmarkSpec("B2", "Max-Max", "K4", "K4", ("Full connectivity", "Ideal baseline")),
    BenchmarkSpec("B3", "Row-Neck", "P4", "K4", ("Row-wise bottleneck", "Asymmetry")),
    BenchmarkSpec("B4", "Col-Neck", "K4", "P4", ("Column-wise bottleneck", "Asymmetry")),
    BenchmarkSpec("B5", "Hub-Centric", "S4", "S4", ("Hub-centricity", "Centralized")),
    BenchmarkSpec("B6", "Balanced", "C4", "C4", ("Sparse regularity", "Ring topology")),
)

BENCHMARK_BY_ID = {spec.benchmark_id: spec for spec in BENCHMARK_SPECS}


def iter_benchmarks() -> tuple[BenchmarkSpec, ...]:
    return BENCHMARK_SPECS


def normalize_graph_name(raw_name: str) -> str:
    text = str(raw_name).strip()
    if not text:
        raise ValueError("Graph name must be a non-empty string.")

    if text in GRAPH_LABELS:
        return text

    key = text.lower()
    if key in GRAPH_ALIASES:
        return GRAPH_ALIASES[key]

    valid = ", ".join(sorted(GRAPH_LABELS))
    raise ValueError(f"Unsupported graph name {raw_name!r}. Expected one of: {valid}.")


def graph_label(graph_name: str) -> str:
    canonical = normalize_graph_name(graph_name)
    return GRAPH_LABELS[canonical]


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


def build_neighbor_map(graph_name: str, *, n_nodes: int = N_GRAPH_NODES) -> dict[int, list[int]]:
    if int(n_nodes) != N_GRAPH_NODES:
        raise ValueError(f"This workflow expects n_nodes={N_GRAPH_NODES}, received {n_nodes}.")

    canonical = normalize_graph_name(graph_name)
    neighbors = {node: set() for node in range(int(n_nodes))}
    for left, right in GRAPH_EDGES[canonical]:
        neighbors[left].add(right)
        neighbors[right].add(left)
    return {node: sorted(neighbors[node]) for node in range(int(n_nodes))}


def adjacency_matrix_from_topology(topology: dict[int, list[int]]) -> np.ndarray:
    n_nodes = len(topology)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    for node, neighbors in topology.items():
        for neighbor in neighbors:
            adjacency[int(node), int(neighbor)] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def adjacency_matrix_from_graph(graph_name: str) -> np.ndarray:
    return adjacency_matrix_from_topology(build_neighbor_map(graph_name))


def laplacian_matrix_from_topology(topology: dict[int, list[int]]) -> np.ndarray:
    adjacency = adjacency_matrix_from_topology(topology)
    degree = np.diag(np.sum(adjacency, axis=1))
    return degree - adjacency


def laplacian_matrix_from_graph(graph_name: str) -> np.ndarray:
    return laplacian_matrix_from_topology(build_neighbor_map(graph_name))


def metropolis_matrix_from_topology(topology: dict[int, list[int]]) -> np.ndarray:
    n_nodes = len(topology)
    adjacency = [set() for _ in range(n_nodes)]
    for node, neighbors in topology.items():
        for neighbor in neighbors:
            if int(node) == int(neighbor):
                continue
            adjacency[int(node)].add(int(neighbor))
            adjacency[int(neighbor)].add(int(node))

    degrees_including_self = [len(adjacency[node]) + 1 for node in range(n_nodes)]
    weights = np.zeros((n_nodes, n_nodes), dtype=float)

    for node in range(n_nodes):
        if not adjacency[node]:
            weights[node, node] = 1.0
            continue

        off_diag_sum = 0.0
        for neighbor in sorted(adjacency[node]):
            value = 1.0 / (1.0 + max(degrees_including_self[node], degrees_including_self[neighbor]))
            weights[node, neighbor] = value
            off_diag_sum += value
        weights[node, node] = 1.0 - off_diag_sum

    return weights


def metropolis_matrix_from_graph(graph_name: str) -> np.ndarray:
    return metropolis_matrix_from_topology(build_neighbor_map(graph_name))


def _format_symbolic_linear_combination(coeffs: Iterable[int], *, symbol_prefix: str) -> str:
    pieces: list[str] = []
    for idx, coeff in enumerate(coeffs, start=1):
        coeff = int(coeff)
        if coeff == 0:
            continue

        term = f"{symbol_prefix}{idx}" + "}"
        abs_coeff = abs(coeff)
        if abs_coeff != 1:
            term = f"{abs_coeff}{term}"

        if not pieces:
            pieces.append(f"-{term}" if coeff < 0 else term)
            continue

        sign = " - " if coeff < 0 else " + "
        pieces.append(sign + term)

    return "".join(pieces) if pieces else "0"


def first_agent_cost_formula(row_graph: str) -> str:
    laplacian = laplacian_matrix_from_graph(row_graph)
    first_row = laplacian[0].astype(int).tolist()
    laplacian_term = _format_symbolic_linear_combination(first_row, symbol_prefix="z_{1")
    return (
        r"f_{11}(x_{11}, z_{11}, z_{12}, z_{13}, z_{14})"
        r" = \left\|A_{11}x_{11} - b_{11} - \left("
        + laplacian_term
        + r"\right)\right\|^2"
    )


def benchmark_summary_lines() -> list[str]:
    lines = []
    for spec in BENCHMARK_SPECS:
        lines.append(
            f"{spec.benchmark_id}: {spec.name} | row={spec.row_graph} | column={spec.column_graph} | "
            + ", ".join(spec.keywords)
        )
    return lines
