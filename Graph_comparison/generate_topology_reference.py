#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Graph_comparison.topology_registry import (
    first_agent_cost_formula,
    graph_label,
    iter_benchmarks,
    laplacian_matrix_from_graph,
    metropolis_matrix_from_graph,
)


OUTPUT_PATH = ROOT / "Graph_comparison" / "topology_reference_details.md"


def _matrix_block(matrix: np.ndarray) -> str:
    return np.array2string(
        np.asarray(matrix, dtype=float),
        precision=6,
        suppress_small=False,
        separator=" ",
        max_line_width=120,
    )


def build_markdown() -> str:
    lines: list[str] = []
    lines.append("# Graph Comparison Topology Reference")
    lines.append("")
    lines.append("Node ordering is `1, 2, 3, 4` for both row and column graphs.")
    lines.append("For the star graph `S4`, node `1` is used as the hub.")
    lines.append("")
    lines.append(
        "The optimizer uses the column-graph Metropolis matrix for the `x/y` consensus step and the row-graph "
        "Laplacian for the `z` coupling term. The row-side Metropolis matrix is listed below as requested."
    )
    lines.append("")

    for spec in iter_benchmarks():
        row_w = metropolis_matrix_from_graph(spec.row_graph)
        col_w = metropolis_matrix_from_graph(spec.column_graph)
        row_l = laplacian_matrix_from_graph(spec.row_graph)
        col_l = laplacian_matrix_from_graph(spec.column_graph)

        lines.append(f"## {spec.benchmark_id}: {spec.name}")
        lines.append("")
        lines.append(f"- Row graph: `{spec.row_graph}` ({graph_label(spec.row_graph)})")
        lines.append(f"- Column graph: `{spec.column_graph}` ({graph_label(spec.column_graph)})")
        lines.append(f"- Keywords: {', '.join(spec.keywords)}")
        lines.append("")
        lines.append("### Row Metropolis Matrix")
        lines.append("")
        lines.append("```text")
        lines.append(_matrix_block(row_w))
        lines.append("```")
        lines.append("")
        lines.append("### Column Metropolis Matrix")
        lines.append("")
        lines.append("```text")
        lines.append(_matrix_block(col_w))
        lines.append("```")
        lines.append("")
        lines.append("### Row Laplacian")
        lines.append("")
        lines.append("```text")
        lines.append(_matrix_block(row_l))
        lines.append("```")
        lines.append("")
        lines.append("### Column Laplacian")
        lines.append("")
        lines.append("```text")
        lines.append(_matrix_block(col_l))
        lines.append("```")
        lines.append("")
        lines.append("### First-Agent Local Cost")
        lines.append("")
        lines.append("The first agent is taken to be `[[1,1]]`, so its local row-neighborhood is determined by the first row of the row-graph Laplacian.")
        lines.append("")
        lines.append("```math")
        lines.append(first_agent_cost_formula(spec.row_graph))
        lines.append("```")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    OUTPUT_PATH.write_text(build_markdown(), encoding="utf-8")
    print(f"[OK] wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
