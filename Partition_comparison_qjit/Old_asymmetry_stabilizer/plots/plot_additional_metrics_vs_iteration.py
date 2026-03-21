#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

from plot_global_cost_vs_iteration import CURVES, CurveSpec, make_metric_plot


@dataclass(frozen=True)
class MetricPlotSpec:
    metric_key: str
    y_label: str
    y_floor: float
    output_stem: str
    curves: tuple[CurveSpec, ...]
    max_iteration: int = 9000


PLOTS = (
    MetricPlotSpec(
        metric_key="residual_norm",
        y_label=r"residual norm $\|Ax-b\|$",
        y_floor=1e-2,
        output_stem="residual_norm_vs_iteration_partition_comparison",
        curves=CURVES,
    ),
    MetricPlotSpec(
        metric_key="l2_error",
        y_label=r"relative $L_2$ error of solution",
        y_floor=1e-2,
        output_stem="relative_l2_error_of_solution_vs_iteration_partition_comparison",
        curves=CURVES,
    ),
    MetricPlotSpec(
        metric_key="consensus_error",
        y_label="consensus error",
        y_floor=1e-5,
        output_stem="consensus_error_vs_iteration_partition_comparison",
        curves=CURVES[1:],
        max_iteration=1000,
    ),
)


def main() -> None:
    for plot in PLOTS:
        png_path, pdf_path = make_metric_plot(
            curves=plot.curves,
            metric_key=plot.metric_key,
            y_label=plot.y_label,
            y_floor=plot.y_floor,
            output_stem=plot.output_stem,
            max_iteration=plot.max_iteration,
        )
        print(f"saved {png_path}")
        print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
