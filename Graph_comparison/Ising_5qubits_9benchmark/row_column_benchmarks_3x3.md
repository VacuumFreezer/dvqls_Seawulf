# Row-Column Topology Interaction Benchmarks ($3 \times 3$ Grid)

To systematically evaluate the interaction between row-wise and column-wise communication, we select nine benchmarks by crossing three representative graphs: **Path ($P_4$)**, **Ring ($C_4$)**, and **Complete ($K_4$)**.

## Selected Benchmarks

The benchmarks are named using the format `R-[Topology]_C-[Topology]` for immediate identification of the network configuration.

| Benchmark Name | Row Graph ($G_R$) | Column Graph ($G_C$) | Keywords |
| :--- | :--- | :--- | :--- |
| **R-Path_C-Path** | Path | Path | Minimal connectivity, Maximum diameter |
| **R-Path_C-Ring** | Path | Ring | Row bottleneck, Balanced column |
| **R-Path_C-Comp** | Path | Complete | Row bottleneck, Ideal column |
| **R-Ring_C-Path** | Ring | Path | Balanced row, Column bottleneck |
| **R-Ring_C-Ring** | Ring | Ring | Sparse regularity, Decentralized balance |
| **R-Ring_C-Comp** | Ring | Complete | Balanced row, Ideal column |
| **R-Comp_C-Path** | Complete | Path | Ideal row, Column bottleneck |
| **R-Comp_C-Ring** | Complete | Ring | Ideal row, Balanced column |
| **R-Comp_C-Comp** | Complete | Complete | Maximal connectivity, Ideal baseline |

---

## Analysis Objectives

1. **Horizontal Scaling:** Fix the row topology and upgrade the column connectivity to observe marginal gains.
2. **Vertical Scaling:** Fix the column topology and upgrade the row connectivity to identify dimensional dependencies.
3. **Symmetry Testing:** Compare pairs like `R-Path_C-Comp` vs `R-Comp_C-Path` to detect if the algorithm favors one communication axis over the other.
