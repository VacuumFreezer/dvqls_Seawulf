# Topology Benchmarks for Row-Column Connectivity Analysis

To evaluate the sensitivity of the proposed algorithm to network topology, we select six representative benchmarks using the set of non-isomorphic connected graphs with four nodes ($N=4$). These benchmarks are designed to isolate the effects of graph density, diameter, and structural symmetry across the row and column dimensions.

## 1. Selected Benchmarks

| Benchmark | Row ($G_R$) | Column ($G_C$) | Keywords |
| :--- | :--- | :--- | :--- |
| **B1: Min-Min** | $P_4$ | $P_4$ | Least connectivity, Max diameter |
| **B2: Max-Max** | $K_4$ | $K_4$ | Full connectivity, Ideal baseline |
| **B3: Row-Neck** | $P_4$ | $K_4$ | Row-wise bottleneck, Asymmetry |
| **B4: Col-Neck** | $K_4$ | $P_4$ | Column-wise bottleneck, Asymmetry |
| **B5: Hub-Centric** | $S_4$ | $S_4$ | Hub-centricity, Centralized |
| **B6: Balanced** | $C_4$ | $C_4$ | Sparse regularity, Ring topology |

---

## 2. Design Logic and Objectives

The selection follows a **controlled-variable approach** to stress-test the algorithm under different communication constraints:

* **Boundary Performance (B1 & B2):** $P_4 \times P_4$ and $K_4 \times K_4$ establish the performance envelope, representing the worst-case (minimal connectivity) and best-case (full connectivity) scenarios.
* **Dimensional Ablation (B3 & B4):** By restricting connectivity in only one dimension (Row or Column), these benchmarks reveal if the algorithm exhibits dimensional sensitivity or asymmetry in information flow.
* **Structural Impact (B5 & B6):** These test the influence of graph "shape" rather than just edge count. $S_4$ (Star) introduces a centralized hub that reduces diameter compared to $P_4$, while $C_4$ (Cycle) tests decentralized stability in a symmetric, regular graph.
