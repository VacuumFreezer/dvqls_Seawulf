# Graph Comparison Topology Reference

Node ordering is `1, 2, 3, 4` for both row and column graphs.
For the star graph `S4`, node `1` is used as the hub.

The optimizer uses the column-graph Metropolis matrix for the `x/y` consensus step and the row-graph Laplacian for the `z` coupling term. The row-side Metropolis matrix is listed below as requested.

## B1: Min-Min

- Row graph: `P4` (Path graph on four nodes)
- Column graph: `P4` (Path graph on four nodes)
- Keywords: Least connectivity, Max diameter

### Row Metropolis Matrix

```text
[[0.75 0.25 0.   0.  ]
 [0.25 0.5  0.25 0.  ]
 [0.   0.25 0.5  0.25]
 [0.   0.   0.25 0.75]]
```

### Column Metropolis Matrix

```text
[[0.75 0.25 0.   0.  ]
 [0.25 0.5  0.25 0.  ]
 [0.   0.25 0.5  0.25]
 [0.   0.   0.25 0.75]]
```

### Row Laplacian

```text
[[ 1. -1.  0.  0.]
 [-1.  2. -1.  0.]
 [ 0. -1.  2. -1.]
 [ 0.  0. -1.  1.]]
```

### Column Laplacian

```text
[[ 1. -1.  0.  0.]
 [-1.  2. -1.  0.]
 [ 0. -1.  2. -1.]
 [ 0.  0. -1.  1.]]
```

### First-Agent Local Cost

The first agent is taken to be `[[1,1]]`, so its local row-neighborhood is determined by the first row of the row-graph Laplacian.

```math
f_{11}(x_{11}, z_{11}, z_{12}, z_{13}, z_{14}) = \left\|A_{11}x_{11} - b_{11} - \left(z_{11} - z_{12}\right)\right\|^2
```

## B2: Max-Max

- Row graph: `K4` (Complete graph on four nodes)
- Column graph: `K4` (Complete graph on four nodes)
- Keywords: Full connectivity, Ideal baseline

### Row Metropolis Matrix

```text
[[0.4 0.2 0.2 0.2]
 [0.2 0.4 0.2 0.2]
 [0.2 0.2 0.4 0.2]
 [0.2 0.2 0.2 0.4]]
```

### Column Metropolis Matrix

```text
[[0.4 0.2 0.2 0.2]
 [0.2 0.4 0.2 0.2]
 [0.2 0.2 0.4 0.2]
 [0.2 0.2 0.2 0.4]]
```

### Row Laplacian

```text
[[ 3. -1. -1. -1.]
 [-1.  3. -1. -1.]
 [-1. -1.  3. -1.]
 [-1. -1. -1.  3.]]
```

### Column Laplacian

```text
[[ 3. -1. -1. -1.]
 [-1.  3. -1. -1.]
 [-1. -1.  3. -1.]
 [-1. -1. -1.  3.]]
```

### First-Agent Local Cost

The first agent is taken to be `[[1,1]]`, so its local row-neighborhood is determined by the first row of the row-graph Laplacian.

```math
f_{11}(x_{11}, z_{11}, z_{12}, z_{13}, z_{14}) = \left\|A_{11}x_{11} - b_{11} - \left(3z_{11} - z_{12} - z_{13} - z_{14}\right)\right\|^2
```

## B3: Row-Neck

- Row graph: `P4` (Path graph on four nodes)
- Column graph: `K4` (Complete graph on four nodes)
- Keywords: Row-wise bottleneck, Asymmetry

### Row Metropolis Matrix

```text
[[0.75 0.25 0.   0.  ]
 [0.25 0.5  0.25 0.  ]
 [0.   0.25 0.5  0.25]
 [0.   0.   0.25 0.75]]
```

### Column Metropolis Matrix

```text
[[0.4 0.2 0.2 0.2]
 [0.2 0.4 0.2 0.2]
 [0.2 0.2 0.4 0.2]
 [0.2 0.2 0.2 0.4]]
```

### Row Laplacian

```text
[[ 1. -1.  0.  0.]
 [-1.  2. -1.  0.]
 [ 0. -1.  2. -1.]
 [ 0.  0. -1.  1.]]
```

### Column Laplacian

```text
[[ 3. -1. -1. -1.]
 [-1.  3. -1. -1.]
 [-1. -1.  3. -1.]
 [-1. -1. -1.  3.]]
```

### First-Agent Local Cost

The first agent is taken to be `[[1,1]]`, so its local row-neighborhood is determined by the first row of the row-graph Laplacian.

```math
f_{11}(x_{11}, z_{11}, z_{12}, z_{13}, z_{14}) = \left\|A_{11}x_{11} - b_{11} - \left(z_{11} - z_{12}\right)\right\|^2
```

## B4: Col-Neck

- Row graph: `K4` (Complete graph on four nodes)
- Column graph: `P4` (Path graph on four nodes)
- Keywords: Column-wise bottleneck, Asymmetry

### Row Metropolis Matrix

```text
[[0.4 0.2 0.2 0.2]
 [0.2 0.4 0.2 0.2]
 [0.2 0.2 0.4 0.2]
 [0.2 0.2 0.2 0.4]]
```

### Column Metropolis Matrix

```text
[[0.75 0.25 0.   0.  ]
 [0.25 0.5  0.25 0.  ]
 [0.   0.25 0.5  0.25]
 [0.   0.   0.25 0.75]]
```

### Row Laplacian

```text
[[ 3. -1. -1. -1.]
 [-1.  3. -1. -1.]
 [-1. -1.  3. -1.]
 [-1. -1. -1.  3.]]
```

### Column Laplacian

```text
[[ 1. -1.  0.  0.]
 [-1.  2. -1.  0.]
 [ 0. -1.  2. -1.]
 [ 0.  0. -1.  1.]]
```

### First-Agent Local Cost

The first agent is taken to be `[[1,1]]`, so its local row-neighborhood is determined by the first row of the row-graph Laplacian.

```math
f_{11}(x_{11}, z_{11}, z_{12}, z_{13}, z_{14}) = \left\|A_{11}x_{11} - b_{11} - \left(3z_{11} - z_{12} - z_{13} - z_{14}\right)\right\|^2
```

## B5: Hub-Centric

- Row graph: `S4` (Star graph on four nodes (node 1 is the hub))
- Column graph: `S4` (Star graph on four nodes (node 1 is the hub))
- Keywords: Hub-centricity, Centralized

### Row Metropolis Matrix

```text
[[0.4 0.2 0.2 0.2]
 [0.2 0.8 0.  0. ]
 [0.2 0.  0.8 0. ]
 [0.2 0.  0.  0.8]]
```

### Column Metropolis Matrix

```text
[[0.4 0.2 0.2 0.2]
 [0.2 0.8 0.  0. ]
 [0.2 0.  0.8 0. ]
 [0.2 0.  0.  0.8]]
```

### Row Laplacian

```text
[[ 3. -1. -1. -1.]
 [-1.  1.  0.  0.]
 [-1.  0.  1.  0.]
 [-1.  0.  0.  1.]]
```

### Column Laplacian

```text
[[ 3. -1. -1. -1.]
 [-1.  1.  0.  0.]
 [-1.  0.  1.  0.]
 [-1.  0.  0.  1.]]
```

### First-Agent Local Cost

The first agent is taken to be `[[1,1]]`, so its local row-neighborhood is determined by the first row of the row-graph Laplacian.

```math
f_{11}(x_{11}, z_{11}, z_{12}, z_{13}, z_{14}) = \left\|A_{11}x_{11} - b_{11} - \left(3z_{11} - z_{12} - z_{13} - z_{14}\right)\right\|^2
```

## B6: Balanced

- Row graph: `C4` (Cycle graph on four nodes)
- Column graph: `C4` (Cycle graph on four nodes)
- Keywords: Sparse regularity, Ring topology

### Row Metropolis Matrix

```text
[[0.5  0.25 0.   0.25]
 [0.25 0.5  0.25 0.  ]
 [0.   0.25 0.5  0.25]
 [0.25 0.   0.25 0.5 ]]
```

### Column Metropolis Matrix

```text
[[0.5  0.25 0.   0.25]
 [0.25 0.5  0.25 0.  ]
 [0.   0.25 0.5  0.25]
 [0.25 0.   0.25 0.5 ]]
```

### Row Laplacian

```text
[[ 2. -1.  0. -1.]
 [-1.  2. -1.  0.]
 [ 0. -1.  2. -1.]
 [-1.  0. -1.  2.]]
```

### Column Laplacian

```text
[[ 2. -1.  0. -1.]
 [-1.  2. -1.  0.]
 [ 0. -1.  2. -1.]
 [-1.  0. -1.  2.]]
```

### First-Agent Local Cost

The first agent is taken to be `[[1,1]]`, so its local row-neighborhood is determined by the first row of the row-graph Laplacian.

```math
f_{11}(x_{11}, z_{11}, z_{12}, z_{13}, z_{14}) = \left\|A_{11}x_{11} - b_{11} - \left(2z_{11} - z_{12} - z_{14}\right)\right\|^2
```
