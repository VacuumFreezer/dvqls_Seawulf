# 30-Qubit Gradient And SPSA Check

## Setup

- Global qubits: `30`
- Local qubits: `29`
- Cost construction: direct MPO/MPS blocks, no dense global `A`, no dense global `b`
- Initialization: structured angles, `sigma=0.75`, `lambda=0.10`

## Direct Block Construction

For the current `2 x 2` split by the first qubit:

- `A11 = (H_rest + J Z_1 + eta I) / zeta`
- `A12 = I / zeta`
- `A21 = I / zeta`
- `A22 = (H_rest - J Z_1 + eta I) / zeta`
- `b_i = (1 / sqrt(2)) |+^{29}>`
- `b_ij = (1 / (2 sqrt(2))) |+^{29}>`

## Reverse-Mode Check

- Forward cost is finite at the initialization point.
- Reverse-mode `beta` gradient is finite.
- Reverse-mode `alpha` gradient is not finite.

Observed pattern:

- `alpha` bad entries only appear on the diagonal agents `(0, 0)` and `(1, 1)`.
- On each of those agents, the scale parameter `sigma` stays finite.
- The non-finite entries are the angle derivatives `1:231`.
- Off-diagonal agents `(0, 1)` and `(1, 0)` have fully finite `alpha` gradients.

This strongly suggests the instability is in autodiff through the nontrivial MPO application for `A11` and `A22`, not in the cost value itself.

## Forward Stability Check

Small direct perturbations of the problematic parameters still give finite cost values. For example:

- `alpha[0, 0, 1] + 1e-3` gives a finite cost
- `alpha[1, 1, 100] + 1e-3` gives a finite cost

So the forward map remains well-defined; the failure is in reverse-mode backpropagation.

## Parameter-Shift Check

Parameter-shift on individual problematic angles is finite. Example timings:

- `alpha[0, 0, 1]`: finite parameter-shift gradient in about `1.26 s`
- `alpha[1, 1, 100]`: finite parameter-shift gradient in about `1.10 s`

With the current ansatz:

- `29` local qubits
- `4` layers
- `232` `RY` angles per agent
- `4` agents

Total `alpha` angles = `928`.

So full parameter-shift for all `alpha` angles needs:

- `1856` cost evaluations
- about `0.73 s` per cost evaluation on this laptop
- estimated total time about `1358 s`, roughly `22.6 min`

This estimate excludes the `4` sigma derivatives, which should be handled analytically or with a small extra finite-difference cost.

## SPSA Check

One SPSA gradient estimate for all `alpha` parameters is feasible:

- two forward cost evaluations only
- total time about `1.46 s`
- finite gradient estimate

One simultaneous SPSA estimate for both `alpha` and `beta`, followed by one post-update cost evaluation:

- two perturbed forward evaluations plus one validation evaluation
- total time about `2.07 s`
- both estimated gradients finite
- the test step slightly reduced the cost:
  - base cost: `1.215079820136056`
  - new cost after one SPSA step: `1.2150178554558115`

## Conclusion

- Memory-wise, the direct MPO/MPS workflow is viable at `30` qubits.
- Reverse-mode `alpha` gradients are not currently reliable at `30` qubits.
- Full parameter-shift is numerically viable but expensive.
- SPSA is numerically viable and much cheaper per iteration, so it is the most practical gradient-free option among the methods checked here for the current `30`-qubit workflow.
