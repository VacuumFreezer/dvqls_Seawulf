# 13-Qubit Benchmark with Uniform \(XZX\) Stabilizers

We consider a 13-qubit real-valued benchmark linear system
\[
A(\epsilon)\ket{x(\epsilon)} = \ket{b},
\]
constructed from a uniform three-body stabilizer structure on the first 12 qubits together with a local perturbation on the 13th qubit.

## Matrix \(A\)

The matrix is defined as
\[
A(\epsilon) = \alpha I + \beta\left(K_1 + K_2 + K_3 + K_4\right) + \epsilon Z_{13},
\]
where
\[
K_1 = X_1 Z_2 X_3, \qquad
K_2 = X_4 Z_5 X_6, \qquad
K_3 = X_7 Z_8 X_9, \qquad
K_4 = X_{10} Z_{11} X_{12}.
\]

This construction has the following features:

- The first 12 qubits are all involved in the stabilizer sector.
- All four stabilizers have the same three-body form.
- The perturbation is carried only by the final term \(\epsilon Z_{13}\).
- The LCU decomposition contains only 6 terms:
  - 1 identity term,
  - 4 stabilizer terms,
  - 1 perturbation term.

## Reference vector \(b\)

The right-hand-side vector is chosen as
\[
\ket{b}
=
\ket{+\,0\,+}_{123}
\otimes
\ket{+\,0\,+}_{456}
\otimes
\ket{+\,0\,+}_{789}
\otimes
\ket{+\,0\,+}_{10,11,12}
\otimes
\ket{+}_{13}.
\]

Equivalently,
\[
\ket{b} = U_b \ket{0}^{\otimes 13},
\]
with
\[
U_b =
\prod_{j \in \{1,3,4,6,7,9,10,12,13\}}
R_Y^{(j)}\!\left(\frac{\pi}{2}\right).
\]

Hence \(\ket{b}\) is a fully real shallow state prepared by a single layer of \(R_Y\) rotations.

## Unperturbed solution

Because \(\ket{b}\) is a common \(+1\) eigenstate of the four stabilizers \(K_1, K_2, K_3, K_4\), the unperturbed system satisfies
\[
\ket{x(0)} = \ket{b}.
\]

## Perturbed solution and shallow ansatz

For nonzero but small \(\epsilon\), the perturbation only changes the state on the 13th qubit. Therefore the exact solution \(\ket{x(\epsilon)}\) remains in the same shallow real-valued ansatz family. In particular, it can still be represented by a single layer of \(R_Y\) rotations.

This also makes the benchmark convenient for hierarchical prefix partitions such as \(1\times 1\), \(2\times 2\), \(4\times 4\), and \(8\times 8\): each block solution remains proportional to a state in the same shallow real ansatz family.

## Spectrum requirement

The coefficients \(\alpha\) and \(\beta\) are tuned so that the spectrum of \(A\) satisfies
\[
\lambda_{\max}(A) = 1, \qquad
\lambda_{\min}(A) = \frac{1}{20},
\]
which fixes the condition number to
\[
\kappa(A) = 20.
\]

## Motivation

This benchmark is designed to satisfy several requirements simultaneously:

- real-valued \(A\), \(\ket{b}\), and ansatz,
- sparse LCU structure,
- uniform stabilizer blocks on the first 12 qubits,
- a controllable perturbation on the final qubit,
- exact solvability at \(\epsilon = 0\),
- shallow representability of both the global solution and the blockwise solutions.
