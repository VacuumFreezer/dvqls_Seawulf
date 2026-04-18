# 13-qubit full-cluster-stabilizer benchmark: assessment

## Proposed setup

Let the 13-qubit linear cluster-state stabilizers be
\[
K_1 = X_1 Z_2,\qquad
K_i = Z_{i-1} X_i Z_{i+1}\ \ (2\le i\le 12),\qquad
K_{13} = Z_{12} X_{13}.
\]

Take the reference state to be the standard 13-qubit linear cluster state
\[
\ket{b} = U_b \ket{0}^{\otimes 13},
\]
with
\[
U_b
=
\left(\prod_{j\ \mathrm{even}} CZ_{j,j+1}\right)
\left(\prod_{j\ \mathrm{odd}} CZ_{j,j+1}\right)
\left(\prod_{j=1}^{13} R_Y^{(j)}(\pi/2)\right).
\]

Consider the matrix
\[
A(\epsilon) = \alpha I + \beta \sum_{i=1}^{13} K_i + \epsilon X_{13}.
\]

## Important note on term count

Strictly speaking, this is **not** a 14-term LCU decomposition.

- \(I\): 1 term
- all 13 stabilizers \(K_i\): 13 terms
- perturbation \(\epsilon X_{13}\): 1 additional term

So the total number of LCU terms is

\[
1 + 13 + 1 = 15.
\]

If one insists on exactly 14 terms, then the perturbation cannot be added as an extra independent Pauli string; it would need to be absorbed into the coefficient of an existing term.

---

## Verdict: does this setup satisfy the requirements (except “all blocks nonzero”)?

### 1. Real-valued \(A\) and \(\ket{b}\): Yes

All Pauli strings in \(A\) are composed of \(X\) and \(Z\), so \(A\) is real.
The cluster state \(\ket{b}\) is prepared by real \(R_Y\) rotations and \(CZ\) gates, so it is also real.

### 2. At zero perturbation, the exact solution is \(\ket{b}\): Yes, after tuning \(\alpha,\beta\)

Since the cluster state is the simultaneous \(+1\) eigenstate of all stabilizers,
\[
K_i \ket{b} = \ket{b},\qquad i=1,\dots,13,
\]
we have
\[
A(0)\ket{b} = (\alpha + 13\beta)\ket{b}.
\]
Therefore, by imposing
\[
\alpha + 13\beta = 1,
\]
one gets
\[
A(0)\ket{b} = \ket{b},
\]
hence
\[
\ket{x(0)} = \ket{b}.
\]

### 3. The perturbation is genuine: Yes

The extra term is \(\epsilon X_{13}\), while \(\ket{b}\) is not a \(+1\) eigenstate of bare \(X_{13}\) in the cluster-state basis.
So for \(\epsilon \neq 0\), the solution direction generally changes and does not remain exactly equal to \(\ket{b}\).

### 4. The exact solution remains in the shallow brickwall-\(CZ\)+\(R_Y\) ansatz family: Yes

Conjugating by the cluster-state preparation circuit \(U_b\),
\[
U_b^\dagger K_i U_b = Z_i,
\]
and for the perturbation,
\[
U_b^\dagger X_{13} U_b = X_{12} Z_{13}.
\]

Therefore
\[
U_b^\dagger A(\epsilon) U_b
=
\alpha I + \beta \sum_{i=1}^{13} Z_i + \epsilon X_{12} Z_{13}.
\]

Acting on the reference computational basis state \(\ket{0}^{\otimes 13}\), this reduces to a nontrivial problem only on qubit 12 (with qubit 13 fixed in \(\ket{0}\)).
Hence the exact solution in the rotated basis is still a product state with only one modified single-qubit factor.
After transforming back by \(U_b\), the exact state \(\ket{x(\epsilon)}\) remains representable by the same shallow ansatz family:
- one layer of \(R_Y\),
- followed by two brickwall \(CZ\) layers.

### 5. Blockwise solutions under prefix partitions remain shallow: Yes

For \(2\times 2\), \(4\times 4\), and \(8\times 8\) prefix partitions, each block solution is obtained by projecting the first \(k=1,2,3\) qubits in the computational basis.
Because the global solution stays within the cluster-state circuit family with only a local modification, the resulting block states remain in the same shallow real ansatz family, up to boundary byproduct corrections that can be absorbed into local angle/sign changes.

### 6. Non-singularity: Yes, for sufficiently small \(\beta,\epsilon\)

After conjugation, the operator becomes
\[
\alpha I + \beta \sum_i Z_i + \epsilon X_{12} Z_{13},
\]
which is block-diagonal in the computational basis except for a \(2\times 2\) mixing on qubit 12.
Thus \(A(\epsilon)\) is non-singular provided the chosen coefficients do not push one of these \(2\times 2\) eigenvalues to zero.
So non-singularity can be ensured by choosing sufficiently small \(\beta\) and \(\epsilon\) together with \(\alpha = 1-13\beta\).

---

## What fails

This setup does **not** satisfy the “all blocks nonzero” requirement under deeper prefix partitions.

Looking only at the first 3 qubits, the available \(X\)-flip patterns are:

- from \(I\): \(000\)
- from \(K_1 = X_1 Z_2\): \(100\)
- from \(K_2 = Z_1 X_2 Z_3\): \(010\)
- from \(K_3 = Z_2 X_3 Z_4\): \(001\)

All remaining stabilizers and \(X_{13}\) contribute pattern \(000\) on the first 3 qubits.

So the only available flip patterns on the first 3 qubits are
\[
\{000,\ 100,\ 010,\ 001\}.
\]

This implies:

- for the \(4\times 4\) partition (first 2 qubits as index), there are **4 zero blocks**
- for the \(8\times 8\) partition (first 3 qubits as index), there are **32 zero blocks**

So the setup is **not block-dense**, even though it still satisfies the other structural requirements.

---

## Final assessment

This benchmark is a valid choice if the goal is:

- to keep the construction physically natural,
- to use the full stabilizer description of the 13-qubit linear cluster state,
- to ensure \(x(0)=b\),
- to keep the exact solution inside the same shallow brickwall-\(CZ\)+\(R_Y\) variational family.

However, it is **not** suitable if one requires dense block structure under \(4\times 4\) and \(8\times 8\) prefix partitions.
