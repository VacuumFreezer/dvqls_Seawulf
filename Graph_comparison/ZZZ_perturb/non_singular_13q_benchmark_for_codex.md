# Non-singular 13-qubit benchmark for distributed VQLS

This note specifies a **13-qubit benchmark linear system** to be implemented in code.

## Goal

Implement a real-valued benchmark linear system

\[
A(J)\,|x(J)\rangle = |b\rangle
\]

with the following properties:

1. The benchmark remains **very simple**.
2. At \(J=0\), the exact solution is exactly
   \[
   |x(0)\rangle = |b\rangle.
   \]
3. The matrix \(A(J)\) is **non-singular** for sufficiently small \(|J|\).
4. The benchmark uses only Pauli strings made of \(I\), \(X\), and \(Z\).
5. The state \(|b\rangle\) is a shallow product state:
   \[
   |b\rangle = |+\rangle^{\otimes 13}.
   \]

---

## Definition of the benchmark

Define the following Pauli-string operators:

\[
S_1 = X_1 X_2 X_3,\qquad
S_2 = X_4 X_5 X_6,\qquad
S_3 = X_7 X_8 X_9,\qquad
S_4 = X_{10} X_{11} X_{12},
\]

\[
T_1 = Z_1 Z_2 Z_3,\qquad
T_2 = Z_4 Z_5 Z_6,\qquad
T_3 = Z_7 Z_8 Z_9,\qquad
T_4 = Z_{10} Z_{11} Z_{12}.
\]

Also define the reference state

\[
|b\rangle = |+\rangle^{\otimes 13}.
\]

Then define the benchmark matrix

\[
A(J)
=
\frac{7}{2} I
-\frac{1}{2}\left(S_1+S_2+S_3+S_4+X_{13}\right)
+J\left(T_1+T_2+T_3+T_4\right).
\]

---

## Important properties

### 1. Exact solution at \(J=0\)

Because \(|+\rangle\) is a \(+1\) eigenstate of \(X\),

\[
S_i |b\rangle = |b\rangle,\qquad X_{13}|b\rangle = |b\rangle.
\]

Therefore,

\[
A(0)|b\rangle
=
\left(\frac{7}{2} - \frac{1}{2}(4+1)\right)|b\rangle
=
|b\rangle.
\]

So the exact solution at zero perturbation is

\[
|x(0)\rangle = |b\rangle.
\]

### 2. Non-singular range

For each 3-qubit block, the operator

\[
-\frac{1}{2}S_i + J T_i
\]

has eigenvalues

\[
\pm \sqrt{\frac{1}{4}+J^2}.
\]

This implies that the smallest eigenvalue of \(A(J)\) is

\[
\lambda_{\min}(A)
=
3 - 4\sqrt{\frac{1}{4}+J^2}.
\]

Hence \(A(J)\) is strictly positive definite, and therefore non-singular, whenever

\[
|J| < \frac{\sqrt{5}}{4} \approx 0.559.
\]

For actual simulations, use a comfortably small perturbation, for example:

- \(J = 0.05\)
- \(J = 0.1\)
- \(J = 0.2\)

### 3. LCU term count

The LCU decomposition has exactly **10 terms**:

- 1 identity term
- 4 \(XXX\) terms
- 1 \(X_{13}\) term
- 4 \(ZZZ\) terms

So this benchmark remains compact.

---

## State preparation

The reference state \(|b\rangle = |+\rangle^{\otimes 13}\) can be prepared with a single layer of \(R_Y\) gates:

\[
|b\rangle
=
\left(\prod_{j=1}^{13} R_Y^{(j)}(\pi/2)\right)|0\rangle^{\otimes 13}.
\]

This is a fully real shallow ansatz.

---

## Block partition warning

This benchmark is **not** designed to eliminate zero blocks under hierarchical prefix partitioning.

If the matrix is partitioned using the first qubits as index qubits, then the number of zero blocks is:

- for the \(4\times 4\) partition: **8 zero blocks**
- for the \(8\times 8\) partition: **48 zero blocks**

These counts are inherited from the limited \(X\)-flip patterns on the first 3 qubits.

So this benchmark should be viewed as a **simple non-singular baseline**, not as a block-dense benchmark.

---

## What Codex should implement

Please implement the following:

1. A function that constructs the matrix \(A(J)\) for a given \(J\).
2. A function that constructs the vector \(|b\rangle = |+\rangle^{\otimes 13}\).
3. A function that solves
   \[
   A(J)|x(J)\rangle = |b\rangle
   \]
   using dense linear algebra for small-scale verification.
4. A verification that at \(J=0\),
   \[
   |x(0)\rangle = |b\rangle
   \]
   up to numerical precision.
5. A verification that \(A(J)\) is non-singular for the chosen \(J\), for example by checking the smallest eigenvalue.
6. Optional: verify the reported zero-block counts for the \(4\times 4\) and \(8\times 8\) prefix partitions.

---

## Suggested outputs

For a chosen value of \(J\), the code should report:

- the smallest eigenvalue of \(A(J)\)
- whether \(A(J)\) is non-singular
- the norm of \(A(0)|b\rangle - |b\rangle\)
- the norm of \(A(J)|x(J)\rangle - |b\rangle\)
- optionally, the number of zero blocks under \(4\times 4\) and \(8\times 8\) prefix partitions

---

## Notes

- All vectors and matrices are real.
- The perturbation parameter is \(J\).
- This benchmark is intentionally simple and shallow.
- It is suitable as a sanity-check case before moving to more structured or block-dense benchmarks.
