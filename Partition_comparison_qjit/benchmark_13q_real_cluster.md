# 13-Qubit Real Benchmark with Small LCU and Hierarchical Block Structure

## 1. Benchmark definition

We define a 13-qubit real benchmark built from a 1D cluster-state scaffold.

The reference vector is

\[
|b\rangle = U_b |0\rangle^{\otimes 13},
\]
with
\[
U_b=
\left(\prod_{j\,\mathrm{even}} CZ_{j,j+1}\right)
\left(\prod_{j\,\mathrm{odd}} CZ_{j,j+1}\right)
\left(\prod_{j=1}^{13} R_Y^{(j)}(\pi/2)\right).
\]

This is a fully real, shallow 13-qubit ansatz consisting of:
- one layer of single-qubit $R_Y$ rotations,
- one odd-bond CZ layer,
- one even-bond CZ layer.

We use four stabilizer terms acting on the first 12 qubits:
\[
K_1 = X_1 Z_2,
\]
\[
K_4 = Z_3 X_4 Z_5,
\]
\[
K_7 = Z_6 X_7 Z_8,
\]
\[
K_{10} = Z_9 X_{10} Z_{11}.
\]

We then define
\[
A(\epsilon)=\alpha I + \beta (K_1+K_4+K_7+K_{10}) + \epsilon Z_{13},
\]
where $\epsilon$ is a small real perturbation parameter.

The LCU term count is therefore
\[
1 + 4 + 1 = 6,
\]
which is very small.

---

## 2. Spectral normalization: $\lambda_{\max}=1$ and $\lambda_{\min}=1/20$

Because the four stabilizers commute with each other and with $Z_{13}$, the eigenvalues of $A(\epsilon)$ are
\[
\lambda = \alpha + \beta \sum_{m=1}^{4} s_m + \epsilon \xi,
\qquad s_m,\xi\in\{\pm 1\}.
\]
Hence
\[
\lambda_{\max}=\alpha + 4\beta + \epsilon,
\qquad
\lambda_{\min}=\alpha - 4\beta - \epsilon.
\]

To enforce
\[
\lambda_{\max}=1,
\qquad
\lambda_{\min}=\frac{1}{20},
\]
we choose
\[
\alpha = \frac{21}{40},
\qquad
\beta(\epsilon)=\frac{19}{160}-\frac{\epsilon}{4}.
\]
Indeed,
\[
\lambda_{\max}=\frac{21}{40}+4\left(\frac{19}{160}-\frac{\epsilon}{4}\right)+\epsilon=1,
\]
\[
\lambda_{\min}=\frac{21}{40}-4\left(\frac{19}{160}-\frac{\epsilon}{4}\right)-\epsilon=\frac{1}{20}.
\]

Thus the full matrix has condition number
\[
\kappa(A)=\frac{1}{1/20}=20.
\]

The perturbation remains small because
\[
A(\epsilon)-A(0)= -\frac{\epsilon}{4}(K_1+K_4+K_7+K_{10}) + \epsilon Z_{13},
\]
which is manifestly $O(\epsilon)$.

---

## 3. Why $x(0)=b$

At $\epsilon=0$, the reference state $|b\rangle$ is a $+1$ eigenstate of all four stabilizers,
\[
K_1|b\rangle = K_4|b\rangle = K_7|b\rangle = K_{10}|b\rangle = |b\rangle.
\]
Therefore
\[
A(0)|b\rangle = \bigl(\alpha+4\beta(0)\bigr)|b\rangle = |b\rangle,
\]
since
\[
\alpha+4\beta(0)=\frac{21}{40}+4\cdot\frac{19}{160}=1.
\]
Hence the exact solution of
\[
A(0)|x\rangle = |b\rangle
\]
is simply
\[
|x(0)\rangle = |b\rangle.
\]

---

## 4. Exact solution for small $\epsilon$

Conjugating by $U_b$ gives
\[
B(\epsilon):=U_b^{\dagger}A(\epsilon)U_b.
\]
Under this transformation,
\[
U_b^{\dagger}K_1U_b = Z_1,
\qquad
U_b^{\dagger}K_4U_b = Z_4,
\qquad
U_b^{\dagger}K_7U_b = Z_7,
\qquad
U_b^{\dagger}K_{10}U_b = Z_{10},
\]
and
\[
U_b^{\dagger} Z_{13} U_b = X_{13}.
\]
So
\[
B(\epsilon)=\alpha I + \beta(\epsilon)(Z_1+Z_4+Z_7+Z_{10}) + \epsilon X_{13}.
\]

On the reference product state $|0\rangle^{\otimes 13}$, the first four $Z$ operators contribute $+1$, so the effective action is
\[
B_{\mathrm{eff}}(\epsilon)=\bigl(\alpha+4\beta(\epsilon)\bigr)I + \epsilon X_{13} = (1-\epsilon)I + \epsilon X_{13}.
\]
Therefore
\[
B_{\mathrm{eff}}(\epsilon)^{-1}|0\rangle
\propto
|0\rangle - \frac{\epsilon}{1-\epsilon}|1\rangle
\propto
R_Y\!\left(-2\arctan\frac{\epsilon}{1-\epsilon}\right)|0\rangle.
\]

Hence the exact global solution is
\[
|x(\epsilon)\rangle
=
\left(\prod_{j\,\mathrm{even}} CZ_{j,j+1}\right)
\left(\prod_{j\,\mathrm{odd}} CZ_{j,j+1}\right)
\left[\prod_{j=1}^{12}R_Y^{(j)}(\pi/2)\right]
R_Y^{(13)}\!\left(\frac{\pi}{2}-2\arctan\frac{\epsilon}{1-\epsilon}\right)
|0\rangle^{\otimes 13}.
\]

So the exact solution is still represented by the same shallow real ansatz:
- one $R_Y$ layer,
- one odd-bond CZ layer,
- one even-bond CZ layer.

---

## 5. Block structure: 1x1, 2x2, 4x4, 8x8

We use the first $k$ qubits as the index register, with:
- $k=0$ for the global $1\times 1$ system,
- $k=1$ for the $2\times 2$ partition,
- $k=2$ for the $4\times 4$ partition,
- $k=3$ for the $8\times 8$ partition.

Each block of the exact solution is defined by projection onto a computational-basis index string,
\[
|x_s^{(k)}\rangle := (\langle s|\otimes I)|x(\epsilon)\rangle,
\qquad s\in\{0,1\}^k.
\]

Because the global state is a linear cluster-state scaffold, projecting the first $k$ qubits in the computational basis removes those vertices and leaves only a local $Z$ correction on the new boundary qubit. This means:
- each block is again a shorter linear-cluster-type state,
- the perturbation on qubit 13 remains the same,
- different blocks differ only by a boundary sign flip.

Since
\[
Z R_Y(\theta)|0\rangle = R_Y(-\theta)|0\rangle,
\]
that boundary $Z$ correction can be fully absorbed into a sign flip of one boundary $R_Y$ angle.

Therefore, for all of the following decompositions:
- $1\times 1$,
- $2\times 2$,
- $4\times 4$,
- $8\times 8$,

every exact block of the true solution is still proportional to a shallow real ansatz of the same type:
- one $R_Y$ layer,
- one odd-bond CZ layer,
- one even-bond CZ layer.

---

## 6. Final benchmark summary

A compact benchmark satisfying all desired properties is:

\[
|b\rangle
=
\left(\prod_{j\,\mathrm{even}} CZ_{j,j+1}\right)
\left(\prod_{j\,\mathrm{odd}} CZ_{j,j+1}\right)
\left(\prod_{j=1}^{13} R_Y^{(j)}(\pi/2)\right)|0\rangle^{\otimes 13},
\]

\[
A(\epsilon)=\frac{21}{40}I+
\left(\frac{19}{160}-\frac{\epsilon}{4}\right)(K_1+K_4+K_7+K_{10})+
\epsilon Z_{13},
\]
with
\[
K_1=X_1Z_2,
\qquad
K_4=Z_3X_4Z_5,
\qquad
K_7=Z_6X_7Z_8,
\qquad
K_{10}=Z_9X_{10}Z_{11}.
\]

This benchmark has the following properties:
- fully real matrix $A$ and vector $b$,
- only 6 LCU terms,
- a small perturbation parameter $\epsilon$,
- exact solution equal to $b$ at $\epsilon=0$,
- exact solution still representable by a 3-layer real ansatz for small $\epsilon$,
- every block of the exact solution under $1\times 1$, $2\times 2$, $4\times 4$, and $8\times 8$ prefix partitioning is also representable by the same shallow real ansatz family,
- exact spectral normalization with
\[
\lambda_{\max}(A)=1,
\qquad
\lambda_{\min}(A)=\frac{1}{20}.
\]
