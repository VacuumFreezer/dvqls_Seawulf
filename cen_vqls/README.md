# Centralized VQLS (Ising 4-agent block system)

Run:

```bash
python cen_vqls/centralized_vqls_ising.py --config cen_vqls/config.yaml
```

Catalyst/JIT trial:

```bash
python cen_vqls/centralized_vqls_ising_qjit.py --config cen_vqls/config_qjit.yaml
```

Single-agent/global residual objective trial:

```bash
python cen_vqls/centralized_vqls_ising_residual.py --config cen_vqls/config_residual.yaml
```

`centralized_vqls_ising_residual.py` now uses the same qjit/JAX-JIT backend selection path as
`centralized_vqls_ising_qjit.py` via `runtime.use_qjit` and `runtime.fallback_to_jax_jit`.

Unified taskgen config (choose script in one file):

```bash
python gen_tasks_cen_vqls_qjit.py --base-config cen_vqls/config.yaml
```

To switch objective/script, edit `taskgen.optimization_script` in `cen_vqls/config.yaml`:

- `qjit_vqls` -> `cen_vqls/centralized_vqls_ising_qjit.py`
- `residual` -> `cen_vqls/centralized_vqls_ising_residual.py`

Random seed scan can also be set directly in YAML (`taskgen.seed_scan`):

- `mode: random` with `num_seeds`, `seed_min`, `seed_max`, `rng_seed`
- `mode: explicit` with `seeds: [..]`

What this script does:

1. Loads the distributed block system from `4agents_Ising/static_ops_16agents_Ising.py`.
2. Builds the centralized global matrix from
   \(A_{new}=(X_0+X_1+\sum_{k=2}^{8}X_k+\eta I+JZ_0Z_1+hZ_1Z_2)/\zeta\).
3. Verifies centralized matrix consistency with the block-partition matrix.
4. Constructs normalized `b` and its unitary representation from distributed `b`.
5. Optimizes centralized VQLS with Pennylane Adam (`autograd` + `backprop`) using Hadamard tests.
6. Computes unnormalized-solution L2 error each iteration.
7. Prints metrics every `print_every` steps (default: 20), writes report + history under `cen_vqls/reports/`.
8. `centralized_vqls_ising_residual.py` optimizes the expanded global residual objective
   `||Ax-b||^2 = sigma^2 <X|A^dagger A|X> + ||b||^2 - 2 sigma ||b|| Re(<B|A|X>)`
   using Hadamard-test terms (`beta`, `gamma_re`), and logs CG/CL in parallel for comparison.

The qjit script keeps the outer optimization loop in Python (first trial) and compiles the expensive Hadamard/state QNodes with Catalyst `qjit` when available. If Catalyst cannot be imported and `fallback_to_jax_jit: true`, it falls back to JAX-JIT and still writes a report including the backend used.

Main outputs per run:

- `report.md`: consistency checks, hyperparameters, final metrics.
- `history.json`: iteration history (loss + L2 block values).
- `checkpoints.json`: detailed Hadamard metrics at print intervals.
- `solution_comparison.txt`: final `theta`, final `sigma`, `x_true`, and recovered `x` vectors.
- `config_used.yaml`: resolved config snapshot.
