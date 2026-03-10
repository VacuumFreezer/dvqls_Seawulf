# Centralized VQLS Report

- Timestamp: 2026-03-03T17:03:32
- Static ops path: `/gpfs/home/tonshen/Seawulf_simulation/problems/static_ops_16agents_Ising_forcen.py`
- Loaded problem source: `distributed_system[4x4]`
- Total qubits: 10
- Agents: 4 (index qubits=2, local qubits=8)

## Matrix Consistency
- Formula vs block max abs diff: 1.110223e-16
- Formula vs block Frobenius diff: 2.195324e-15
- allclose(atol=1e-12, rtol=0): True
- Condition number of A (2-norm): 5.019156051503e+01

## Equation Match vs Distributed Reference
- Reference system key: `4x4`
- A_target vs A_ref max abs diff: 0.000000e+00
- A_target vs A_ref Frobenius diff: 0.000000e+00
- A_target vs A_ref allclose(atol=1e-12, rtol=0): True
- b_target vs b_ref max abs diff: 0.000000e+00
- b_target vs b_ref L2 diff: 0.000000e+00
- b_target vs b_ref allclose(atol=1e-12, rtol=0): True

## b-Vector and Unitary
- ||b_unnorm||: 8.000000000000
- b unitary mode: hadamard_tensor
- ||b_normed - |+>^n||: 4.440892e-16
- ||U_b|0> - b_normed|| (phase-aligned): 3.330669e-16

## Optimization Summary
- Optimize metric: global_CG
- Steps: 1
- Learning rate: 0.001
- Adam betas: (0.9, 0.999)
- Adam eps: 1e-08
- Best iteration: 1
- Best global_CG: 9.999999982296e-01

## Final Metrics
- global_CG: 9.999999982296e-01
- global_CL: nan
- global_CG_hat: 2.478840020071e-01
- global_CL_hat: nan
- beta: 2.478840024459e-01
- overlap_sq: 4.388649155141e-10
- Residual on linear system (using reconstructed global x):
  - ||A x_est - b||_2 (raw x_est): 1.131394651797e+01
  - ||A x_est - b||_2 / ||b||_2 (raw x_est): 1.414243314746e+00
  - ||A x_est - b||_2 (phase-aligned x_est): 1.131347047499e+01
  - ||A x_est - b||_2 / ||b||_2 (phase-aligned x_est): 1.414183809374e+00
- Residual to exact solution (using reconstructed global x):
  - ||x_est - x_true||_2 (raw): 1.795469665190e+01
  - ||x_est - x_true||_2 / ||x_true||_2 (raw): 2.241166030594e+00
  - ||x_est - x_true||_2 (phase-aligned): 1.795443947840e+01
  - ||x_est - x_true||_2 / ||x_true||_2 (phase-aligned): 2.241133929328e+00
- phase alignment angle (rad): 3.141592653590e+00

## Timing
- First output time: 2026-03-03T17:01:42.389
- Last output time: 2026-03-03T17:01:42.389
- Total elapsed seconds: 0.199119

## Files
- History: `/gpfs/home/tonshen/Seawulf_simulation/cen_vqls/reports/smoke_tracking_defaults_20260303_165740/history.json`
- Checkpoints: `/gpfs/home/tonshen/Seawulf_simulation/cen_vqls/reports/smoke_tracking_defaults_20260303_165740/checkpoints.json`
- Config used: `/gpfs/home/tonshen/Seawulf_simulation/cen_vqls/reports/smoke_tracking_defaults_20260303_165740/config_used.yaml`
- solution_txt: `/gpfs/home/tonshen/Seawulf_simulation/cen_vqls/reports/smoke_tracking_defaults_20260303_165740/solution_comparison.txt`

## Solution Preview
### x_true
```text
size=1024, showing first 16 entries
[   0] +2.373743325392e-01 +0.000000000000e+00j
[   1] +2.400789603672e-01 +0.000000000000e+00j
[   2] +2.428540384341e-01 +0.000000000000e+00j
[   3] +2.400878129660e-01 +0.000000000000e+00j
[   4] +2.428636559937e-01 +0.000000000000e+00j
[   5] +2.457120427694e-01 +0.000000000000e+00j
[   6] +2.428725999641e-01 +0.000000000000e+00j
[   7] +2.400881063693e-01 +0.000000000000e+00j
[   8] +2.428638551476e-01 +0.000000000000e+00j
[   9] +2.457321003904e-01 +0.000000000000e+00j
[  10] +2.486582444579e-01 +0.000000000000e+00j
[  11] +2.457226859244e-01 +0.000000000000e+00j
[  12] +2.428824168221e-01 +0.000000000000e+00j
[  13] +2.457324099197e-01 +0.000000000000e+00j
[  14] +2.428731013554e-01 +0.000000000000e+00j
[  15] +2.400879995702e-01 +0.000000000000e+00j
...
```
### x_est_unnorm_raw
```text
size=1024, showing first 16 entries
[   0] +6.844084256654e-02 +0.000000000000e+00j
[   1] +2.514376909627e-02 +0.000000000000e+00j
[   2] -3.151021064032e-02 +0.000000000000e+00j
[   3] -3.979978630757e-02 +0.000000000000e+00j
[   4] +3.340780782030e-01 +0.000000000000e+00j
[   5] +1.227334694236e-01 +0.000000000000e+00j
[   6] -2.110559587820e-01 +0.000000000000e+00j
[   7] -2.665796860055e-01 +0.000000000000e+00j
[   8] +5.843766359235e-01 +0.000000000000e+00j
[   9] +2.146880524539e-01 +0.000000000000e+00j
[  10] -2.690474021170e-01 +0.000000000000e+00j
[  11] -3.398272779924e-01 +0.000000000000e+00j
[  12] +8.724620236971e-02 +0.000000000000e+00j
[  13] +3.205247458457e-02 +0.000000000000e+00j
[  14] -5.511834535889e-02 +0.000000000000e+00j
[  15] -6.961865129851e-02 +0.000000000000e+00j
...
```
### x_est_unnorm_aligned
```text
size=1024, showing first 16 entries
[   0] -6.844084256654e-02 -8.381585878007e-18j
[   1] -2.514376909627e-02 -3.079223634225e-18j
[   2] +3.151021064032e-02 +3.858887860113e-18j
[   3] +3.979978630757e-02 +4.874068090832e-18j
[   4] -3.340780782030e-01 -4.091276491366e-17j
[   5] -1.227334694236e-01 -1.503051504778e-17j
[   6] +2.110559587820e-01 +2.584690043634e-17j
[   7] +2.665796860055e-01 +3.264659591844e-17j
[   8] -5.843766359235e-01 -7.156549766802e-17j
[   9] -2.146880524539e-01 -2.629170362529e-17j
[  10] +2.690474021170e-01 +3.294880398216e-17j
[  11] +3.398272779924e-01 +4.161683882563e-17j
[  12] -8.724620236971e-02 -1.068457824698e-17j
[  13] -3.205247458457e-02 -3.925296040475e-18j
[  14] +5.511834535889e-02 +6.750050521807e-18j
[  15] +6.961865129851e-02 +8.525825847367e-18j
...
```
### final_theta
```text
[[-1.4970176984 -1.2655889557  1.9743234902 -2.5642717804  0.6290034055
   1.4359993802 -1.9614323387 -2.7953182645 -1.4132488079  0.9892060999]]
```

## Hyperparameters
```yaml
problem:
  static_ops_path: problems/static_ops_16agents_Ising_forcen.py
  system_key: 4x4
  prefer_centralized_problem: true
  centralized_problem_key: centralized
  consistency_system_key: 4x4
  consistency_atol: 1.0e-12
  b_consistency_atol: 1.0e-12
  b_state_tolerance: 1.0e-10
ansatz:
  layers: 1
  init_low: -3.141592653589793
  init_high: 3.141592653589793
  init_sigma: 1.0
optimization:
  steps: 1
  learning_rate: 0.001
  beta1: 0.9
  beta2: 0.999
  eps: 1.0e-08
  print_every: 1
  optimize_metric: global_CG
runtime:
  seed: 2
  device: lightning.qubit
  interface: jax
  diff_method: adjoint
  use_qjit: true
  fallback_to_jax_jit: false
  qjit_autograph: false
  grad_mode: qjit
  qnode_compile: none
  jit_training_step: true
  warmup_compile: true
  log_local_cost: false
report:
  out_dir: cen_vqls/reports
  tag: smoke_tracking_defaults
taskgen:
  optimization_script: tracking_vqls
  tag: my_cen_vqls_tasks
  seed_scan:
    mode: explicit
    num_seeds: 5
    seed_min: 0
    seed_max: 1000
    rng_seed: null
    seeds:
    - 2
    - 3
    - 7
    - 42
    - 54
    - 532
    - 564
    - 601
    - 1234
    - 4321
    - 6026
    - 9312
  script_map:
    qjit_vqls: cen_vqls/centralized_vqls_ising_qjit.py
    tracking_vqls: cen_vqls/centralized_vqls_ising_tracking.py
    single: cen_vqls/single_agent_qjit.py
  profile_overrides:
    qjit_vqls:
      optimization:
        optimize_metric: global_CG
      runtime:
        interface: jax
        device: lightning.qubit
        diff_method: adjoint
        use_qjit: true
        fallback_to_jax_jit: false
      report:
        tag: centralized_vqls_ising_qjit
    tracking_vqls:
      optimization:
        optimize_metric: global_CG
      runtime:
        interface: jax
        device: lightning.qubit
        diff_method: adjoint
        use_qjit: true
        fallback_to_jax_jit: true
        grad_mode: qjit
        qnode_compile: none
        qjit_autograph: false
        jit_training_step: false
        warmup_compile: true
        log_local_cost: false
      report:
        tag: centralized_vqls_ising_tracking
    single:
      ansatz:
        init_sigma: 1.0
      optimization:
        steps: 20000
        learning_rate: 0.001
        print_every: 20
        optimize_metric: global_residual_sq
      runtime:
        interface: jax
        device: lightning.qubit
        diff_method: adjoint
        use_qjit: true
        fallback_to_jax_jit: true
        qjit_autograph: false
        jit_training_step: true
        warmup_compile: true
      report:
        tag: centralized_vqls_ising_residual
```

## Compile Strategy
- Requested grad mode: qjit
- Resolved grad mode: qjit
- Requested qnode compile: none
- Resolved qnode compile: none
- Catalyst available: True
- qjit_autograph: False
- jit_training_step: True
- log_local_cost: False
