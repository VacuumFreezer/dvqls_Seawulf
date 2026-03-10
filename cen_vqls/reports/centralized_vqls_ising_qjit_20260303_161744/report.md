# Centralized VQLS Report

- Timestamp: 2026-03-03T16:18:42
- Static ops path: `/gpfs/home/tonshen/Seawulf_simulation/problems/static_ops_16agents_Ising_forcen.py`
- Loaded problem source: `distributed_system[4x4]`
- Total qubits: 4
- Agents: 4 (index qubits=2, local qubits=2)

## Matrix Consistency
- Formula vs block max abs diff: 1.110223e-16
- Formula vs block Frobenius diff: 2.830524e-16
- allclose(atol=1e-12, rtol=0): True
- Condition number of A (2-norm): 2.247957191452e+00

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
- ||b_normed - |+>^n||: 0.000000e+00
- ||U_b|0> - b_normed|| (phase-aligned): 3.330669e-16

## Optimization Summary
- Optimize metric: global_CG
- Steps: 1
- Learning rate: 0.001
- Adam betas: (0.9, 0.999)
- Adam eps: 1e-08
- Best iteration: 1
- Best global_CG: 9.999822266198e-01

## Final Metrics
- global_CG: 9.999822266198e-01
- global_CL: nan
- global_CG_hat: 2.162751536405e-01
- global_CL_hat: nan
- beta: 2.162789976494e-01
- overlap_sq: 3.844008855924e-06
- Residual on linear system (using reconstructed global x):
  - ||A x_est - b||_2 (raw x_est): 1.133753185807e+01
  - ||A x_est - b||_2 / ||b||_2 (raw x_est): 1.417191482259e+00
  - ||A x_est - b||_2 (phase-aligned x_est): 1.128983486891e+01
  - ||A x_est - b||_2 / ||b||_2 (phase-aligned x_est): 1.411229358614e+00
- Residual to exact solution (using reconstructed global x):
  - ||x_est - x_true||_2 (raw): 2.062701995464e+01
  - ||x_est - x_true||_2 / ||x_true||_2 (raw): 1.818749736484e+00
  - ||x_est - x_true||_2 (phase-aligned): 2.058167172026e+01
  - ||x_est - x_true||_2 / ||x_true||_2 (phase-aligned): 1.814751238906e+00
- phase alignment angle (rad): 3.141592653590e+00

## Timing
- First output time: 2026-03-03T16:18:20.669
- Last output time: 2026-03-03T16:18:20.669
- Total elapsed seconds: 15.500757

## Files
- History: `/gpfs/home/tonshen/Seawulf_simulation/cen_vqls/reports/centralized_vqls_ising_qjit_20260303_161744/history.json`
- Checkpoints: `/gpfs/home/tonshen/Seawulf_simulation/cen_vqls/reports/centralized_vqls_ising_qjit_20260303_161744/checkpoints.json`
- Config used: `/gpfs/home/tonshen/Seawulf_simulation/cen_vqls/reports/centralized_vqls_ising_qjit_20260303_161744/config_used.yaml`
- solution_txt: `/gpfs/home/tonshen/Seawulf_simulation/cen_vqls/reports/centralized_vqls_ising_qjit_20260303_161744/solution_comparison.txt`

## Solution Preview
### x_true
```text
size=16, showing first 16 entries
[   0] +2.755231295384e+00 +0.000000000000e+00j
[   1] +2.806919547254e+00 +0.000000000000e+00j
[   2] +2.861264553010e+00 +0.000000000000e+00j
[   3] +2.807562374119e+00 +0.000000000000e+00j
[   4] +2.861264553010e+00 +0.000000000000e+00j
[   5] +2.918423672208e+00 +0.000000000000e+00j
[   6] +2.861919825699e+00 +0.000000000000e+00j
[   7] +2.806919547254e+00 +0.000000000000e+00j
[   8] +2.806919547254e+00 +0.000000000000e+00j
[   9] +2.861919825699e+00 +0.000000000000e+00j
[  10] +2.918423672208e+00 +0.000000000000e+00j
[  11] +2.861264553010e+00 +0.000000000000e+00j
[  12] +2.807562374119e+00 +0.000000000000e+00j
[  13] +2.861264553010e+00 +0.000000000000e+00j
[  14] +2.806919547254e+00 +0.000000000000e+00j
[  15] +2.755231295384e+00 +0.000000000000e+00j
```
### x_est_unnorm_raw
```text
size=16, showing first 16 entries
[   0] +1.591454634818e+00 +0.000000000000e+00j
[   1] -3.653556645644e+00 +0.000000000000e+00j
[   2] -8.136023809133e+00 +0.000000000000e+00j
[   3] +1.642278486377e+00 +0.000000000000e+00j
[   4] -1.768467390716e+00 +0.000000000000e+00j
[   5] +7.555478227384e+00 +0.000000000000e+00j
[   6] +3.934287541090e+00 +0.000000000000e+00j
[   7] -1.477896467035e+00 +0.000000000000e+00j
[   8] +1.083358888050e+00 +0.000000000000e+00j
[   9] -5.367076156770e+00 +0.000000000000e+00j
[  10] -5.538476255732e+00 +0.000000000000e+00j
[  11] +2.412507745711e+00 +0.000000000000e+00j
[  12] -2.240363373280e+00 +0.000000000000e+00j
[  13] +5.964040041785e+00 +0.000000000000e+00j
[  14] +4.984108699592e+00 +0.000000000000e+00j
[  15] -1.166601695054e+00 +0.000000000000e+00j
```
### x_est_unnorm_aligned
```text
size=16, showing first 16 entries
[   0] -1.591454634818e+00 -1.948969824518e-16j
[   1] +3.653556645644e+00 +4.474316451592e-16j
[   2] +8.136023809133e+00 +9.963755515641e-16j
[   3] -1.642278486377e+00 -2.011211091650e-16j
[   4] +1.768467390716e+00 +2.165747929436e-16j
[   5] -7.555478227384e+00 -9.252792227194e-16j
[   6] -3.934287541090e+00 -4.818112644122e-16j
[   7] +1.477896467035e+00 +1.809901177826e-16j
[   8] -1.083358888050e+00 -1.326731994578e-16j
[   9] +5.367076156770e+00 +6.572772636169e-16j
[  10] +5.538476255732e+00 +6.782677218736e-16j
[  11] -2.412507745711e+00 -2.954469888704e-16j
[  12] +2.240363373280e+00 +2.743653834015e-16j
[  13] -5.964040041785e+00 -7.303842547158e-16j
[  14] -4.984108699592e+00 -6.103772765558e-16j
[  15] +1.166601695054e+00 +1.428675031728e-16j
```
### final_theta
```text
[[-1.4968351583 -1.2651175766  1.9753089601 -2.5650673301]]
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
  grad_mode: jax_jit
  qnode_compile: none
  jit_training_step: true
  warmup_compile: true
  log_local_cost: false
report:
  out_dir: cen_vqls/reports
  tag: centralized_vqls_ising_qjit
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
        grad_mode: jax_jit
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
- Requested grad mode: jax_jit
- Resolved grad mode: jax_jit
- Requested qnode compile: none
- Resolved qnode compile: none
- Catalyst available: True
- qjit_autograph: False
- jit_training_step: True
- log_local_cost: False
