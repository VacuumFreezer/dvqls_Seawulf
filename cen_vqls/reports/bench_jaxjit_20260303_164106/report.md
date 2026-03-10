# Centralized VQLS Report

- Timestamp: 2026-03-03T16:53:16
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
- Steps: 2
- Learning rate: 0.001
- Adam betas: (0.9, 0.999)
- Adam eps: 1e-08
- Best iteration: 2
- Best global_CG: 9.999999981661e-01

## Final Metrics
- global_CG: 9.999999981661e-01
- global_CL: nan
- global_CG_hat: 2.478783498356e-01
- global_CL_hat: nan
- beta: 2.478783502901e-01
- overlap_sq: 4.545718448319e-10
- Residual on linear system (using reconstructed global x):
  - ||A x_est - b||_2 (raw x_est): 1.131395074258e+01
  - ||A x_est - b||_2 / ||b||_2 (raw x_est): 1.414243842823e+00
  - ||A x_est - b||_2 (phase-aligned x_est): 1.131346625020e+01
  - ||A x_est - b||_2 / ||b||_2 (phase-aligned x_est): 1.414183281275e+00
- Residual to exact solution (using reconstructed global x):
  - ||x_est - x_true||_2 (raw): 1.795486315009e+01
  - ||x_est - x_true||_2 / ||x_true||_2 (raw): 2.241186813462e+00
  - ||x_est - x_true||_2 (phase-aligned): 1.795460087159e+01
  - ||x_est - x_true||_2 / ||x_true||_2 (phase-aligned): 2.241154074972e+00
- phase alignment angle (rad): 3.141592653590e+00

## Timing
- First output time: 2026-03-03T16:49:14.868
- Last output time: 2026-03-03T16:51:33.069
- Total elapsed seconds: 223.598693

## Files
- History: `/gpfs/home/tonshen/Seawulf_simulation/cen_vqls/reports/bench_jaxjit_20260303_164106/history.json`
- Checkpoints: `/gpfs/home/tonshen/Seawulf_simulation/cen_vqls/reports/bench_jaxjit_20260303_164106/checkpoints.json`
- Config used: `/gpfs/home/tonshen/Seawulf_simulation/cen_vqls/reports/bench_jaxjit_20260303_164106/config_used.yaml`
- solution_txt: `/gpfs/home/tonshen/Seawulf_simulation/cen_vqls/reports/bench_jaxjit_20260303_164106/solution_comparison.txt`

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
[   0] +6.841125253400e-02 +0.000000000000e+00j
[   1] +2.509899191753e-02 +0.000000000000e+00j
[   2] -3.147638129960e-02 +0.000000000000e+00j
[   3] -3.970095279567e-02 +0.000000000000e+00j
[   4] +3.339308441640e-01 +0.000000000000e+00j
[   5] +1.225138737889e-01 +0.000000000000e+00j
[   6] -2.111114965383e-01 +0.000000000000e+00j
[   7] -2.662735426577e-01 +0.000000000000e+00j
[   8] +5.848041871271e-01 +0.000000000000e+00j
[   9] +2.145552818047e-01 +0.000000000000e+00j
[  10] -2.690715181755e-01 +0.000000000000e+00j
[  11] -3.393781368979e-01 +0.000000000000e+00j
[  12] +8.719335460861e-02 +0.000000000000e+00j
[  13] +3.198984409029e-02 +0.000000000000e+00j
[  14] -5.512374763013e-02 +0.000000000000e+00j
[  15] -6.952722048173e-02 +0.000000000000e+00j
...
```
### x_est_unnorm_aligned
```text
size=1024, showing first 16 entries
[   0] -6.841125253400e-02 -8.377962144143e-18j
[   1] -2.509899191753e-02 -3.073740011363e-18j
[   2] +3.147638129960e-02 +3.854744960729e-18j
[   3] +3.970095279567e-02 +4.861964476432e-18j
[   4] -3.339308441640e-01 -4.089473394421e-17j
[   5] -1.225138737889e-01 -1.500362233867e-17j
[   6] +2.111114965383e-01 +2.585370184988e-17j
[   7] +2.662735426577e-01 +3.260910417134e-17j
[   8] -5.848041871271e-01 -7.161785758932e-17j
[   9] -2.145552818047e-01 -2.627544391022e-17j
[  10] +2.690715181755e-01 +3.295175734754e-17j
[  11] +3.393781368979e-01 +4.156183490526e-17j
[  12] -8.719335460861e-02 -1.067810626283e-17j
[  13] -3.198984409029e-02 -3.917626017039e-18j
[  14] +5.512374763013e-02 +6.750712109224e-18j
[  15] +6.952722048173e-02 +8.514628801656e-18j
...
```
### final_theta
```text
[[-1.4961979231 -1.2650557862  1.9743078306 -2.5644789629  0.6290577476
   1.4359099151 -1.9618940304 -2.7955437067 -1.4125843459  0.9892320804]]
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
  steps: 2
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
  tag: bench_jaxjit
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
