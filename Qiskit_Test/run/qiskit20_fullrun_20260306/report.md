# Qiskit 20-Qubit Hadamard Test Benchmark Report

- Generated: 2026-03-06T20:37:42.200551Z
- Suite directory: `/gpfs/home/tonshen/Seawulf_simulation/Qiskit_Test/run/qiskit20_fullrun_20260306`
- Circuit: Hadamard test with shallow variational ansatz (total qubits=20, layers=2).
- SamplerV2 shots: 1024 (as requested, no noise model).

## Execution Summary

| Target | Total | OK | Unsupported | Failed | Missing Dependency |
|---|---:|---:|---:|---:|---:|
| cpu | 4 | 4 | 0 | 0 | 0 |
| gpu | 4 | 4 | 0 | 0 | 0 |

## Successful Configurations

| Target | Eval Mode | Gradient | Eval Mean (s) | Grad Mean (s) | Expectation | Grad L2 | GPU Mem MB |
|---|---|---|---:|---:|---:|---:|---:|
| cpu | estimator | reverse | 0.403277 | 12.213113 | -0.678764 | 0.481835 | 0.00 |
| cpu | estimator | spsa | 0.405961 | 0.817758 | -0.657304 | 1.491258 | 0.00 |
| cpu | sampler_v2 | reverse | 0.399775 | 12.119261 | -0.681641 | 0.492395 | 0.00 |
| cpu | sampler_v2 | spsa | 0.398807 | 0.808443 | -0.724609 | 3.501021 | 0.00 |
| gpu | estimator | reverse | 0.013027 | 9.687251 | -0.716014 | 0.441364 | 81152.00 |
| gpu | estimator | spsa | 0.012997 | 0.026149 | -0.657892 | 2.174017 | 81152.00 |
| gpu | sampler_v2 | reverse | 0.013654 | 9.917450 | -0.591797 | 0.518166 | 81152.00 |
| gpu | sampler_v2 | spsa | 0.014104 | 0.026440 | -0.732422 | 6.325845 | 81152.00 |

## Best Cases

| Target | Fastest Eval | Eval Mean (s) | Fastest Gradient | Grad Mean (s) |
|---|---|---:|---|---:|
| cpu | sampler_v2 + spsa | 0.398807 | sampler_v2 + spsa | 0.808443 |
| gpu | estimator + spsa | 0.012997 | estimator + spsa | 0.026149 |

## Version Check

- cuquantum-cu11: missing -> 24.8.0
- numpy: 2.0.2 (unchanged)
- qiskit: 2.2.3 (unchanged)
- qiskit-aer-gpu: missing -> MISSING: PackageNotFoundError: qiskit-aer-gpu
- qiskit-aer-gpu-cu11: missing -> 0.17.2
- qiskit_aer: 0.17.2 (unchanged)
- qiskit_algorithms: 0.4.0 (unchanged)

## Commentary

- Median CPU/GPU eval speedup (CPU_time / GPU_time): 30.118.
- Median CPU/GPU gradient speedup (CPU_time / GPU_time): 15.919.
- SamplerV2 results include shot noise (shots=1024), while estimator-based values are deterministic in this setup.
