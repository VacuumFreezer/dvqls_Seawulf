# Qiskit 20-Qubit Hadamard Benchmark

This suite benchmarks the same 20-qubit Hadamard-test style circuit and full-parameter gradient evaluation using Qiskit.

- Environment: `vqls`
- No noise model
- Sampler benchmark includes `qiskit_aer.primitives.SamplerV2` with `shots=1024`
- Gradient methods: `SPSAEstimatorGradient` and `ReverseEstimatorGradient` from `qiskit_algorithms`
- CPU and GPU runs are separated through Slurm arrays

## Main scripts

- `benchmark_hadamard_qiskit.py`: runs one benchmark configuration and writes one JSON.
- `generate_tasks_qiskit.py`: generates the CPU/GPU task files for all combinations.
- `submit_array_qiskit_cpu.slurm`: CPU array launcher (`long-40core-shared`).
- `submit_array_qiskit_gpu.slurm`: GPU array launcher (`a100` by default; partition can be overridden at submit time).
- `capture_env_versions.py`: records package versions before/after runs.
- `generate_report_qiskit.py`: creates markdown + CSV summary report.
