# PennyLane 20-Qubit Hadamard Benchmark (CPU + GPU)

This folder benchmarks a **20-qubit Hadamard test** circuit in PennyLane using a shallow variational ansatz.

- Total qubits: 20 (1 ancilla + 19 system qubits)
- Ansatz: `StronglyEntanglingLayers` with configurable shallow depth (`--layers`, default 2)
- Timings collected:
  - expectation-value evaluation time
  - full gradient time over all ansatz parameters
- Sweep dimensions:
  - device: `default.qubit`, `lightning.qubit`
  - interface: `numpy`, `torch`, `jax`
  - diff method: `backprop`, `adjoint`, `finite_diff`
  - target: CPU and GPU runs

The sweep intentionally includes combinations that may be unsupported by PennyLane; those runs are captured as `unsupported` in JSON and included in the final report.

## Files

- `benchmark_hadamard_test.py`: run one configuration and write JSON result.
- `generate_tasks.py`: generate full CPU/GPU task lists and suite metadata.
- `submit_array_cpu.slurm`: CPU array job on `long-40core-shared`.
- `submit_array_gpu.slurm`: GPU array job on `a100`.
- `submit_suite.sh`: convenience wrapper to generate tasks and submit both arrays.
- `generate_report.py`: aggregate all JSON outputs into markdown and CSV report.

## Usage

From repository root (`/gpfs/home/tonshen/Seawulf_simulation`):

```bash
module purge
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pennylane
```

Generate a sweep suite (18 CPU tasks + 18 GPU tasks):

```bash
python GPU_Test/generate_tasks.py --num-qubits 20 --layers 2 --eval-repeats 3 --grad-repeats 1
```

The command prints:

- `TAG=<suite_tag>`
- `SUITE_DIR=...`

Submit arrays:

```bash
sbatch --export=ALL,TAG=<suite_tag> GPU_Test/submit_array_cpu.slurm
sbatch --export=ALL,TAG=<suite_tag> GPU_Test/submit_array_gpu.slurm
```

Or submit both with one command:

```bash
bash GPU_Test/submit_suite.sh --num-qubits 20 --layers 2 --eval-repeats 3 --grad-repeats 1
```

After jobs finish, generate the comprehensive report:

```bash
python GPU_Test/generate_report.py --suite-dir GPU_Test/run/<suite_tag>
```

Outputs:

- `GPU_Test/run/<suite_tag>/report.md`
- `GPU_Test/run/<suite_tag>/summary.csv`

## Notes

- The Slurm scripts activate the `pennylane` conda environment.
- CPU partition is set to `long-40core-shared` and GPU partition to `a100` as requested.
- For the `numpy` sweep entry, the QNode uses PennyLane's `autograd` interface internally so full gradients can be timed.
- If a combination is unsupported on your installed PennyLane version/plugins, that result is still recorded and explained in the report.
