# Partition Comparison qjit Benchmark Guide

This folder contains the 13-qubit partition-comparison benchmark workflow for Seawulf.

## Main files

- `Partition_comparison_qjit/config.yaml`
  Default hyperparameters and the list of partitions/cases to run.
- `Partition_comparison_qjit/gen_tasks.py`
  Builds a `tasks.txt` file for one run suite.
- `Partition_comparison_qjit/submit_array.slurm`
  Slurm array launcher used on Seawulf.
- `Partition_comparison_qjit/seawulf_partition_comparison_qjit.py`
  Main optimizer/benchmark driver.

## How to change hyperparameters

Edit `Partition_comparison_qjit/config.yaml`.

Most common fields:

- `python_bin`
  Python interpreter to use on Seawulf.
- `problems`
  Which partition definitions to include.
- `common.topology`
  Communication topology. For this benchmark we use `line`.
- `common.ansatz`
  Variational ansatz name.
- `common.layers`
  Number of ansatz layers.
- `common.epochs`
  Number of optimization iterations.
- `common.log_every`
  Logging frequency in iterations.
- `lr_sets`
  Learning-rate presets.
- `decay_sets`
  Learning-rate decay presets.
- `seeds`
  Random seeds to launch.

Example:

```yaml
common:
  epochs: 1000
  log_every: 5

lr_sets:
  - tag: lr5e-3
    lr: 0.005

seeds: [0, 1, 2]
```

## How to generate a task suite

From `/gpfs/home/tonshen/Seawulf_simulation`:

```bash
python Partition_comparison_qjit/gen_tasks.py \
  --config Partition_comparison_qjit/config.yaml \
  --tag my_run_tag
```

This creates:

- `Partition_comparison_qjit/run/my_run_tag/tasks.txt`
- one output directory per task under `Partition_comparison_qjit/run/my_run_tag/`

## How to submit on Seawulf

Submit all tasks in the generated suite:

```bash
sbatch --array=0-3 --export=ALL,TAG=my_run_tag Partition_comparison_qjit/submit_array.slurm
```

Submit only selected tasks:

```bash
sbatch --array=0,1 --export=ALL,TAG=my_run_tag Partition_comparison_qjit/submit_array.slurm
```

The `TAG` must match the folder created by `gen_tasks.py`.

## How to monitor

Check queue state:

```bash
squeue -j JOBID
```

Inspect scheduler details:

```bash
scontrol show job JOBID
```

Slurm logs:

- `Partition_comparison_qjit/run/slurm/partcmp13_<arrayjobid>_<taskid>.out`
- `Partition_comparison_qjit/run/slurm/partcmp13_<arrayjobid>_<taskid>.err`

Per-task benchmark logs:

- `Partition_comparison_qjit/run/<tag>/<task_dir>/stdout.log`
- `Partition_comparison_qjit/run/<tag>/<task_dir>/stderr.log`
- `Partition_comparison_qjit/run/<tag>/<task_dir>/metrics.jsonl`

## Notes

- The current Slurm file requests `16G` per task, which is enough for the current 13-qubit partition-comparison runs and avoids the per-user memory QoS bottleneck seen with `128G`.
- If you change the resource request in `submit_array.slurm`, regenerate and resubmit normally; no code change is needed in the optimizer.
