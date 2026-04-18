#!/bin/bash
#SBATCH --job-name=ising5qnec
#SBATCH --partition=long-40core-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=Necessarity_comparison/run/slurm/%x_%j.out
#SBATCH --error=Necessarity_comparison/run/slurm/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CONDA_PREFIX="/gpfs/home/tonshen/.conda/envs/pennylane"
export PATH="$CONDA_PREFIX/bin:/gpfs/software/Anaconda/bin:/gpfs/software/Anaconda/condabin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p Necessarity_comparison/run/slurm

if [ -z "${TASKS_FILE:-}" ]; then
  echo "Usage: sbatch --export=ALL,TASKS_FILE=<abs_tasks_path>,TASK_INDEX=<0-based-index> Necessarity_comparison/submit_array.slurm"
  exit 1
fi

if [ ! -f "$TASKS_FILE" ]; then
  echo "Tasks file not found: $TASKS_FILE"
  exit 1
fi

TASK_INDEX="${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"
if [ -z "${TASK_INDEX:-}" ]; then
  echo "TASK_INDEX is required unless this script is launched as a Slurm array job."
  exit 1
fi

cmd=$(sed -n "$((TASK_INDEX+1))p" "$TASKS_FILE")
if [ -z "$cmd" ]; then
  echo "No command found at TASK_INDEX=$TASK_INDEX in $TASKS_FILE"
  exit 1
fi

echo "========================================="
echo "JobID: $SLURM_JOB_ID  TaskIndex: $TASK_INDEX"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Tasks file: $TASKS_FILE"
echo "CMD: $cmd"
echo "========================================="

out_dir=$(echo "$cmd" | sed -n 's/.*--out \([^ ]*\).*/\1/p')
if [ -n "$out_dir" ]; then
  mkdir -p "$out_dir"
  eval "$cmd" >"$out_dir/stdout.log" 2>"$out_dir/stderr.log"
else
  eval "$cmd"
fi

echo "End: $(date)"
