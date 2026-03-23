#!/bin/bash
#SBATCH --job-name=graphcmp13_4x4
#SBATCH --partition=long-40core-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=Graph_comparison/run/slurm/%x_%A_%a.out
#SBATCH --error=Graph_comparison/run/slurm/%x_%A_%a.err

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

mkdir -p Graph_comparison/run/slurm

if [ -z "${TAG:-}" ]; then
  echo "Usage: sbatch --export=ALL,TAG=<suite_tag> Graph_comparison/submit_array.slurm"
  exit 1
fi

TASKS_FILE="Graph_comparison/run/${TAG}/tasks.txt"
if [ ! -f "$TASKS_FILE" ]; then
  echo "Tasks file not found: $TASKS_FILE"
  exit 1
fi

cmd=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$TASKS_FILE")

echo "========================================="
echo "JobID: $SLURM_JOB_ID  ArrayTask: $SLURM_ARRAY_TASK_ID"
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
