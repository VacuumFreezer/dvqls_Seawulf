#!/bin/bash
#SBATCH --job-name=dvqls_arr
#SBATCH --partition=long-40core-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=run/slurm/%x_%A_%a.out
#SBATCH --error=run/slurm/%x_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pennylane
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# incase numpy/mkl/pennylane-lightning occupying full nodes
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

TASKS_FILE="run/${TAG}/tasks.txt"
cmd=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$TASKS_FILE")

echo "========================================="
echo "JobID: $SLURM_JOB_ID  ArrayTask: $SLURM_ARRAY_TASK_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Tasks file: $TASKS_FILE"
echo "CMD: $cmd"
echo "========================================="

# Write stdout/stderr for all task into its own out_dir
out_dir=$(echo "$cmd" | sed -n 's/.*--out \([^ ]*\).*/\1/p')
if [ -n "$out_dir" ]; then
  mkdir -p "$out_dir"
  eval "$cmd" >"$out_dir/stdout.log" 2>"$out_dir/stderr.log"
else
  eval "$cmd"
fi

echo "End: $(date)"
