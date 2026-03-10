#!/bin/bash
#SBATCH --job-name=pl20_cpu_arr
#SBATCH --partition=long-40core-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-17%4
#SBATCH --output=GPU_Test/run/slurm/%x_%A_%a.out
#SBATCH --error=GPU_Test/run/slurm/%x_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [[ -z "${TAG:-}" ]]; then
  echo "Missing TAG environment variable."
  echo "Usage: sbatch --export=ALL,TAG=<suite_tag> GPU_Test/submit_array_cpu.slurm"
  exit 2
fi

TASKS_FILE="GPU_Test/run/${TAG}/tasks_cpu.txt"
if [[ ! -f "$TASKS_FILE" ]]; then
  echo "Tasks file not found: $TASKS_FILE"
  exit 2
fi

cmd=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS_FILE")
if [[ -z "${cmd}" ]]; then
  echo "No command at array index ${SLURM_ARRAY_TASK_ID} in $TASKS_FILE"
  exit 2
fi

module purge
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pennylane

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

log_dir="GPU_Test/run/${TAG}/task_logs/cpu"
mkdir -p "$log_dir"

echo "========================================="
echo "JobID: $SLURM_JOB_ID  ArrayTask: $SLURM_ARRAY_TASK_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Tasks file: $TASKS_FILE"
echo "CMD: $cmd"
echo "========================================="

eval "$cmd" >"${log_dir}/task_${SLURM_ARRAY_TASK_ID}.out" 2>"${log_dir}/task_${SLURM_ARRAY_TASK_ID}.err"

echo "End: $(date)"
