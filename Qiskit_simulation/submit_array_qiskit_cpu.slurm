#!/bin/bash
#SBATCH --job-name=qk_cluster30_arr
#SBATCH --partition=long-40core-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-0
#SBATCH --output=Qiskit_simulation/run/slurm/%x_%A_%a.out
#SBATCH --error=Qiskit_simulation/run/slurm/%x_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [[ -z "${TAG:-}" ]]; then
  echo "Missing TAG environment variable."
  echo "Usage: sbatch --export=ALL,TAG=<suite_tag> Qiskit_simulation/submit_array_qiskit_cpu.slurm"
  exit 2
fi

TASKS_FILE="Qiskit_simulation/run/${TAG}/tasks.txt"
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
conda activate vqls

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

out_dir=$(echo "$cmd" | sed -n 's/.*--out \([^ ]*\).*/\1/p')
if [[ -n "$out_dir" ]]; then
  mkdir -p "$out_dir"
fi
mkdir -p "Qiskit_simulation/run/slurm"

echo "========================================="
echo "JobID: $SLURM_JOB_ID  ArrayTask: $SLURM_ARRAY_TASK_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Tasks file: $TASKS_FILE"
echo "CMD: $cmd"
echo "========================================="

if [[ -n "$out_dir" ]]; then
  eval "$cmd" >"$out_dir/stdout.log" 2>"$out_dir/stderr.log"
else
  eval "$cmd"
fi

echo "End: $(date)"
