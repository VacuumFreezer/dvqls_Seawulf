#!/bin/bash
#SBATCH --job-name=formal_single_agent_mps
#SBATCH --partition=long-40core-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=MPS_simulation/dist/10qubits/formal_single_agent_compare/slurm/%x_%A_%a.out
#SBATCH --error=MPS_simulation/dist/10qubits/formal_single_agent_compare/slurm/%x_%A_%a.err

set -euo pipefail
WORKDIR="/gpfs/home/tonshen/Seawulf_simulation"
COMPARE_DIR="/gpfs/home/tonshen/Seawulf_simulation/MPS_simulation/dist/10qubits/formal_single_agent_compare"

cd "$WORKDIR"

mkdir -p "$COMPARE_DIR/slurm"

PYTHON_BIN="/gpfs/home/tonshen/.conda/envs/MPS/bin/python"
export PATH="/gpfs/home/tonshen/.conda/envs/MPS/bin:${PATH}"
export LD_LIBRARY_PATH="/gpfs/home/tonshen/.conda/envs/MPS/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

TASKS_FILE="$COMPARE_DIR/runs/${TAG}/tasks.txt"
cmd=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$TASKS_FILE")

echo "========================================="
echo "JobID: $SLURM_JOB_ID  ArrayTask: $SLURM_ARRAY_TASK_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Tasks file: $TASKS_FILE"
echo "CMD: $cmd"
echo "========================================="

out_dir=$(echo "$cmd" | sed -n 's/.*--out-dir \([^ ]*\).*/\1/p')
if [ -n "$out_dir" ]; then
  mkdir -p "$out_dir"
  eval "$cmd" >"$out_dir/stdout.log" 2>"$out_dir/stderr.log"
else
  eval "$cmd"
fi

echo "End: $(date)"
