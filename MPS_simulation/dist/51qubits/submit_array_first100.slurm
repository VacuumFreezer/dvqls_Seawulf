#!/bin/bash
#SBATCH --job-name=mps_51q_first100
#SBATCH --partition=long-40core-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=MPS_simulation/dist/51qubits/slurm/%x_%A_%a.out
#SBATCH --error=MPS_simulation/dist/51qubits/slurm/%x_%A_%a.err

set -euo pipefail

WORKDIR="/gpfs/home/tonshen/Seawulf_simulation"
BENCH_DIR="/gpfs/home/tonshen/Seawulf_simulation/MPS_simulation/dist/51qubits"

cd "$WORKDIR"
mkdir -p "$BENCH_DIR/slurm"

module purge
module load anaconda/3-new
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate MPS
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

TASKS_FILE="$BENCH_DIR/run/${TAG}/tasks.txt"
cmd=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$TASKS_FILE")

echo "========================================="
echo "JobID: $SLURM_JOB_ID  ArrayTask: $SLURM_ARRAY_TASK_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Tasks file: $TASKS_FILE"
echo "CMD: $cmd"
echo "========================================="

out_json=$(echo "$cmd" | sed -n 's/.*--out-json \([^ ]*\).*/\1/p')
if [ -n "$out_json" ]; then
  out_dir=$(dirname "$out_json")
  log_base="${out_json%.json}"
  mkdir -p "$out_dir"
  eval "$cmd" >"${log_base}.stdout.log" 2>"${log_base}.stderr.log"
else
  eval "$cmd"
fi

echo "End: $(date)"
