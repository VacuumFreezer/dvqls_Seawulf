#!/bin/bash
#SBATCH --job-name=qk30_60p_spsa_gpu
#SBATCH --partition=a100,a100-long,a100-large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=Qiskit_Test/run/qiskit30q_60p_spsa_gpu_retry_20260306/slurm_logs/%x_%j.out
#SBATCH --error=Qiskit_Test/run/qiskit30q_60p_spsa_gpu_retry_20260306/slurm_logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p Qiskit_Test/run/qiskit30q_60p_spsa_gpu_retry_20260306/results/gpu
mkdir -p Qiskit_Test/run/qiskit30q_60p_spsa_gpu_retry_20260306/slurm_logs

module purge
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vqls

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "========================================="
echo "JobID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "========================================="

python Qiskit_Test/benchmark_hadamard_qiskit.py \
  --compute-target gpu \
  --eval-mode estimator \
  --gradient-method spsa \
  --num-qubits 30 \
  --num-parameters 60 \
  --layers 2 \
  --eval-warmup 2 \
  --eval-repeats 8 \
  --grad-warmup 1 \
  --grad-repeats 5 \
  --json-out Qiskit_Test/run/qiskit30q_60p_spsa_gpu_retry_20260306/results/gpu/qiskit_gpu_30q_60p_estimator_spsa.json \
  --strict \
  --verbose

echo "End: $(date --iso-8601=seconds)"
