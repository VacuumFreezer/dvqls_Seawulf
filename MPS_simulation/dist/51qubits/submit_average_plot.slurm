#!/bin/bash
#SBATCH --job-name=mps_51q_avgplot
#SBATCH --partition=short-40core-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=MPS_simulation/dist/51qubits/slurm/%x_%j.out
#SBATCH --error=MPS_simulation/dist/51qubits/slurm/%x_%j.err

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

PLOT_INPUTS_FILE="$BENCH_DIR/run/${TAG}/plot_jsons.txt"
PLOT_OUTDIR="$BENCH_DIR/run/${TAG}/plots"
mkdir -p "$PLOT_OUTDIR"

mapfile -t JSONS < "$PLOT_INPUTS_FILE"
if [ "${#JSONS[@]}" -eq 0 ]; then
  echo "No plot inputs found in $PLOT_INPUTS_FILE" >&2
  exit 1
fi

echo "========================================="
echo "JobID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Plot input list: $PLOT_INPUTS_FILE"
echo "Plot outdir: $PLOT_OUTDIR"
echo "Averaging ${#JSONS[@]} runs"
echo "========================================="

python "$BENCH_DIR/make_first100_residual_consensus_horizontal.py" \
  --json "${JSONS[@]}" \
  --outdir "$PLOT_OUTDIR" \
  --max-iteration 100 \
  >"$PLOT_OUTDIR/plot_stdout.log" 2>"$PLOT_OUTDIR/plot_stderr.log"

echo "End: $(date)"
