#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

gen_output=$(python GPU_Test/generate_tasks.py "$@")
echo "$gen_output"

tag=$(echo "$gen_output" | awk -F= '/^TAG=/{print $2}')
if [[ -z "${tag}" ]]; then
  echo "Could not parse TAG from generate_tasks.py output."
  exit 2
fi

cpu_submit=$(sbatch --export=ALL,TAG="$tag" GPU_Test/submit_array_cpu.slurm)
gpu_submit=$(sbatch --export=ALL,TAG="$tag" GPU_Test/submit_array_gpu.slurm)

echo "$cpu_submit"
echo "$gpu_submit"
echo "SUITE_DIR=${ROOT_DIR}/GPU_Test/run/${tag}"
