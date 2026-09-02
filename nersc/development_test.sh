#!/bin/bash
#SBATCH --job-name=strider_test
#SBATCH --account=m4385_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=55G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export STRIDER_CONFIG="${STRIDER_CONFIG:-configs/nersc/dev.yaml}"
export STRIDER_REQUIRE_CUDA=1
source "${STRIDER_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/nersc/common.sh"

"$STRIDER_PYTHON" -m pytest -q
run_strider benchmark
