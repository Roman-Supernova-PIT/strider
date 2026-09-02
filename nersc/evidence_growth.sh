#!/bin/bash
#SBATCH --job-name=strider_growth
#SBATCH --account=m4385_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=55G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export STRIDER_REQUIRE_CUDA=1
source "${STRIDER_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/nersc/common.sh"

run_strider evidence-growth \
    --max-objects 500 \
    --repeats 1 \
    --visit-counts 1 4 16 32 \
    --exponents 0.0 0.15 0.25 0.35 0.5
