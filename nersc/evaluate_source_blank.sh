#!/bin/bash
#SBATCH --job-name=strider_source_blank
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

STRIDER_EVAL_OBJECTS="${STRIDER_EVAL_OBJECTS:-1600}"

echo "Source and blank spectra — ${STRIDER_EVAL_OBJECTS} objects"
run_strider paired-controls --max-objects "$STRIDER_EVAL_OBJECTS"
