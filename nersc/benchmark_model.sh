#!/bin/bash
#SBATCH --job-name=strider_benchmark
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
export STRIDER_REQUIRE_CUDA=1

# A benchmark is normally tied to a specific experiment config.  Do not let a
# missing sbatch export silently fall back to the generic spectral config,
# because that can benchmark the wrong prepared store and ONIR bank.
if [[ -z "${STRIDER_CONFIG:-}" ]]; then
    echo "Set STRIDER_CONFIG to the experiment configuration before submitting." >&2
    exit 2
fi

source "${STRIDER_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/nersc/common.sh"

run_strider benchmark
