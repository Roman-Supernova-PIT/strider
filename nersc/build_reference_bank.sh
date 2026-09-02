#!/bin/bash
#SBATCH --job-name=strider_reference
#SBATCH --account=m4385
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
source "${STRIDER_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/nersc/common.sh"

run_strider build-reference
