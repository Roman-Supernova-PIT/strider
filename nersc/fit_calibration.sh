#!/bin/bash
#SBATCH --job-name=strider_calibrate
#SBATCH --account=m4385
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
source "${STRIDER_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/nersc/common.sh"

calibration_arguments=(fit-calibration)
if [[ -n "${STRIDER_CALIBRATION_SOURCE:-}" ]]; then
    calibration_arguments+=(--source-predictions "$STRIDER_CALIBRATION_SOURCE")
fi
if [[ -n "${STRIDER_CALIBRATION_BLANK:-}" ]]; then
    calibration_arguments+=(--blank-predictions "$STRIDER_CALIBRATION_BLANK")
fi
if [[ -n "${STRIDER_CALIBRATION_OUTPUT:-}" ]]; then
    calibration_arguments+=(--output "$STRIDER_CALIBRATION_OUTPUT")
fi

run_strider "${calibration_arguments[@]}"
