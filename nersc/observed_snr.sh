#!/bin/bash
#SBATCH --job-name=strider_snr
#SBATCH --account=m4385
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
source "${STRIDER_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/nersc/common.sh"

arguments=(observed-snr --split "${STRIDER_SNR_SPLIT:-test}")
if [[ -n "${STRIDER_SNR_PREDICTIONS:-}" ]]; then
    arguments+=(--predictions "$STRIDER_SNR_PREDICTIONS")
fi
if [[ -n "${STRIDER_SNR_OUTPUT:-}" ]]; then
    arguments+=(--output "$STRIDER_SNR_OUTPUT")
fi
if [[ -n "${STRIDER_SNR_EPOCHS_OUTPUT:-}" ]]; then
    arguments+=(--epochs-output "$STRIDER_SNR_EPOCHS_OUTPUT")
fi
if [[ -n "${STRIDER_SNR_EXTERNAL_PREPARED_DIR:-}" ]]; then
    arguments+=(--external-prepared-dir "$STRIDER_SNR_EXTERNAL_PREPARED_DIR")
fi
if [[ -n "${STRIDER_SNR_EDGE_TRIM_FRACTION:-}" ]]; then
    arguments+=(--edge-trim-fraction "$STRIDER_SNR_EDGE_TRIM_FRACTION")
fi
if [[ -n "${STRIDER_SNR_MAXIMUM_RELATIVE_ERROR:-}" ]]; then
    arguments+=(--maximum-relative-error "$STRIDER_SNR_MAXIMUM_RELATIVE_ERROR")
fi

run_strider "${arguments[@]}"
