#!/bin/bash
#SBATCH --job-name=strider_atlas
#SBATCH --account=m4385
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

export STRIDER_CONFIG="${STRIDER_CONFIG:-configs/nersc/ia_binary_20k.yaml}"
source "${STRIDER_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/nersc/common.sh"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

study_output="${STRIDER_ATLAS_STUDY_OUTPUT:-$STRIDER_OUTPUT_DIR/atlas_study/binary_selection}"

arguments=(
    --config "$STRIDER_CONFIG"
    --output-dir "$study_output"
    --train-objects "${STRIDER_ATLAS_TRAIN_OBJECTS:-0}"
    --selection-objects "${STRIDER_ATLAS_SELECTION_OBJECTS:-0}"
    --prototypes-per-class "${STRIDER_ATLAS_PROFILES_PER_CLASS:-8}"
    --phase-profiles-per-cell "${STRIDER_ATLAS_PHASE_PROFILES_PER_CELL:-4}"
    --redshift-bins "${STRIDER_ATLAS_REDSHIFT_BINS:-161}"
    --max-visits "${STRIDER_ATLAS_MAX_VISITS:-all}"
    --phase-visits "${STRIDER_ATLAS_PHASE_VISITS:-6}"
    --phase-starting-points "${STRIDER_ATLAS_PHASE_STARTING_POINTS:-17}"
)

python scripts/study_full_spectrum_atlas.py "${arguments[@]}"

download_file="$STRIDER_OUTPUT_DIR/atlas_study_results.tar.gz"
tar -czf "$download_file" -C "$study_output" .
echo "Download: $download_file"
