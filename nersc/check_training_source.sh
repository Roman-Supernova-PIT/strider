#!/bin/bash
#SBATCH --job-name=strider_source_test
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

# Read a bounded part of the flat-z training production without requiring the
# independent validation or final-test paths.
set -euo pipefail

STRIDER_ROOT="${STRIDER_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
STRIDER_PYTHON="${STRIDER_PYTHON:-python}"
STRIDER_CONFIG="configs/nersc/training_data_test.yaml"

: "${STRIDER_TRAIN_DIR:?Set STRIDER_TRAIN_DIR to the flat-z training TRANSIENTS directory}"
: "${STRIDER_DATA_DIR:?Set STRIDER_DATA_DIR to prepared-data storage}"
: "${STRIDER_OUTPUT_DIR:?Set STRIDER_OUTPUT_DIR to run output storage}"

cd "$STRIDER_ROOT"
mkdir -p logs "$STRIDER_DATA_DIR" "$STRIDER_OUTPUT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$STRIDER_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$STRIDER_PYTHON" -m strider.cli inspect --config "$STRIDER_CONFIG"
"$STRIDER_PYTHON" -m strider.cli prepare --config "$STRIDER_CONFIG"
"$STRIDER_PYTHON" -m strider.cli class-support --config "$STRIDER_CONFIG"
