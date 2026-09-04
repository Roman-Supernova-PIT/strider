#!/bin/bash

# Shared checks for the small NERSC launch scripts. The Python environment is
# created separately so batch jobs never install or change packages mid-run.
set -euo pipefail

STRIDER_ROOT="${STRIDER_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
STRIDER_CONFIG="${STRIDER_CONFIG:-configs/nersc/spectral.yaml}"
STRIDER_PYTHON="${STRIDER_PYTHON:-python}"

: "${STRIDER_TRAIN_DIR:?Set STRIDER_TRAIN_DIR to the ten-seed flat-redshift TRANSIENTS directory}"
: "${STRIDER_VALIDATION_DIR:?Set STRIDER_VALIDATION_DIR to the August flat-redshift TRANSIENTS directory}"
: "${STRIDER_TEST_DIR:?Set STRIDER_TEST_DIR to the final Sundial TRANSIENTS directory}"
: "${STRIDER_DATA_DIR:?Set STRIDER_DATA_DIR to persistent prepared-data storage}"
: "${STRIDER_OUTPUT_DIR:?Set STRIDER_OUTPUT_DIR to the run output directory}"

cd "$STRIDER_ROOT"
mkdir -p logs "$STRIDER_DATA_DIR" "$STRIDER_OUTPUT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$STRIDER_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

run_strider() {
    "$STRIDER_PYTHON" -m strider.cli "$@" --config "$STRIDER_CONFIG"
}
