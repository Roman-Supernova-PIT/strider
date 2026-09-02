#!/bin/bash
#SBATCH --job-name=strider_external_prepare
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

: "${STRIDER_EXTERNAL_TEST_SOURCE_DIR:?Set STRIDER_EXTERNAL_TEST_SOURCE_DIR}"
: "${STRIDER_EXTERNAL_TEST_PREPARED_DIR:?Set STRIDER_EXTERNAL_TEST_PREPARED_DIR}"
: "${STRIDER_EXTERNAL_TEST_TAG:?Set STRIDER_EXTERNAL_TEST_TAG}"

arguments=(
    prepare-external-test
    --source-dir "$STRIDER_EXTERNAL_TEST_SOURCE_DIR"
    --prepared-dir "$STRIDER_EXTERNAL_TEST_PREPARED_DIR"
    --dataset-tag "$STRIDER_EXTERNAL_TEST_TAG"
)
if [[ -n "${STRIDER_EXTERNAL_TEST_BLOCKS:-}" ]]; then
    read -r -a blocks <<< "$STRIDER_EXTERNAL_TEST_BLOCKS"
    arguments+=(--blocks "${blocks[@]}")
fi

run_strider "${arguments[@]}"
