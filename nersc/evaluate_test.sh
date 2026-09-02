#!/bin/bash
#SBATCH --job-name=strider_test_set
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

evaluation_split="${STRIDER_EVAL_SPLIT:-test}"
case "$evaluation_split" in
    selection|calibration|test) ;;
    *)
        echo "STRIDER_EVAL_SPLIT must be selection, calibration, or test" >&2
        exit 2
        ;;
esac

echo "STRIDER evaluation — $evaluation_split"
evaluation_arguments=(evaluate --split "$evaluation_split")
if [[ -n "${STRIDER_EVAL_VIEWS:-}" ]]; then
    read -r -a requested_views <<< "$STRIDER_EVAL_VIEWS"
    evaluation_arguments+=(--views "${requested_views[@]}")
fi
if [[ -n "${STRIDER_EVAL_EXTERNAL_PREPARED_DIR:-}" ]]; then
    : "${STRIDER_EVAL_OUTPUT_DIR:?Set STRIDER_EVAL_OUTPUT_DIR for external evaluation}"
    evaluation_arguments+=(
        --external-prepared-dir "$STRIDER_EVAL_EXTERNAL_PREPARED_DIR"
        --output-dir "$STRIDER_EVAL_OUTPUT_DIR"
    )
elif [[ -n "${STRIDER_EVAL_OUTPUT_DIR:-}" ]]; then
    echo "STRIDER_EVAL_OUTPUT_DIR requires STRIDER_EVAL_EXTERNAL_PREPARED_DIR" >&2
    exit 2
fi
run_strider "${evaluation_arguments[@]}"
