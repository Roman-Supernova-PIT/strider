#!/bin/bash
#SBATCH --job-name=strider_pair
#SBATCH --account=m4385_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=55G
#SBATCH --time=10:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
export STRIDER_REQUIRE_CUDA=1
source "${STRIDER_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/nersc/common.sh"

: "${STRIDER_NOISE_MANIFEST:?Set STRIDER_NOISE_MANIFEST to the complete manifest CSV}"

shard_count="${SLURM_ARRAY_TASK_COUNT:-8}"
shard_index="${SLURM_ARRAY_TASK_ID:-0}"
printf -v shard_label "%02d" "$shard_index"
printf -v count_label "%02d" "$shard_count"
manifest_stem="${STRIDER_NOISE_MANIFEST%.csv}"
manifest="${manifest_stem}_shard_${shard_label}_of_${count_label}.csv"
tag="${STRIDER_NOISE_TAG:-sundial_mixed}_shard_${shard_label}_of_${count_label}"

test -f "$manifest" || {
  echo "Missing paired-noise manifest shard: $manifest" >&2
  exit 1
}

read -r -a noise_scales <<< "${STRIDER_NOISE_SCALES:-0.5 0.75 1.0 1.5 2.0 3.0}"

run_strider noise-check \
  --split "${STRIDER_NOISE_SPLIT:-test}" \
  --noise-family reported-error \
  --scales "${noise_scales[@]}" \
  --repeats "${STRIDER_NOISE_REPEATS:-2}" \
  --object-list "$manifest" \
  --paired-noise-seed "${STRIDER_NOISE_PAIRED_SEED:-20260803}" \
  --output-tag "$tag" \
  --save-predictions
