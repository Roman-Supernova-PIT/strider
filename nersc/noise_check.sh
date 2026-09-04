#!/bin/bash
#SBATCH --job-name=strider_noise
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

read -r -a noise_scales <<< "${STRIDER_NOISE_SCALES:-0.0 0.25 0.5 0.75 1.0 1.5}"
noise_args=(noise-check --scales "${noise_scales[@]}")

if [[ -n "${STRIDER_NOISE_SPLIT:-}" ]]; then
  noise_args+=(--split "$STRIDER_NOISE_SPLIT")
fi
if [[ -n "${STRIDER_NOISE_FAMILY:-}" ]]; then
  noise_args+=(--noise-family "$STRIDER_NOISE_FAMILY")
fi
if [[ -n "${STRIDER_NOISE_OBJECTS_PER_BIN:-}" ]]; then
  noise_args+=(--objects-per-redshift-bin "$STRIDER_NOISE_OBJECTS_PER_BIN")
fi
if [[ -n "${STRIDER_NOISE_REDSHIFT_EDGES:-}" ]]; then
  read -r -a redshift_edges <<< "$STRIDER_NOISE_REDSHIFT_EDGES"
  noise_args+=(--redshift-edges "${redshift_edges[@]}")
fi
if [[ -n "${STRIDER_NOISE_REPEATS:-}" ]]; then
  noise_args+=(--repeats "$STRIDER_NOISE_REPEATS")
fi
if [[ -n "${STRIDER_NOISE_TAG:-}" ]]; then
  noise_args+=(--output-tag "$STRIDER_NOISE_TAG")
fi
if [[ -n "${STRIDER_NOISE_OBJECT_LIST:-}" ]]; then
  noise_args+=(--object-list "$STRIDER_NOISE_OBJECT_LIST")
fi
if [[ -n "${STRIDER_NOISE_PAIRED_SEED:-}" ]]; then
  noise_args+=(--paired-noise-seed "$STRIDER_NOISE_PAIRED_SEED")
fi
if [[ "${STRIDER_NOISE_SAVE_PREDICTIONS:-0}" == "1" ]]; then
  noise_args+=(--save-predictions)
fi
if [[ "${STRIDER_NOISE_IA_ONLY:-0}" == "1" ]]; then
  noise_args+=(--ia-only)
fi
if [[ -n "${STRIDER_EVAL_OBJECTS:-}" \
  && -z "${STRIDER_NOISE_OBJECTS_PER_BIN:-}" \
  && -z "${STRIDER_NOISE_OBJECT_LIST:-}" \
  && "${STRIDER_NOISE_IA_ONLY:-0}" != "1" ]]; then
  noise_args+=(--max-objects "$STRIDER_EVAL_OBJECTS")
fi

run_strider "${noise_args[@]}"
