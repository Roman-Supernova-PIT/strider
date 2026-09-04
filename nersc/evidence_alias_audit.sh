#!/bin/bash
#SBATCH --job-name=strider_alias
#SBATCH --account=m4385_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=55G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export STRIDER_REQUIRE_CUDA=1
source "${STRIDER_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/nersc/common.sh"

echo "Sundial redshift-alias evidence audit"
checkpoint_config="${STRIDER_CONFIG:-$STRIDER_OUTPUT_DIR/ia_binary_full_main/config.resolved.yaml}"
object_list="${STRIDER_EVIDENCE_OBJECT_LIST:-$STRIDER_ROOT/analysis/sundial_alias_audit_snids.csv}"
layout="${STRIDER_EVIDENCE_LAYOUT:-summary}"
split="${STRIDER_EVIDENCE_SPLIT:-test}"
view="${STRIDER_EVIDENCE_VIEW:-original}"
test -f "$checkpoint_config"
test -f "$object_list"

export STRIDER_CONFIG="$checkpoint_config"
run_strider evidence-maps \
    --split "$split" \
    --view "$view" \
    --layout "$layout" \
    --object-list "$object_list" \
    --competing-peak-ratio 0.5

run_dir="$(dirname "$checkpoint_config")"
audit_dir="${STRIDER_ALIAS_AUDIT_DIR:-$run_dir/alias_audit}"
"$STRIDER_PYTHON" scripts/summarize_alias_route_audit.py \
    --input "$audit_dir/${split}_${view}_route_audit.csv" \
    --output "$audit_dir/${split}_${view}_route_summary.csv"
