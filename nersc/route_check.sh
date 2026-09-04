#!/bin/bash
#SBATCH --job-name=strider_routes
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

STRIDER_EVAL_OBJECTS="${STRIDER_EVAL_OBJECTS:-800}"
STRIDER_ROUTE_VIEWS="${STRIDER_ROUTE_VIEWS:-generated clean no_source}"
read -r -a route_views <<< "$STRIDER_ROUTE_VIEWS"

route_args=(route-check \
    --max-objects "$STRIDER_EVAL_OBJECTS" \
    --views "${route_views[@]}")

if [[ -n "${STRIDER_ROUTE_SPLIT:-}" ]]; then
    route_args+=(--split "$STRIDER_ROUTE_SPLIT")
fi

run_strider "${route_args[@]}"
