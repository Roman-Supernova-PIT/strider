#!/bin/bash
#SBATCH --job-name=strider_train
#SBATCH --account=m4385_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=55G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export STRIDER_REQUIRE_CUDA=1
source "${STRIDER_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/nersc/common.sh"

echo "Class and redshift coverage"
run_strider class-support

monitor_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi dmon -s pucm -d 10 -o DT > "$STRIDER_OUTPUT_DIR/gpu_usage_${SLURM_JOB_ID:-interactive}.txt" &
    monitor_pid=$!
    trap '[[ -n "$monitor_pid" ]] && kill "$monitor_pid" 2>/dev/null || true' EXIT
fi

if [[ "${RESUME:-0}" == "1" ]]; then
    echo
    echo "Continuing training"
    run_strider train --resume
else
    echo
    echo "Speed check"
    run_strider benchmark
    echo
    echo "Model training"
    run_strider train
fi
