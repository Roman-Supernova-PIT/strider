#!/bin/bash
#SBATCH --job-name=strider_evaluate
#SBATCH --account=m4385_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=55G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export STRIDER_REQUIRE_CUDA=1
source "${STRIDER_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/nersc/common.sh"

run_step() {
    local title="$1"
    shift
    echo
    echo "$title"
    run_strider "$@"
}

run_step "Main evaluation" evaluate
run_step "Source and no-source controls" paired-controls
run_step "Visit-count controls" visit-controls
run_step "Observation-time controls" time-controls
run_step "Example plots" plot-examples
run_step "Evidence maps" evidence-maps \
    --objects-per-redshift "${STRIDER_EVIDENCE_OBJECTS_PER_REDSHIFT:-8}"
