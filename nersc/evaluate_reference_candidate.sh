#!/bin/bash

# Evaluate the active reference candidate on selection only. The historical
# configuration inherits a calibration default, so the split is explicit here.
set -euo pipefail
source "${STRIDER_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/nersc/common.sh"

run_strider evaluate --split selection
