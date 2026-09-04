#!/bin/bash

set -euo pipefail

run_name="${1:?Usage: nersc/bundle_results.sh RUN_NAME}"
run_dir="${STRIDER_OUTPUT_DIR:?Set STRIDER_OUTPUT_DIR}/$run_name"
bundle_dir="${STRIDER_DATA_DIR:?Set STRIDER_DATA_DIR}/artifacts"

test -d "$run_dir"
mkdir -p "$bundle_dir"

archive="$bundle_dir/${run_name}.tar.gz"
tar -czf "$archive" \
  --exclude='*.pt' \
  --exclude='*.parquet' \
  --exclude='*.h5' \
  --exclude='*.npz' \
  -C "$STRIDER_OUTPUT_DIR" \
  "$run_name"

sha256sum "$archive" > "$archive.sha256"
echo "$archive"
