#!/usr/bin/env bash
# Download scalers.yaml and the pretrained Surya backbone into this app's assets/ directory.
#
# Thin wrapper: the implementation lives in workshop_infrastructure/assets.py, which the
# training script and notebooks call directly. Use this when you want the files without
# going through a config.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."
exec python -m workshop_infrastructure.assets --scalers --weights --dest "${SCRIPT_DIR}/assets"
