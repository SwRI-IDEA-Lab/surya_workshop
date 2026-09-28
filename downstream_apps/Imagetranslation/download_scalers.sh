#!/usr/bin/env bash
set -euo pipefail

# ---- Config ----
REPO_ID="nasa-ibm-ai4science/core-sdo"
REPO_TYPE="dataset"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSET_DIR="${SCRIPT_DIR}/assets"

# Optional: use an existing token non-interactively
HF_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"
HF_HUB_ENABLE_HF_TRANSFER=1

have() { command -v "$1" >/dev/null 2>&1; }

# ---- Step 1: Check assets directory exists ----
echo "==> Checking assets directory at: ${ASSET_DIR}"
if [[ ! -d "${ASSET_DIR}" ]]; then
  echo "ERROR: Required directory '${ASSET_DIR}' does not exist."
  echo "Create it (e.g., 'mkdir -p \"${ASSET_DIR}\"') and re-run."
  exit 1
fi

# ---- Step 2: Download scalers.yaml only ----
echo "==> Downloading scalers.yaml from ${REPO_ID}"

if have hf; then
  hf download "${REPO_ID}" \
    --repo-type "${REPO_TYPE}" \
    --local-dir "${ASSET_DIR}" \
    --include "scalers.yaml"
else
  python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nasa-ibm-ai4science/core-sdo",
    repo_type="dataset",
    local_dir="assets",
    local_dir_use_symlinks=False,
    allow_patterns=["scalers.yaml"],
)
print("✓ Downloaded scalers.yaml")
PY
fi

echo "✓ Done. File is located at: ${ASSET_DIR}/scalers.yaml"
