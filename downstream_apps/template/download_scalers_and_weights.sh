#!/usr/bin/env bash
set -euo pipefail

# ---- Config ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSET_DIR="${SCRIPT_DIR}/assets"          # must exist
TARGET_DIR="${ASSET_DIR}/surya_assets"    # download destination

# Optional: token via env var
HF_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"
export HF_HUB_ENABLE_HF_TRANSFER=1

have() { command -v "$1" >/dev/null 2>&1; }

echo "==> Checking assets directory at: ${ASSET_DIR}"
if [[ ! -d "${ASSET_DIR}" ]]; then
  echo "ERROR: Required directory '${ASSET_DIR}' does not exist."
  echo "Create it (e.g., 'mkdir -p \"${ASSET_DIR}\"') and re-run."
  exit 1
fi

mkdir -p "${TARGET_DIR}"

echo "==> Downloading scalers and model weights into: ${TARGET_DIR}"

if have hf; then
  # ---- Scalers ----
  hf download "nasa-ibm-ai4science/core-sdo" \
    --repo-type dataset \
    --local-dir "${TARGET_DIR}" \
    --include "scalers.yaml"

  # ---- Surya base model weights ----
  hf download "nasa-ibm-ai4science/Surya-1.0" \
    --repo-type model \
    --local-dir "${TARGET_DIR}" \
    --include "surya.366m.v1.pt"

else
  # ---- Python fallback ----
  python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

target_dir = os.environ["TARGET_DIR"]
token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

def dl(repo_id, repo_type, patterns):
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        token=token,
        allow_patterns=patterns,
    )

dl("nasa-ibm-ai4science/core-sdo", "dataset", ["scalers.yaml"])
dl("nasa-ibm-ai4science/Surya-1.0", "model", ["surya.366m.v1.pt"])

print("Download complete:", target_dir)
PY
fi

echo "✓ Done. Files are in: ${TARGET_DIR}"
