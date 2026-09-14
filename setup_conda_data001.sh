#!/usr/bin/env bash
set -euo pipefail

BASE="/data001/finetuning_dh"
WORKSHOP="$BASE/surya_workshop"
ENV_YML="$WORKSHOP/environment.yml"
ENV_NAME="surya_dhegde"
ENV_PREFIX="$BASE/conda_envs/$ENV_NAME"

echo "[1/6] Ensuring target directories on $BASE"
TARGET_DIRS=(
  "$BASE/conda_envs"
  "$BASE/conda_pkgs"
  "$BASE/.cache/pip"
  "$BASE/.cache/huggingface"
  "$BASE/.cache/torch"
)
for dir in "${TARGET_DIRS[@]}"; do
  if [ -d "$dir" ]; then
    echo "Exists:  $dir"
  else
    mkdir -p "$dir"
    echo "Created: $dir"
  fi
done

echo "[2/6] Configuring conda to use /data001 paths"
conda config --remove-key envs_dirs >/dev/null 2>&1 || true
conda config --remove-key pkgs_dirs >/dev/null 2>&1 || true
conda config --add envs_dirs "$BASE/conda_envs"
conda config --add pkgs_dirs "$BASE/conda_pkgs"

echo "[3/6] Updating ~/.bashrc cache exports"
sed -i '/^export PIP_CACHE_DIR=/d;/^export HF_HOME=/d;/^export TORCH_HOME=/d' ~/.bashrc
cat >> ~/.bashrc <<EOF
export PIP_CACHE_DIR=$BASE/.cache/pip
export HF_HOME=$BASE/.cache/huggingface
export TORCH_HOME=$BASE/.cache/torch
EOF

echo "[4/6] Exporting cache vars for current run"
export PIP_CACHE_DIR="$BASE/.cache/pip"
export HF_HOME="$BASE/.cache/huggingface"
export TORCH_HOME="$BASE/.cache/torch"

echo "[5/6] Creating environment from environment.yml"
if [ -d "$ENV_PREFIX" ]; then
  echo "Environment already exists at $ENV_PREFIX; updating instead of creating."
  conda env update -n "$ENV_NAME" -f "$ENV_YML" --prune
else
  conda env create -f "$ENV_YML"
fi

echo "[6/6] Verifying conda paths"
conda info | grep -E "envs directories|package cache" || true
echo "Environment list entry:"
conda env list | grep surya_dhegde || true

echo
echo "Done."
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "Script was sourced; loading ~/.bashrc into current shell."
  # shellcheck disable=SC1090
  source ~/.bashrc
else
  echo "Run: source ~/.bashrc"
fi
echo "Then: conda activate surya_dhegde"
