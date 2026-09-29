#!/usr/bin/env bash
#
# Put conda environments and package caches on a shared/scratch filesystem instead of $HOME.
#
# Why you would want this: a full surya_ws environment plus its package cache runs to tens
# of GB, and the HuggingFace cache holds a 1.8 GB checkpoint. On a cluster where $HOME is a
# small quota'd NFS volume, that either fails outright or makes every import slow. This
# points conda's envs_dirs/pkgs_dirs and the pip/HF/torch caches at a filesystem with room.
#
# Generalized from a script written for one machine (/data001/finetuning_dh, env name
# surya_dhegde) during the first workshop. Both are now variables.
#
# Usage:
#   SURYA_WS_BASE=/scratch/$USER bash workshop_infrastructure/setup_scripts/setup_conda_shared_storage.sh
#
#   SURYA_WS_BASE  Filesystem with room for environments and caches. Default: $HOME,
#                  which makes the script a no-op reorganization -- set it to somewhere
#                  with space for it to be worth running.
#   SURYA_WS_ENV   Conda environment name. Default: surya_ws, matching environment.yml.
#   SURYA_WS_REPO  Repo checkout holding environment.yml. Default: inferred from this
#                  script's location.
#
# It edits ~/.bashrc (idempotently) so the cache variables survive new shells.

set -euo pipefail

BASE="${SURYA_WS_BASE:-$HOME}"
ENV_NAME="${SURYA_WS_ENV:-surya_ws}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SURYA_WS_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
ENV_YML="$REPO/environment.yml"

if [ ! -f "$ENV_YML" ]; then
  echo "ERROR: no environment.yml at $ENV_YML" >&2
  echo "Set SURYA_WS_REPO to your surya_workshop checkout." >&2
  exit 1
fi

if [ "$BASE" = "$HOME" ]; then
  echo "NOTE: SURYA_WS_BASE is \$HOME, so nothing moves off your home filesystem."
  echo "      Set SURYA_WS_BASE=/scratch/\$USER (or your cluster's equivalent) if that"
  echo "      is what you meant to do."
fi

echo "Base:        $BASE"
echo "Environment: $ENV_NAME"
echo "Repo:        $REPO"
echo

echo "[1/6] Ensuring target directories under $BASE"
for dir in "$BASE/conda_envs" "$BASE/conda_pkgs" "$BASE/.cache/pip" \
           "$BASE/.cache/huggingface" "$BASE/.cache/torch"; do
  mkdir -p "$dir"
  echo "  $dir"
done

echo "[2/6] Pointing conda at $BASE"
# --remove (one entry) rather than --remove-key (the whole list): the list may hold
# directories this script never added, and dropping those would make the user's other named
# environments stop resolving. Removing our own entry first keeps re-runs from stacking
# duplicates; the || true covers the first run, when there is nothing to remove.
conda config --remove envs_dirs "$BASE/conda_envs" >/dev/null 2>&1 || true
conda config --remove pkgs_dirs "$BASE/conda_pkgs" >/dev/null 2>&1 || true
conda config --add envs_dirs "$BASE/conda_envs"
conda config --add pkgs_dirs "$BASE/conda_pkgs"

echo "[3/6] Updating ~/.bashrc cache exports"
# Delete before appending, so this is idempotent rather than additive. Guarded because
# set -e would otherwise abort here on a machine with no ~/.bashrc -- after step 2 has
# already reconfigured conda, leaving a half-applied setup.
touch ~/.bashrc
sed -i '/^export PIP_CACHE_DIR=/d;/^export HF_HOME=/d;/^export TORCH_HOME=/d' ~/.bashrc
cat >> ~/.bashrc <<EOF
export PIP_CACHE_DIR=$BASE/.cache/pip
export HF_HOME=$BASE/.cache/huggingface
export TORCH_HOME=$BASE/.cache/torch
EOF

echo "[4/6] Exporting cache vars for this run"
export PIP_CACHE_DIR="$BASE/.cache/pip"
export HF_HOME="$BASE/.cache/huggingface"
export TORCH_HOME="$BASE/.cache/torch"

echo "[5/6] Creating or updating the environment from environment.yml"
if [ -d "$BASE/conda_envs/$ENV_NAME" ]; then
  echo "  Exists; updating with --prune."
  conda env update -n "$ENV_NAME" -f "$ENV_YML" --prune
else
  conda env create -n "$ENV_NAME" -f "$ENV_YML"
fi

echo "[6/6] Verifying"
conda info | grep -E "envs directories|package cache" || true
conda env list | grep -E "(^|/)$ENV_NAME\b" || true

cat <<EOF

Done.

  source ~/.bashrc
  conda activate $ENV_NAME

You will also want a cache directory for the S3 data reads, which is separate from the
caches above (budget ~1 GB per unique timestep):

  export SURYA_WS_CACHE_DIR=$BASE/helio_s3_cache

That is the third of the three ways to set data.s3_cache_dir, and the one that keeps the
checked-in config machine-independent.
EOF
