#!/bin/bash
#
# SLURM submission for the EUV2MAG fine-tune.
#
# The sample caps, epochs, batch size and everything else live in
# configs/config_script.yaml -- this file only asks SLURM for resources and launches the
# script. The name "pilot_100" is historical: the 100-train/20-val regime is now
# data.max_train_samples / data.max_val_samples in the YAML, so changing the size of the
# run is a config edit (or --max-train-samples), not a new sbatch file.
#
# Set these before submitting, or edit the defaults below:
#   SURYA_WS_ENV        conda environment name          (default: surya_ws)
#   SURYA_WS_CACHE_DIR  local cache for the S3 reads     (no default -- required)
#
#   cd downstream_apps/Imagetranslation
#   mkdir -p slurm_logs          # SLURM opens --output BEFORE the job runs and will not
#                                # create the directory; without this the submission fails
#                                # with "Unable to open file". Only needed once.
#   sbatch --export=ALL,SURYA_WS_CACHE_DIR=/scratch/$USER/helio_cache pilot_100.sbatch
#
# --output is relative to the directory you submit from, which is why the two lines above
# go together. Submitting from elsewhere means creating slurm_logs/ there instead.
#
#SBATCH --job-name=euv2mag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/euv2mag-%j.out

set -euo pipefail

# Run from the repo root: the script is launched with -m, and the config's relative paths
# resolve against the config file rather than the working directory. Derived from this
# file's own location, so there is no absolute path to fix up on a different machine.
SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SBATCH_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg          # No display on a compute node; the figure still writes.

ENV_NAME="${SURYA_WS_ENV:-surya_ws}"

if [ -z "${SURYA_WS_CACHE_DIR:-}" ]; then
  echo "ERROR: SURYA_WS_CACHE_DIR is not set." >&2
  echo "The index points at s3://, so the run needs a local cache directory (~1 GB per" >&2
  echo "unique timestep). Submit with:" >&2
  echo "  sbatch --export=ALL,SURYA_WS_CACHE_DIR=/scratch/\$USER/helio_cache $(basename "${BASH_SOURCE[0]}")" >&2
  exit 1
fi
mkdir -p "$SURYA_WS_CACHE_DIR"

echo "Host:   $(hostname)"
echo "Repo:   $REPO_ROOT"
echo "Env:    $ENV_NAME"
echo "Cache:  $SURYA_WS_CACHE_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

# conda run rather than a hardcoded interpreter path, so this works wherever the
# environment lives. "$@" forwards any extra flags given to sbatch.
conda run --no-capture-output -n "$ENV_NAME" \
  python -m downstream_apps.Imagetranslation.3_finetune_euv2mag "$@"
