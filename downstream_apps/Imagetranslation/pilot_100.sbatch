#!/bin/bash
#SBATCH --job-name=euv2mag-pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/euv2mag-pilot-%j.out

set -euo pipefail

cd /data001/finetuning_dh/surya_workshop/downstream_apps/Imagetranslation

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

/data001/finetuning_dh/conda_envs/surya_dhegde/bin/ipython \
  3_euv2mag_100train_20val_pilot.py