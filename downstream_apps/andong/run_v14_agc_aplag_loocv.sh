#!/usr/bin/env bash
# Lagged-ap-history LOOCV: all 32 folds, 7-way parallel on GPUs 0,2-7
# (GPU 1 busy with another user's job). Workers skip folds whose
# predictions already exist (fold 20 done in the single-fold check).
set -uo pipefail
cd /media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong
mkdir -p runs/v14_agc_ap_emu_aplag

GPUS=(0 2 3 4 5 6 7)
NG=${#GPUS[@]}
i=0
for fold in $(seq 0 31); do
    gpu=${GPUS[$((i % NG))]}
    conda run -n dst_longterm_forecast --no-capture-output \
        python -u v14_agc_aplag_fold.py --fold $fold --gpu $gpu --seed 42 \
        > runs/v14_agc_ap_emu_aplag/fold_${fold}.log 2>&1 &
    i=$((i + 1))
    # wait after each wave of NG launches
    if [ $((i % NG)) -eq 0 ]; then
        wait
        echo "wave $((i / NG)) done"
    fi
done
wait
echo "ALL 32 APLAG FOLDS DONE"
