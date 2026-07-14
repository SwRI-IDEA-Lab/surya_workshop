#!/usr/bin/env bash
# Pre-2015 catalog extension: folds 26-31 (Events #28-#33) on GPUs 0-5.
# Same protocol as run_v14_agc_loocv_8gpu.sh (EPOCHS=30, seed 42).
# Output: runs/v14_agc_ap_emu_loocv_pre2015 (canonical 26-fold dir untouched).
set -euo pipefail
cd /media/faraday/andong/Workspace/surya_workshop/downstream_apps/andong
mkdir -p runs/v14_agc_ap_emu_loocv_pre2015

pids=()
for i in 0 1 2 3 4 5; do
    fold=$((26 + i))
    gpu=$i
    conda run -n dst_longterm_forecast --no-capture-output \
        python -u paris_agc_loocv_fold_v14_ap_emu_pre2015.py --fold $fold --gpu $gpu --seed 42 \
        > runs/v14_agc_ap_emu_loocv_pre2015/fold_${fold}.log 2>&1 &
    pids+=($!)
    echo "launched fold $fold on GPU $gpu (pid ${pids[-1]})"
done
wait
echo "ALL 6 PRE-2015 FOLDS DONE"
