#!/usr/bin/env bash
# v3d seeds on the extra splits, 12 workers, no --cold (copied from v3c, see merge_cold.py).
set -uo pipefail
cd "$(dirname "$0")/../../.."
O=out/rl_mb_v3d/eval_extra
for s in 0 2 1; do
  ck=out/rl_mb_v3d/clb_r3_seed$s/checkpoint.pt
  for split in val_extra test_extra; do
    n=320; [ $split = test_extra ] && n=600
    log=out/rl_mb_v3d/eval_d${s}_$split.log
    if grep -qs "paired Wilcoxon" $log; then echo "[eval_after] seed $s $split already done, skip"; continue; fi
    rm -f $O/d${s}_${split}_*                      # a stopped run leaves partial shards
    echo "[eval_after $(date +%H:%M:%S)] seed $s $split"
    JOBS=12 O=$O EPISODES=$n bash docs/experiments/2026-11-12-scripts/eval_extra.sh "$ck" d$s $split --colour > $log 2>&1 \
      || echo "[eval_after] FAILED seed $s $split"
    .venv/bin/python docs/experiments/2026-11-14-scripts/merge_cold.py $split "$O/d${s}_${split}_*.json" >> $log 2>&1
    .venv/bin/python docs/experiments/2026-11-12-scripts/patient_stats.py $O/d${s}_${split}_*.json >> $log 2>&1
    grep -v "^\[visibility\]" $log | tail -16
  done
done
echo "[eval_after $(date +%H:%M:%S)] finished"
