#!/usr/bin/env bash
# Night of 2026-11-18: SAC on task v3 (fair model-free baseline), then the
# frozen viewer model re-scored with the new shell. Resumable: every stage
# skips what is already there.
#   bash docs/experiments/2026-11-18-scripts/overnight.sh > out/sac_v3/overnight.log 2>&1
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
SAC=docs/experiments/2026-11-18-scripts/sac_v3.py
O=out/sac_v3
export CUDA_VISIBLE_DEVICES=""
mkdir -p $O
step() { echo "[overnight $(date +%H:%M:%S)] $*"; }

# 1. The episode pool: 7500 starts x 8 instructions, as the model-based data.
[ -f $O/pool.pkl ] || { step pool; nice -n 10 $PY -u $SAC pool --workers 10 --out $O/pool.pkl > $O/pool.log 2>&1; tail -n1 $O/pool.log; }

# 2. SAC, three seeds in parallel, 300k steps (checkpoint at 150k = the model-based budget).
for s in 0 1 2; do
  [ -f $O/seed$s/steps_300000.zip ] && continue
  rm -rf $O/seed$s
  step "train seed $s"
  nice -n 10 $PY -u $SAC train --pool $O/pool.pkl --seed $s --out $O/seed$s > $O/seed$s.log 2>&1 &
done
wait
step "training done"; tail -n2 $O/seed*.log

# 3. Evaluation: the SAC answer and a warm search of 30 / 200 from it, on the
#    fixed test (300) and test_extra (600) episodes, 10 shards each.
evaluate() {   # checkpoint tag split episodes
  local ck=$1 tag=$2 split=$3 n=$4 jobs=10
  ls $O/eval/${tag}_${split}_*.json >/dev/null 2>&1 && return
  mkdir -p $O/eval
  local per=$(( (n + jobs - 1) / jobs ))
  for ((k = 0; k < jobs; k++)); do
    local lo=$((k * per)) hi=$((k * per + per)); ((hi > n)) && hi=$n
    nice -n 10 $PY -u $SAC eval $ck --split $split --range $lo $hi --out $O/eval/${tag}_${split}_$k.json \
      > $O/eval/${tag}_${split}_$k.log 2>&1 &
  done
  wait
}
for s in 0 1 2; do
  for steps in 150000 300000; do
    for pair in "test 300" "test_extra 600"; do
      set -- $pair
      step "eval seed $s, $steps steps, $1"
      evaluate $O/seed$s/steps_$steps.zip s${s}_${steps} $1 $2
    done
  done
done
$PY - <<'EOF'
import glob, json, numpy as np
for split in ("test", "test_extra"):
    for steps in (150000, 300000):
        meds = {k: [] for k in ("policy", "warm30", "warm200")}
        for s in (0, 1, 2):
            rows = [r for f in glob.glob(f"out/sac_v3/eval/s{s}_{steps}_{split}_*.json") for r in json.load(open(f))]
            for k in meds:
                meds[k].append(float(np.median([r[k] for r in rows])) if rows else float("nan"))
        print(f"[sac] {split} {steps // 1000}k steps:", "  ".join(
            f"{k} {min(v):.3f}-{max(v):.3f}" for k, v in meds.items()))
EOF

# 4. The frozen viewer model (v3c seed 1) with the new shell, every split.
R=out/eval_new_shell
for pair in "test 300" "val_extra 320" "test_extra 600"; do
  set -- $pair
  ls $R/s1_$1_*.json >/dev/null 2>&1 && grep -q "paired Wilcoxon" $R/s1_$1.log 2>/dev/null && continue
  step "new-shell eval $1"
  mkdir -p $R
  JOBS=12 O=$R EPISODES=$2 bash docs/experiments/2026-11-12-scripts/eval_extra.sh \
    models/v3c_seed1/checkpoint.pt s1 $1 --cold > $R/s1_$1.log 2>&1
  tail -n 12 $R/s1_$1.log
done
step "all done"
