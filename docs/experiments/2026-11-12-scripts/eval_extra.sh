#!/usr/bin/env bash
# rl.evaluate on a split, sharded over processes (one thread each), then the
# patient-level summary.
# Usage: [JOBS=16] [EPISODES=600] [O=out/eval_extra] eval_extra.sh CKPT TAG SPLIT [rl.evaluate flags...]
#   eval_extra.sh models/v3c_seed1/checkpoint.pt s1 test_extra --cold --colour
set -euo pipefail
ckpt=$1; tag=$2; split=$3; shift 3
JOBS=${JOBS:-16}; EPISODES=${EPISODES:-600}; O=${O:-out/eval_extra}
mkdir -p "$O"
export CUDA_VISIBLE_DEVICES=""   # CPU only: torch would open a CUDA context next to Ollama
per=$(( (EPISODES + JOBS - 1) / JOBS ))
started=$(date +%s)
for ((k = 0; k < JOBS; k++)); do
  lo=$((k * per)); hi=$((lo + per)); ((hi > EPISODES)) && hi=$EPISODES; ((lo >= hi)) && break
  .venv/bin/python -m rl.evaluate "$ckpt" --split "$split" --range $lo $hi --out "$O/${tag}_${split}_$k.json" "$@" \
    > "$O/${tag}_${split}_$k.log" 2>&1 &
done
wait
echo "[eval_extra] $EPISODES episodes in $(( $(date +%s) - started )) s"
.venv/bin/python docs/experiments/2026-11-12-scripts/patient_stats.py "$O/${tag}_${split}"_*.json
