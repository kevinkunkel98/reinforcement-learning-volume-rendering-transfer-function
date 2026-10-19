#!/usr/bin/env bash
# Shards eval_v3.py 6 ways per (checkpoint, split).
# Usage: [O=out/dir/eval] eval_v3.sh CKPT TAG SPLIT [--cold] [--plan]
set -euo pipefail
ckpt=$1; tag=$2; split=$3; shift 3; flags="$*"
O=${O:-out/rl_mb_v3/eval}; mkdir -p $O
for k in 0 1 2 3 4 5; do
  lo=$((k * 50)); hi=$((lo + 50))
  .venv/bin/python docs/experiments/2026-10-19-scripts/eval_v3.py $ckpt $O/${tag}_${split}_$k.json $split $lo $hi $flags > $O/${tag}_${split}_$k.log 2>&1 &
done
wait
