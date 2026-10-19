#!/usr/bin/env bash
# Task v3, the v2 recipe end to end: collect, train, three Dyna rounds with
# balanced closed-loop fine-tuning, three seeds, official scores.
#   [O=out/dir] bash docs/experiments/2026-10-19-scripts/pipeline_v3.sh
set -euo pipefail
PY=.venv/bin/python
O=${O:-out/rl_mb_v3}
B="--splits train,train_extra --task-version v3"
T="--task-version v3"
mkdir -p $O
step() { echo "[pipeline $(date +%H:%M:%S)] $*"; }
done_ckpt() { [ -f "$1/checkpoint.pt" ]; }

[ -f $O/data_r0.npz ] || { step collect r0; $PY -m rl.model_based collect --out $O/data_r0.npz --starts 3000 --seed-base 7000000 --workers 4 $B > $O/data_r0.log 2>&1; }
done_ckpt $O/r0_seed0 || { step train r0; $PY -m rl.model_based train --data $O/data_r0.npz --out $O/r0_seed0 $T > $O/r0_seed0.log 2>&1; }

[ -f $O/data_r1.npz ] || { step collect r1; $PY -m rl.model_based collect --out $O/data_r1.npz --starts 1500 --seed-base 8000000 --workers 4 --policy $O/r0_seed0/checkpoint.pt $B > $O/data_r1.log 2>&1; }
D=$O/data_r0.npz,$O/data_r1.npz
done_ckpt $O/r1_seed0 || { step train r1; $PY -m rl.model_based train --data $D --out $O/r1_seed0 $T > $O/r1_seed0.log 2>&1; }
done_ckpt $O/clb_r1_seed0 || { step closed-loop r1; $PY -m rl.closed_loop --init $O/r1_seed0/checkpoint.pt --data $D --out $O/clb_r1_seed0 --balance-kinds $T > $O/clb_r1_seed0.log 2>&1; }

[ -f $O/data_r2.npz ] || { step collect r2; $PY -m rl.model_based collect --out $O/data_r2.npz --starts 1500 --seed-base 9000000 --workers 4 --policy $O/clb_r1_seed0/checkpoint.pt $B > $O/data_r2.log 2>&1; }
D=$D,$O/data_r2.npz
done_ckpt $O/r2_seed0 || { step train r2; $PY -m rl.model_based train --data $D --out $O/r2_seed0 $T > $O/r2_seed0.log 2>&1; }
done_ckpt $O/clb_r2_seed0 || { step closed-loop r2; $PY -m rl.closed_loop --init $O/r2_seed0/checkpoint.pt --data $D --out $O/clb_r2_seed0 --balance-kinds $T > $O/clb_r2_seed0.log 2>&1; }

[ -f $O/data_r3.npz ] || { step collect r3; $PY -m rl.model_based collect --out $O/data_r3.npz --starts 1500 --seed-base 4000000 --workers 4 --policy $O/clb_r2_seed0/checkpoint.pt $B > $O/data_r3.log 2>&1; }
D=$D,$O/data_r3.npz
step train r3 x3
for s in 0 1 2; do
  done_ckpt $O/r3_seed$s || $PY -m rl.model_based train --data $D --out $O/r3_seed$s --seed $s --threads 3 $T > $O/r3_seed$s.log 2>&1 &
done
wait
step closed-loop r3 x3
for s in 0 1 2; do
  done_ckpt $O/clb_r3_seed$s || $PY -m rl.closed_loop --init $O/r3_seed$s/checkpoint.pt --data $D --out $O/clb_r3_seed$s --seed $s --balance-kinds --threads 3 $T > $O/clb_r3_seed$s.log 2>&1 &
done
wait
for s in 0 1 2; do done_ckpt $O/clb_r3_seed$s || { echo "missing clb_r3_seed$s"; exit 1; }; done
step done
