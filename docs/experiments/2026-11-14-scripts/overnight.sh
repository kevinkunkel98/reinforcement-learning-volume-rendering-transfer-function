#!/usr/bin/env bash
# Overnight retrain 2026-11-14 (v3d): all 100 scans, caches, the full v3 pipeline (3 seeds).
set -euo pipefail
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES=""
echo "[overnight $(date +%H:%M:%S)] fetch scans"
nice -n 10 .venv/bin/python -u -m tools.fetch_scans --splits all
echo "[overnight $(date +%H:%M:%S)] visibility caches"
nice -n 10 .venv/bin/python -u -m tools.build_visibility_cache
nice -n 10 .venv/bin/python -u -m tools.build_visibility_cache --splits train_extra
echo "[overnight $(date +%H:%M:%S)] pipeline"
O=out/rl_mb_v3d nice -n 10 bash docs/experiments/2026-10-19-scripts/pipeline_v3.sh
echo "[overnight $(date +%H:%M:%S)] finished"
