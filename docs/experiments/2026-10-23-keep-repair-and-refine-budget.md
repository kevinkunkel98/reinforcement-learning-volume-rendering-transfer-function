# Keep repair, refine budget and the two-stage answer — 2026-10-23

Branch `voice-lexicon`. Why: a live VR session (2026-10-22) showed classes
nobody mentioned drifting over several commands (heart rendered black, a
bluish shell with streaks), and voice-to-render took ~4.5 s, ~3 s of it the
viewer's 200-evaluation refine. See `2026-10-22-voice-e2e-run.md` and STATUS
"Drift accumulates over a session".

## What changed

- **Keep repair** (`rl/keep_repair.py`, `04703be`): after an answer, every
  peak of a class the goal does not mention (neither visibility nor
  brightness mask) that differs from the start is reverted, largest change
  first, when attainment drops by at most `eps = 0.01` against the answer's
  own score. ~6.5 visibility evaluations per instruction. Occluders (mask set,
  weight 0) count as mentioned and are not candidates.
- **Two-stage answer** (`02a2ce2`, `7a20694`): the viewer's policy mode shows
  policy + planning + keep at once, runs the 200-evaluation refine (+ keep) in
  a worker thread and swaps the refined step in place (same id). The final
  params are bit-identical to the old synchronous path. `POLICY_TWO_STAGE=0`
  restores the synchronous behaviour.
- `eval_v3.py --keep` adds `<name>_keep` rows and unmentioned-class drift
  (`drift`: Σ|Δlog10 vis| over unmentioned classes + the unlabelled shell;
  `bdrift`: Σ|Δbrightness|).

## Result (v3c, the fixed 300 episodes per split, seeds 0/1/2)

Median attainment (val / test, range over seeds), median brightness drift:

| answer | val | test | bdrift | when the viewer shows it |
|---|---|---|---|---|
| policy | 0.49–0.50 | 0.47–0.48 | 0.10–0.15 | |
| policy + keep | 0.52 | 0.48–0.50 | 0.01–0.03 | |
| plan | 0.62–0.63 | 0.62–0.65 | 0.24–0.28 | |
| **plan + keep** | 0.66–0.69 | 0.68–0.71 | 0.03–0.05 | first stage (~0.7 s) |
| plan + 30 | 0.75–0.77 | 0.77–0.78 | 0.24–0.27 | |
| plan + 30 + keep | 0.77–0.81 | 0.78–0.79 | 0.05–0.07 | |
| plan + 200 | 0.89–0.90 | 0.90–0.92 | 0.23–0.26 | old viewer answer |
| **plan + 200 + keep** | 0.89–0.91 | 0.90–0.91 | 0.09–0.11 | final stage (~3.8 s) |

- The keep repair costs at most 0.004 in median attainment (warm30, test
  seed 2) and raises it where the answer still has room (plan: +0.04 to
  +0.07; paired Wilcoxon p ≤ 2e-9 on every seed and split). After the
  200-evaluation search the median is unchanged (within ±0.004) while
  brightness drift falls by more than half.
- A 30-evaluation refine is clearly worse than 200 (≈0.78 vs ≈0.90), so the
  viewer keeps 200 and hides its cost behind the first stage instead.
- "p = 0" in `summarize_v3.py` output is float underflow in its 2-digit
  format, not a bug in the test.

## Latency

Measured under load (the eval above running): fast stage 2.9–4.1 s, final
17.6–18.5 s, synchronous 15.1–16.4 s for the same four commands; the refine
alone 12.5–13 s. Scaled by the idle numbers from 2026-10-22 (refine 3.0 s,
plan 0.6 s): first ≈ 0.7 s, final ≈ 3.8 s. To re-measure idle:
`/private/...scratchpad/measure_two_stage.py` (not in the repo; the server's
`[voice-timing]` console line now reports `first=` and `final=`).

## Reproduce

```bash
O=<dir> docs/experiments/2026-10-19-scripts/eval_v3.sh out/rl_mb_v3c/clb_r3_seed$S/checkpoint.pt bud_s$S $SPLIT --plan --keep
O=<dir> .venv/bin/python docs/experiments/2026-10-19-scripts/summarize_v3.py bud_s0 $SPLIT bud_s1 $SPLIT bud_s2 $SPLIT
```

for S in 0 1 2 and SPLIT in val, test (~90 min for all six on the M1 Pro).

## Open

- Occluders as keep-repair candidates: "more lungs" still erases soft tissue
  completely and leaves the shell bluish over a session (occluders are
  mentioned with weight 0, so never reverted).
- `ts_s0477` is flagged no-contrast (`contrast` = "angiography" in the study
  type) but its vessel voxels have median 127 HU (p90 152), i.e. a contrast
  phase; `ts_s0358` is native (median 40 HU), `ts_s1379` angiographic (214).
  Vessel goals exist only on angiography scans today. Defining "contrast" by
  measured vessel HU would change the task set (and every reported number) —
  a thesis decision, not an app fix.
