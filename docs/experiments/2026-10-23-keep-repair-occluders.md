# Keep repair with occluders — 2026-10-23

Branch `voice-lexicon`. Follows `2026-10-23-keep-repair-and-refine-budget.md`,
"Open": task v2/v3 goals mark a target's occluders (soft tissue, skeleton, the
unlabelled shell for "more lungs") as mentioned with weight 0, so the keep
repair never reverted them, and "crank up the lungs" / "make the liver more
visible" erase soft tissue completely (−5.0 log-vis).

## What changed

- `keep_repair(..., include_occluders=False)` (`rl/keep_repair.py`): when
  True, a second pass runs on the plain keep repair's result and tries the
  occluders' peaks (`occluder_classes`: mentioned, weight 0), largest change
  first, against max(proposal score, first-pass score) − eps. Mentioned
  classes with a non-zero weight are never touched. A first version tried
  occluders in the same pass as the unmentioned peaks; on v3c val that lost
  up to 0.17 against plain keep on single episodes (an occluder revert,
  cheap on its own, changed what later reverts could gain), hence the second
  pass and its bar: the answer stays within eps of both the proposal and the
  plain keep repair.
- Why the attainment check protects fades the goal needs: occluders are
  weight 0, so `distance` sees them only through what they hide. Reverting a
  fade that reveals the target costs the target's term and fails the bar.
- `eval_v3.py --keep-occ` adds `<name>_keepocc` rows (+ `_evals`, `_ms`,
  drift, bdrift) and, for every answer, `_odrift` / `_obdrift` (occluder
  drift, which `_drift` does not cover). `summarize_v3.py` prints them in the
  keep-repair block with p against keep.
- `server.KEEP_REPAIR_OCCLUDERS` (env, default off) passes the flag in both
  policy stages.

## Result (v3c, the fixed 300 episodes per split, seeds 0/1/2)

Attainment median / mean / share ≥ 0.5, median drift / bdrift, paired
Wilcoxon p of keepocc vs keep, and how many of 300 answers keepocc changes:

| split | answer | seed | keep | keepocc | drift k/o | bdrift k/o | p | changed |
|---|---|---|---|---|---|---|---|---|
| val | plan | 0 | 0.664 / 0.632 / 0.71 | 0.666 / 0.635 / 0.71 | 0.53 / 0.52 | 0.05 / 0.05 | 0.005 | 18 |
| val | plan | 1 | 0.689 / 0.647 / 0.75 | 0.707 / 0.651 / 0.75 | 0.55 / 0.54 | 0.05 / 0.05 | 2e-4 | 19 |
| val | plan | 2 | 0.682 / 0.628 / 0.75 | 0.684 / 0.632 / 0.75 | 0.53 / 0.53 | 0.04 / 0.04 | 0.001 | 18 |
| test | plan | 0 | 0.689 / 0.612 / 0.68 | 0.692 / 0.614 / 0.68 | 0.53 / 0.52 | 0.04 / 0.04 | 0.14 | 8 |
| test | plan | 1 | 0.706 / 0.614 / 0.69 | 0.708 / 0.616 / 0.70 | 0.54 / 0.54 | 0.05 / 0.05 | 0.03 | 8 |
| test | plan | 2 | 0.684 / 0.610 / 0.70 | 0.692 / 0.613 / 0.71 | 0.52 / 0.52 | 0.03 / 0.03 | 0.04 | 6 |
| val | plan + 200 | 0 | 0.892 / 0.819 / 0.93 | identical | 0.65 / 0.65 | 0.11 / 0.11 | 0.10 | 4 |
| val | plan + 200 | 1 | 0.906 / 0.824 / 0.93 | identical | 0.63 / 0.63 | 0.10 / 0.10 | 1 | 2 |
| val | plan + 200 | 2 | 0.903 / 0.816 / 0.94 | identical | 0.63 / 0.63 | 0.10 / 0.10 | 1 | 3 |
| test | plan + 200 | 0 | 0.909 / 0.791 / 0.92 | identical | 0.57 / 0.57 | 0.10 / 0.10 | 1 | 2 |
| test | plan + 200 | 1 | 0.913 / 0.790 / 0.90 | identical | 0.57 / 0.57 | 0.09 / 0.09 | 1 | 1 |
| test | plan + 200 | 2 | 0.903 / 0.786 / 0.91 | identical | 0.57 / 0.58 | 0.09 / 0.09 | 1 | 1 |

- 79 (val) / 60 (test) of 300 instructions have occluders. On the plan
  answer keepocc changes 6–19 of them; there attainment rises (mean +0.05 to
  +0.15 on the changed episodes, because the occluder revert also undoes
  keep-term drift elsewhere) and occluder drift falls (val: −0.31 log-vis on
  the changed episodes). Mean occluder drift over all 300: val 0.23–0.24 → 0.21–0.22,
  test unchanged within 0.003.
- On plan + 200, the viewer's final answer, keepocc changes 1–4 of 300
  answers and nothing measurable.
- Cost: +0.2 to +0.4 visibility evaluations per instruction (~25 ms).

## Visual check (ts_s0477, viewer path, `Session._run_policy`)

The 7-command chain and "make the liver more visible" from the start give
**bit-identical final params** with keep and keepocc: every occluder revert
is refused. For the liver command the answer scores 0.886; reverting the
soft peak scores −0.75 (all 6 values) or 0.43 (height only), and even a 10 %
partial revert costs 0.05–0.08 (soft visibility 0.12 at the start, 0.02
after the 10 % revert). "highlight the lungs a lot" behaves the same (0.914 →
0.78 at a 10 % revert). So partial reverts would be refused too at
eps = 0.01, and none were added.

The soft-tissue erasure is not a side effect the repair can undo: under the
goal as defined (target +0.3 / +0.6 log-vis, occluder free at weight 0) it is
what the score asks for. Fixing it means changing the goal, e.g. charging
occluders a reduced keep weight instead of 0, or clipping the target to what
is reachable with the occluder partly present. That is a task-definition
decision and would change every reported number.

## Recommendation

Leave `KEEP_REPAIR_OCCLUDERS` off. It is safe (never worse than keep by more
than eps, small significant gains on the plan answer on val), but it does not
change the viewer's final answer and does not touch the case it was meant
for.

## Reproduce

```bash
O=<dir> docs/experiments/2026-10-19-scripts/eval_v3.sh out/rl_mb_v3c/clb_r3_seed$S/checkpoint.pt occ_s$S $SPLIT --plan --keep --keep-occ
O=<dir> .venv/bin/python docs/experiments/2026-10-19-scripts/summarize_v3.py occ_s0 $SPLIT occ_s1 $SPLIT occ_s2 $SPLIT
```
