# Larger held-out splits and a fresh retrain — 2026-11-13 to 11-15

Branch `fix/occluder-colour-and-lexicon`. Why: the test split had 6
patients (val 4), so "works on patients it has never seen" rested on six
samples; the 300 episodes per split are not independent. This adds 92 new
patients and reports statistics over patients, not episodes.

## Setup

- **New splits** (`325e34f`, `tools/extend_eval_splits.py`): `val_extra`
  (32 patients) and `test_extra` (60) from the full TotalSegmentator v2
  (1228 CTs), same region filter as the original 100, no overlap with them.
  10 episodes per patient (4–17 after the fixed episode draw): 320 and 600.
- **Frozen model, evaluated once** (11-13): the viewer's checkpoint,
  v3c seed 1 (`models/v3c_seed1`), unchanged code, all methods.
- **Fresh retrain** (11-14, `out/rl_mb_v3d`): seeds 0 and 2 of v3c live
  on the M1 only, so the whole pipeline ran again on the Leipzig PC
  (`2026-11-14-scripts/overnight.sh` → `2026-10-19-scripts/pipeline_v3.sh`,
  CPU, 8 h 10 against 4 h 08 on the M1: collection has 4 workers). New
  data, new seeds 0/1/2: **v3d is a second model set, not v3c.** The
  viewer keeps v3c seed 1.
- **Statistics** (`2026-11-12-scripts/patient_stats.py`): the median per
  patient first, then the median and a bootstrap 95 % CI over patients;
  paired Wilcoxon on the per-patient medians.
- **Search from the start (`cold200`) is run once**: `hill_climb` has no
  randomness and does not use the policy, and the episodes depend only on
  split and index, so the v3d runs copy it from the v3c run
  (`2026-11-14-scripts/merge_cold.py`, checked by volume and kind; removing
  and re-merging it on the v3c files gives the same summary). That and
  12 workers cut the evaluation to ~53 min per 320 episodes.

## Result

Median over patients of the per-patient median attainment:

| | val_extra (32) v3c s1 | v3d s0 / s1 / s2 | test_extra (60) v3c s1 | v3d s0 / s1 / s2 |
|---|---|---|---|---|
| policy alone | 0.48 | 0.45 / 0.47 / 0.47 | 0.50 | 0.48 / 0.48 / 0.47 |
| + planning (0 renders) | 0.61 | 0.65 / 0.61 / 0.61 | 0.63 | 0.62 / 0.61 / 0.63 |
| policy + 30 | 0.64 | 0.64 / 0.63 / 0.64 | 0.66 | 0.65 / 0.65 / 0.65 |
| **+ planning + 30** | **0.74** | **0.75 / 0.77 / 0.76** | **0.77** | **0.73 / 0.73 / 0.75** |
| policy + 200 | 0.89 | 0.86 / 0.87 / 0.86 | 0.90 | 0.90 / 0.90 / 0.89 |
| + planning + 200 (viewer) | 0.93 | 0.93 / 0.92 / 0.92 | 0.91 | 0.91 / 0.92 / 0.91 |
| search 200 from the start | 0.74 | (same) | 0.75 | (same) |

95 % CIs over patients are ±0.04–0.08 on val_extra and ±0.03–0.06 on
test_extra (full output in `out/eval_extra_*.log`, `out/rl_mb_v3d/eval_d*.log`).

Paired Wilcoxon over patients, all four models:

| | val_extra | test_extra |
|---|---|---|
| planning > policy alone | p ≤ 1.2e-5 (27–29 / 32 higher) | p ≤ 5.9e-7 (49–52 / 60) |
| planning + 30 > policy + 30 | p ≤ 2.6e-4 (26–27 / 32) | p ≤ 1.1e-6 (47–50 / 60) |
| planning + 30 vs search 200 from the start | p = 0.23–0.8 (16–20 / 32) | p = 0.19–0.7 (25–32 / 60) |
| planning + 200 > search 200 from the start | p ≤ 1.6e-6 (30–31 / 32) | p ≤ 1.9e-10 (54–56 / 60) |

## What it shows

- **The claims hold on 92 new patients, now over patients:** planning
  helps the policy, planning + 30 renders is level with 200 renders of
  search from the start, planning + 200 beats it on 90–97 % of patients.
- **The numbers match the 6-patient test** (policy 0.47–0.48, planning
  0.62–0.65, + 30 0.77–0.78, + 200 0.90–0.92): the small split was not
  lucky.
- **The pipeline reproduces:** a retrain from scratch on other hardware
  lands within ±0.04 of v3c on every arm; the three v3d seeds lie within
  0.04 of each other (mostly within 0.02).
- **Level, not better:** on test_extra the retrained seeds put planning
  + 30 slightly below the search (0.73–0.75 vs 0.75, 25–32 of 60 higher,
  n.s.). Report it as "level".

## Not done

- The keep-repair and colour-guard arms are in the result files
  (`<name>_keep`, `<name>_keep_colour`, drift) but not summarised here.
- No breakdown by region, scanner or contrast yet. The new patients make
  that possible.
- v3c seeds 0/2 were not run on the new splits (the checkpoints are on the M1).
