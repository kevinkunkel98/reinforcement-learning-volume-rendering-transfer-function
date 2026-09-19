# Reproducing every number and figure

Every command below was run on this machine (Apple M1 Pro, macOS 26.6, Python
3.14) against `.venv`. Timings are what they actually took, not estimates. Run
everything from the repository root.

Prefix each command with the project interpreter:

```bash
.venv/bin/python -m <module>
```

The pipeline has five stages. Each one's output is the next one's input:

```
data  ->  visibility cache  ->  training  ->  evaluation  ->  figures & tables
```

---

## 0. Environment

```bash
.venv/bin/python -m pytest -q        # ~60 s, everything must pass
```

The suite is the fastest check that the environment is intact: it exercises the
transfer function, the visibility estimate, the goal encoding, the environments,
the parsers, the server and the figure scripts.

---

## 1. Data

Two datasets are involved:

| what | where it comes from | size |
|---|---|---|
| Slicer sample CTs (`ct_chest`, `ct_abdomen`, `mri_head`, …) | downloaded on first use, cached under `data/` | ~100 MB |
| TotalSegmentator subjects (`ts_s0357` …) | extracted from the TotalSegmentator release zip | ~4.1 GB |

The Slicer volumes need no action — `datasets.load_dataset` fetches and caches
them on first call. The RL work uses the TotalSegmentator subjects, which are
selected once into `data/totalseg/` with a manifest recording the train/val/test
split:

```bash
.venv/bin/python -m tools.select_totalseg --zip ~/Downloads/Totalsegmentator_dataset_v201.zip
```

Writes `data/totalseg/s<id>/` per subject plus `data/totalseg_manifest.json`.
The manifest is the source of truth for splits — `datasets.volumes_for_split`
reads it, and `totalseg.split_names` refuses to run if any listed volume is
missing, so an experiment can never silently train on fewer volumes than it
claims.

**The six test subjects are never trained on.** That is what makes the held-out
table meaningful.

---

## 2. Visibility cache

`visibility.py` resamples each volume once per view into an 80³ cube and stores
it, because doing that per training step would dominate the run. Build it ahead
of time:

```bash
.venv/bin/python -m tools.build_visibility_cache --splits train val test
```

Writes `out/cache/visibility/<volume>-<hash>.npz` (~72 MB total, ~1 min/volume
on first build). Training and evaluation both build it lazily if it is missing,
so this step is an optimisation, not a prerequisite.

**Validate the estimate against real renders** before trusting any number that
comes out of it:

```bash
.venv/bin/python -m tools.validate_visibility --trials 40 --out out/visibility_validation.json
```

Reports, per anatomical class, how well the cheap estimate tracks a full VTK
render. Current agreement is 0.87–0.97 for lungs across the six test subjects.

---

## 3. Training

One seed, the one-shot formulation the thesis reports:

```bash
.venv/bin/python -m rl.oneshot_train --timesteps 150000 --seed 0 --out out/rl_v2/oneshot_v3_seed0
```

~3 h per seed on CPU (~14 environment steps/s; the checkpoint timestamps of the
committed runs span 23:31 to 01:26). Note that `time/time_elapsed` in
`progress.csv` does **not** report total wall clock — read the file timestamps
or `time/fps` instead. Writes into the `--out` directory:

| file | what it is |
|---|---|
| `best.zip` | the checkpoint with the best evaluation attainment |
| `checkpoint_<n>.zip` | periodic snapshots |
| `progress.csv` | SB3 training diagnostics (losses, entropy coefficient) |
| `eval_progress.csv` | attainment on the evaluation episodes, overall and per instruction kind |

The thesis uses three seeds (0, 1, 2). Seeds differ by more than the effects
being measured, so **never report a single seed**.

> **Known limitation:** at 150k steps the runs have not converged — see
> `plots/output/kind_curves.png`, where relative and compound instructions are
> still climbing at the final evaluation. Longer runs are the cheapest available
> improvement.

---

## 4. Evaluation

```bash
.venv/bin/python -m rl.vis_eval \
    --policy out/rl_v2/oneshot_v3_seed0/best.zip \
    --split test --episodes 200 --seed 0 \
    --formulation one_shot --refine 3 \
    --out out/rl_v2/eval_rerun_v3_seed0.json
```

~13 min per seed. Most of that is the 200-evaluation hill-climb baseline, not
the policy, which answers in one forward pass.

The 200 episodes are deterministic in `--seed`, so every method is scored on
*identical* instructions and start states (`rl.vis_eval.fixed_episodes`). Repeat
for seeds 1 and 2, changing `--policy` and `--out` only — the episode seed stays
0 so all three arms are comparable.

Re-print a stored result without recomputing it:

```bash
.venv/bin/python -m rl.vis_eval --show out/rl_v2/eval_rerun_v3_seed0.json
```

### Comparing two training configurations

The v2-against-v3 numbers in the README and in
`docs/experiments/2026-09-16-retrain-after-measurement-fixes.md` are **not** the
table above aggregated differently — they are a paired test over episodes, and
they come from here:

```bash
.venv/bin/python -m tools.compare_runs \
    --a "out/rl_v2/eval_detail_v2_seed*.json" \
    --b "out/rl_v2/eval_detail_v3_seed*.json"
```

It reads the per-episode rows each result file stores under `episodes_detail`
and never re-runs a policy, so the comparison cannot drift from the files the
table is built on.

Two statistics are in play and crossing them inflates the gap:

| statistic | v2 | v3 | gap |
|---|---|---|---|
| median of the three seeds' medians (the table above) | +0.247 | +0.275 | +0.028 |
| median of the seed-averaged per-episode attainment | +0.201 | +0.286 | +0.086 |

The paired Wilcoxon (p = 0.0064) belongs to the second row, because the pairing
is per episode. Quote a gap with the statistic that produced it.

### Provenance — read this before quoting any number

Every result file records the git commit, whether the tree was dirty, and a
fingerprint of the **imported** scoring modules (`goals`, `visibility`,
`rl.vis_eval`, `rl.oneshot_env`), captured at import time rather than at write
time. `--show` prints a `!!! STALE RESULT` banner when that fingerprint no
longer matches the code on disk.

This exists because of a real incident on 2026-09-17: an evaluation batch
started before a scoring fix landed and wrote its files six hours after it, so
the published figures were computed by code that no longer existed. The
corrected numbers were 60 % higher. Result files written before `8ff1f6d` carry
no provenance block and `--show` will say so.

The fingerprint covers the whole of each scoring module, not just the lines that
compute a score, so an edit that cannot change a number still invalidates older
files — adding `episodes_detail` to `rl/vis_eval.py` did exactly that. The check
is deliberately blunt in that direction: it would rather send you back to a
re-run than let a changed module pass unnoticed. When a re-run reproduces the
old numbers exactly, that is the evidence the edit was cosmetic; record it and
move on.

---

## 5. Figures

```bash
.venv/bin/python -m plots.held_out_results      # frontier, reliability, per-kind curves
.venv/bin/python -m plots.training_curves       # SAC diagnostics for the latest run
```

Both write PNGs into `plots/output/` (gitignored — regenerate rather than
commit). `plots.held_out_results` reads the evaluation result files and never
recomputes a score, so a figure cannot disagree with the table it illustrates.

---

## 6. The application

```bash
.venv/bin/python server.py                      # http://127.0.0.1:8000
```

Two pages on one server:

| URL | what it is |
|---|---|
| `/` | the viewer: speak or type an instruction, watch the transfer function change |
| `/collect` | the preference collector: two candidate renders, judge which better answers the instruction |

The viewer's defaults are the **LLM parser** and **exact application**. The LLM
parser needs Ollama:

```bash
ollama serve
ollama pull qwen2.5:7b
```

If Ollama is unreachable the parser falls back to the rule parser, and the step
records `parser_used` so the interface can say which one answered.

Switch answering modes in the toolbar: **exact** (apply the parsed command
directly), **search** (hill-climb), **policy** (the trained checkpoint at
`server.POLICY_PATH`). Exact is the default because the policy does not isolate
tissues well — on "show only bones" exact gives 16.7 % skeleton visibility
against the policy's 4.9 %.

---

## 7. Preference judgments

Judgments land in `out/vis_preferences.jsonl`, one JSON object per line,
appended immediately — stopping and resuming is safe.

```bash
.venv/bin/python -m tools.preference_agreement        # metric vs human agreement
.venv/bin/python -m tools.merge_preferences a.jsonl b.jsonl --out merged.jsonl
```

Rows carry an `assisted` flag marking judgments made while an assistant
commented on the pair. Those are excluded from every evaluation set:

```bash
.venv/bin/python -m tools.flag_assisted_rows --start 2026-09-17T13:00:00 --end 2026-09-17T17:00:00
```

**Back the file up first** (`cp out/vis_preferences.jsonl out/vis_preferences.jsonl.bak`)
and stop the server, since the tool rewrites the file the server appends to.

The 54 rows collected on 2026-09-17 are flagged as a pilot. The clean evaluation
set is everything collected after that.

---

## 8. The numbers in the thesis, end to end

To regenerate the held-out table from nothing but data and code:

```bash
# 1. three training runs, ~3 h each
for s in 0 1 2; do
  .venv/bin/python -m rl.oneshot_train --timesteps 150000 --seed $s \
      --out out/rl_v2/oneshot_v3_seed$s
done

# 2. three evaluations, ~13 min each
for s in 0 1 2; do
  .venv/bin/python -m rl.vis_eval --policy out/rl_v2/oneshot_v3_seed$s/best.zip \
      --split test --episodes 200 --seed 0 --formulation one_shot --refine 3 \
      --out out/rl_v2/eval_rerun_v3_seed$s.json
done

# 3. figures
.venv/bin/python -m plots.held_out_results
```

Expected, median over the three seeds (200 instructions, six unseen patients):

| method | evaluations | median attainment | improved |
|---|---|---|---|
| hill-climb (thorough) | 200 | +0.730 | 100 % |
| policy + 3 refinements | 4 | +0.316 | 76 % |
| policy alone | 0 | +0.275 | 73 % |
| hill-climb (cheap) | 10 | +0.263 | 93 % |
| do nothing | 0 | 0.000 | — |
| rule executor | 0 | −0.022 | 37 % |
| random | 0 | −0.040 | 39 % |
| occlusion heuristic | 0 | −0.191 | 30 % |

Seed-to-seed spread on the policy is ±0.04, so treat differences smaller than
that as noise.
