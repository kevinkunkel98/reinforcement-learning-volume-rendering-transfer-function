# Status — 2026-09-17 (updated 21:50)

MVP due in ~4 days. The software is essentially built; the thesis argument
depends on data that does not exist yet.

## Where things stand

| | state |
|---|---|
| Measurement (`visibility.py`) | validated against real renders, all five classes |
| Policy | 3 seeds trained on the corrected pipeline (`oneshot_v3_seed{0,1,2}`) |
| Held-out evaluation | done, 200 episodes, 3 seeds, baselines B0–B5 |
| Language → goal → policy | built; LLM parser measured for the first time |
| Preference collection page | built, multi-rater, blind, running at `127.0.0.1:8000/collect` |
| **Preference judgments collected** | **0 rows** ← the critical path |
| Tests | 530 passing |

## Results

### Held-out attainment (test split, 200 instructions, 6 unseen patients)

Median over three seeds:

| Method | Evaluations | Median attainment | Improved |
|---|---|---|---|
| hill-climb (thorough) | 200 | +0.730 | 100 % |
| **policy + 3 refinements** | **4** | **+0.316** | 76 % |
| **policy alone** | **0** | **+0.275** | 73 % |
| hill-climb (cheap) | 10 | +0.263 | 93 % |
| do nothing | 0 | 0.000 | — |
| rule-based executor | 0 | −0.022 | 37 % |
| random | 0 | −0.040 | 39 % |
| occlusion heuristic | 0 | −0.191 | 30 % |

The claim that survives: the policy answers with **no evaluations at all** at the
level of a hill-climber allowed ten, and beats every non-search baseline at
p < 0.001. Search is still better and more reliable (93 % of instructions
improved against 73 %); the argument for a policy is cost per instruction, not
peak quality.

Per seed, the paired comparison against cheap search splits — seed 0 favours
search (p = 3.0e-04), seeds 1 and 2 favour the policy (p = 0.054, p = 0.029) — so
"matches cheap search" is defensible and "beats" is not.

### Four defects found and fixed

| commit | defect | consequence |
|---|---|---|
| `cc167af` | policy's colour action wrote one scalar to r, g and b | every policy render grey → preference pairs were **not blind** |
| `a0b009a` | instructions sampled from label presence, not reachability | raters shown unanswerable items |
| `62b5720` | reachable-ceiling probe sat on the retired band layout's fat peak (−100 HU, not −800) | **the policy's own observation** was wrong for lungs; lungs failed render validation |
| `40895be` | viewer started from a different peak layout than the policy trained on | "more lungs" in policy mode adjusted the fat peak |

### The published +0.194 was inflated

Episodes did not change (B3/B4/B5 score bit-identically before and after), so
the same checkpoint on the same 200 episodes decomposes cleanly:

| | median |
|---|---|
| as originally published | +0.194 |
| ceiling measured at the right peak | +0.167 |
| colour no longer collapsed to grey | +0.158 |
| retrained on the corrected observation | +0.169 |

### Retraining did help — the earlier null was a stale measurement

Reported yesterday as **+0.1732 vs +0.1743, p = 0.85, no detectable change**.
That batch ran on pre-fix scoring code: the job started before `62b5720` landed
at 22:48 and wrote its files at 03:26, carrying numbers from the code it had
imported. Re-measured on the corrected pipeline, same checkpoints, same
episodes:

| | seed-averaged median | per-seed |
|---|---|---|
| v2 (pre-fix observation) | +0.2008 | 0.193 / 0.247 / 0.249 |
| v3 (corrected observation) | **+0.2863** | 0.231 / 0.307 / 0.275 |

Paired Wilcoxon **p = 0.0064**, v3 ahead on 59 % of episodes. The gain is
largest on absolute instructions (+0.122), which are exactly the ones whose
targets come from the reachable ceiling that `62b5720` fixed — mechanism and
measurement agree.

The reading recorded yesterday (that the ceiling channel earns less of its place
than assumed) is **withdrawn**: it earns its place, and the stale ruler could not
see it. See the addendum in
`docs/experiments/2026-09-16-retrain-after-measurement-fixes.md`.

Caveat: three checkpoints per arm, and the p-value is a paired test over 200
episodes rather than over training runs.

### Lungs now validate

Previously reported as unvalidatable, "below the renderer's noise floor" — that
was the wrong probe peak, not the renderer.

| volume | lungs | others |
|---|---|---|
| ts_s1245 / ts_s0425 / ts_s0407 | 0.974 / 0.972 / 0.965 | 0.89–1.00 |
| ts_s0811 / ts_s0357 | 0.890 / 0.869 | |
| ts_s0363 | 0.689 (only failure) | |

### LLM parser, measured for the first time

Plan 8's evaluation step had been skipped because Ollama wasn't running.

| parser | score on 21 free-form phrases |
|---|---|
| rule | 0 / 21 |
| qwen2.5:7b as found | 14 / 21 |
| qwen2.5:7b after fixes | **20 / 21** |

Most of that jump was measurement error removed, not capability added: no
temperature was set (default 0.8, so results weren't reproducible), and four
"failures" were correct answers wrapped in a single-element compound.

---

## Today

### 1. Collect preference judgments — everything else is secondary

```bash
python server.py    # then http://127.0.0.1:8000/collect   (already running)
```

- Target ≥ 300 for a usable reward model; 2000 later with supervisors.
- ~2.4 s per item to generate, so budget thinking time plus that.
- Keys: **A** / **B** / **E** equal / **S** skip. Prefer **E** over guessing —
  a forced coin-flip is noise the reward model will try to fit.
- Rows append to `out/vis_preferences.jsonl` immediately; stopping and
  resuming is safe.

This is the only task that cannot be parallelised, accelerated, or done at the
last minute. Zero rows exist.

### 2. Known rough edge, decide before collecting far

Pairs where **both** candidates fail the instruction still get shown. Skipping
them is correct but burns attention. Filtering so at least one candidate
substantially achieves the goal is ~30 min of work. Worth doing first if the
skip rate feels high in the first 20 items.

### 3. If the cluster is available

Run more seeds (10 rather than 3) while you rate. Needs only:

- `out/cache/visibility/` (72 MB) and `data/totalseg_manifest.json` — **not**
  the 4.1 GB of CT data, and **not** VTK
- watch out: Python 3.14 CUDA wheels are thin; test a 2000-step job first

This would settle the lung trend (p = 0.081) one way or the other.

## Backlog — after the MVP

- Figures: learning curves, per-kind bar chart, qualitative before/after renders
- `docs/rl-v2-pipeline.typ` tables still carry pre-fix numbers (README is updated)
- Held-out phrase set for the parser — the 21 phrases are now a development
  set, so 20/21 is not a publishable figure
- `qwen2.5:32b` comparison — a real latency/accuracy trade-off for VR speech
- DINOv2 → Bradley-Terry reward model → RLHF fine-tune → blind A/B
- Candidate-generation filter, chest-CT re-selection

## For the write-up

The four defects are good methods material, not an embarrassment: the
validation apparatus caught problems that would have invalidated the preference
data, and the pre-registration makes the null result defensible. That is a
stronger methods chapter than a clean run with no story.
