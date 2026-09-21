# Status — 2026-09-21

Found and fixed a reward-hacking blind spot the same day as the presentation
prep: `goals.distance` never charged a transfer function for rendering
unclassified ("other") tissue, so a search or a policy could satisfy "show
only X" by lighting up material that belongs to none of the five measured
classes instead of X. Fixed, retrained three fresh seeds (`oneshot_v4_seed
{0,1,2}`), re-ran the full held-out protocol. The fix helped the aggregate
numbers; it did not fully solve the failure mode that exposed it. Both halves
of that sentence are below, not just the first one.

## Where things stand

| | state |
|---|---|
| Measurement (`visibility.py`) | validated against real renders, four of five classes; now also reports an "other" (unclassified-tissue) bucket per render |
| Policy | 3 seeds on the reward-corrected pipeline (`oneshot_v4_seed{0,1,2}`); viewer ships seed 2, the strongest |
| Held-out evaluation | 200 episodes, 3 seeds, baselines B0–B5, provenance recorded, re-run under the corrected objective |
| Reproducibility | result files carry per-episode rows; `tools.compare_runs` recomputes seed-to-seed comparisons from them |
| Viewer | four-mode comparison panel and a 20-instruction sweep |
| Language → goal → policy | end to end, LLM parser default |
| Figures | frontier, reliability, per-kind curves, qualitative, SAC diagnostics — **not yet re-rendered for v4**, still reflect v3 |
| **Clean preference judgments** | **0** — the 54 collected are quarantined as a pilot |
| Tests | 669 passing |

## What changed today: the "other" keep term

`visibility.py` measures five anatomical classes, but every voxel the label
volume assigns to none of them — fat, connective tissue, partial-volume edges
— was invisible to the objective: not measured, not penalised, not present
anywhere in `goals.distance`. Driving the viewer's "show only bones" on
`ts_s0477` with `hill_climb` (200 evaluations) reached **full frame
coverage** while every one of the five labelled classes read below 0.001 of
the image — 99% of the rendered frame was material the objective could not
see, and the search still scored it as a strong answer.

Fix: `visibility.py`'s `features()` now reports an `"other"` bucket for
those unclassified voxels, and `goals.distance` charges for it under the same
keep-tolerance any unmentioned class already gets (it can never be *named* by
an instruction, so it is always the unmentioned case). Verified in isolation
before retraining — 5 new tests, full suite green at 669/669 — then three
fresh seeds retrained from scratch under the corrected reward (same
architecture, same 150k timesteps).

## Results — v4, the corrected objective (measured 2026-09-21)

200 instructions, six unseen patients, three fresh seeds. Same protocol as
before: attainment is the median of the three seeds' medians; "improved" is
their mean.

| Method | Evaluations | v3 (old) | **v4 (now)** | Improved |
|---|---|---|---|---|
| hill-climb (thorough) | 200 | +0.730 | +0.692 | 100 % |
| **policy + 3 refinements** | **4** | +0.316 | **+0.371** | 80 % |
| **policy alone** | **0** | +0.275 | **+0.331** | 78 % |
| hill-climb (cheap) | 10 | +0.263 | +0.258 | 93 % |
| do nothing | 0 | 0.000 | 0.000 | — |
| rule-based executor | 0 | −0.022 | −0.022 | 37 % |
| random | 0 | −0.040 | −0.040 | 39 % |
| occlusion heuristic | 0 | −0.191 | −0.198 | 30 % |

**The policy improved (+0.275→+0.331 alone, +0.316→+0.371 with refinement)
without any change to architecture or training budget** — the only change
was what the reward charges for. Thorough search's own number *dropped*
(+0.730→+0.692): part of its old advantage was the same exploit, now
correctly discounted rather than rewarded. That B4 drops the most while the
deterministic baselines (B0–B2, B5) barely move at all is the expected
signature of a reward-hacking fix, not noise.

Per instruction kind (median of the three seeds' medians): absolute +0.511,
brightness +0.561, relative +0.354, show-only +0.166, compound +0.117. Three
of five kinds clearly improved on the old single-seed numbers (absolute
+0.50, brightness +0.48, relative +0.30); **compound moved from +0.09 to
+0.117 — still weak, but not nothing.**

### The honest part: `show_only` itself is not fixed

Re-ran "show only bones" on `ts_s0477` with the strongest new seed (2),
independently of the viewer, straight from the checkpoint:

```
final vis: skeleton=0.0018, lungs=0.00006, soft=0.00093, other=0.604
attainment: 0.289  (was 0.246 with the old checkpoint)
```

Skeleton is still tiny; "other" still dominates 60% of the image. The
attainment gain is from slightly cleaner suppression of lungs/soft tissue,
not from the policy learning to actually raise the named class. `show_only`'s
own median-of-medians (+0.166) sits at or slightly below the old single-seed
figure (+0.20) — the mean, less sensitive to this specific failure, moved
from +0.20 to ~+0.22-0.23 across the three new seeds.

**Conclusion: the reward fix was correct and necessary, and it measurably
helped the aggregate numbers, but one retrain did not teach the policy to
solve `show_only` — that remains open.** Untested hypotheses for next steps:
train longer, raise `LAMBDA_KEEP`, oversample `show_only` episodes during
training. None of these have been tried yet.

## v1–v4, for the record

| Version | Checkpoints | What changed |
|---|---|---|
| v1 | `oneshot_seed{0,1}` | First one-shot checkpoint, replacing an earlier ten-step formulation that never learned. Exploratory — no held-out eval was ever run; superseded within a day. |
| v2 | `oneshot_v2_seed{0,1,2}` | Added the reachable-ceiling (`solo_max`) channel to the observation. Evaluated, but on two undetected bugs. |
| v3 | `oneshot_v3_seed{0,1,2}` | Both bugs fixed: the ceiling was probed at the wrong peak centre, and a colour action collapsed r=g=b. This was the headline checkpoint until today. |
| v4 | `oneshot_v4_seed{0,1,2}` | `goals.distance` now charges for hiding behind unclassified ("other") tissue — a gap v1–v3 all shared. **Current.** |

Full v2→v3 and v3→v4 write-ups with per-kind tables and significance tests
are in `docs/rl-paper.typ` (`@retrain`, `@v4-retrain`).

## The viewer, as of today

**compare** answers one instruction four ways from the same start state —
applied directly, hill-climbed at 10 evaluations (B3) and 200 (B4), and by the
policy — reporting each arm's attainment, evaluation count, wall clock and
per-class effect beside its render. The search arms call
`rl.baselines.hill_climb` and attainment is `goals.attainment`, so the panel
computes the same quantities as the table above rather than a lookalike. It
plots whichever channel the instruction names: a brightness goal barely moves
visibility, so plotting visibility for one would show four near-identical bars
beside attainments ranging from +0.00 to +0.94.

**sweep 20** runs twenty instructions sampled from the same grammar and reports
each method's median attainment and how often it improved on doing nothing. One
comparison is a single draw; the claim above is a median, and the reliability
gap means roughly one instruction in four has the policy not improving.

Measured on `ts_s0477`, a held-out test subject, reset to the default transfer
function before every measurement, **with the v4 seed-2 checkpoint**:

| instruction | exact | search·10 | search·200 | policy |
|---|---|---|---|---|
| more bone | −0.001 | +0.348 | +0.582 | **+0.541** |
| brighten the skeleton | +0.025 | +0.000 | +0.941 | **+0.775** |

| sweep seed | exact | search·10 | policy |
|---|---|---|---|
| 0 | −0.209 (37 %) | +0.342 (100 %) | **+0.473 (90 %)** |
| 1 | −0.263 (22 %) | +0.387 (100 %) | **+0.443 (95 %)** |
| 2 | +0.041 (60 %) | +0.384 (100 %) | **+0.352 (90 %)** |

The two single-episode examples read slightly lower than the old checkpoint's
(+0.627/+0.819 → +0.541/+0.775) — expected variance on n=1, not a regression;
the sweep numbers (which average over 20 instructions) are up on every seed
(old +0.425/+0.405/+0.340 → new +0.473/+0.443/+0.352), consistent with the
held-out table above.

Exact and policy spend 0 evaluations and answer in 0 ms; search·10 takes
~140 ms, search·200 ~2.7 s.

## Demo notes

- **Press Reset before measuring anything.** Both panels start from wherever
  the session is, so comparing an instruction you have already applied measures
  from its own result — a different question, with worse-looking numbers.
- **Drive absolute and brightness instructions for a "policy works" demo** —
  both are strong and improved further under v4. Avoid leading with `show only`
  — it's the one category the reward fix didn't resolve (see above).
- **Vessels only work on `ts_s1379`.** Every named demo dataset (`ct_chest`,
  `ct_skull`, `ct_cardio`, `ct_abdomen`, `mri_head`, `stag_beetle`) has no
  TotalSegmentator labels at all, so `goal_classes_for_volume` never offers
  vessels as a goal there regardless of what the scan actually shows — this
  includes `ct_cardio`, whose name suggests otherwise. `datasets.py`'s
  `OUT_OF_SOURCE_CT` names exactly these four as a generalization split that
  has never actually been evaluated.
- **Avoid "a bit less soft tissue"** (policy −2.8 on two subjects) and compound
  instructions (median −0.25 in the sweep).
- **Four of six test subjects have no lungs.** Lung instructions are correctly
  refused there; that is not a bug.

## Defects found and fixed this weekend (2026-09-20)

- The viewer's **search mode was not the search the thesis measures** — a
  bespoke coordinate stepper over `evaluate.objective` rather than
  `rl.baselines.hill_climb` — and it only engaged for opacity commands, so
  "show only the lungs" with search on silently did nothing.
- The **policy fallback died on a missing visibility cache** instead of
  degrading to exact application.
- Every **`ts_session` test ran against the intensity-only visibility model**,
  because the fixture patched `totalseg.subject` but not the `_subjects`
  chokepoint that `has_labels` reads.
- **Visibility models were rebuilt on every call**, discarding `solo_max`'s
  memo and repaying ~190 ms each time.
- The **comparison panel timed the render, not the arm**, which made the first
  press of a session report the policy as slower than cheap search.

## Cheapest improvements, in order

1. **Actually solve `show_only`.** The reward no longer rewards the wrong
   thing, but the policy hasn't learned the right thing either — try more
   training steps, a larger `LAMBDA_KEEP`, or oversampling `show_only`
   episodes, and check per-episode renders, not just the attainment number.
2. **Fix compound instructions** — still the second-clearest capability gap
   (+0.117), the one a user notices first after show-only.
3. **Distil search into the policy** — supervised pretraining on hill-climb
   solutions before RL. The standard way to close an amortisation gap.
4. **Make `visibility.py` differentiable.** The index cube is constant with
   respect to the transfer function; only `transfer_tables` needs porting to
   torch. Gives an analytic ceiling, a teacher for distillation, and an honest
   new baseline.
5. **Re-render the figures for v4** — frontier, reliability, per-kind curves,
   qualitative and SAC diagnostics all still reflect v3.

## Known issues

- **The "policy, no ceiling input" ablation** was never validly run: the stored
  result file is byte-identical to its own control, and no ablation switch has
  ever existed in the code. Withdrawn rather than repaired.
- **`evaluations` in the comparison panel is the budget granted**, not what
  search spent — `hill_climb` can stop early, so B4 reports 200 where it spends
  about 193.
- **The per-class analysis of which anatomy benefits from retraining** was
  measured in the stale batch and has not been re-run; per-episode rows record
  instruction kind and volume, not goal class.
- **Anchor-pool and image caches** under `out/cache/` are serialisation
  contracts across raters. Do not change their key formats without invalidating
  them deliberately.
- **`hill_climb`'s coordinate descent still finds the same "suppress
  everything, light up unclassified tissue" local optimum on `show_only`
  from the default start and from randomized starts**, even under the
  corrected objective — the fix removes the wrong incentive but doesn't
  guarantee a greedy per-episode local search finds the right one. Not
  addressed: would require its own scoped change to `rl.baselines.hill_climb`
  and re-validation of the B3/B4 budget numbers, which this session
  deliberately left alone.
- **Out-of-source generalization (`ct_chest`, `ct_skull`, `ct_cardio`,
  `ct_abdomen`) has never been formally evaluated**, despite
  `datasets.OUT_OF_SOURCE_CT` and `rl.vis_eval --split out_of_source` existing
  for exactly that. These volumes have no TotalSegmentator labels, so
  `visibility.py` scores them on the coarser intensity fallback, and the
  policy never trained on them. A live spot-check found the policy
  performing at or below "do nothing" on `ct_skull`.
