# Status — 2026-09-18

MVP due in 4 days. The software is built and the thesis result is stronger than
it was yesterday, because yesterday's numbers were measured with broken code.
What is missing is presentation, not results.

## Where things stand

| | state |
|---|---|
| Measurement (`visibility.py`) | validated against real renders, all five classes |
| Policy | 3 seeds on the corrected pipeline (`oneshot_v3_seed{0,1,2}`) |
| Held-out evaluation | re-measured, 200 episodes, 3 seeds, baselines B0–B5, provenance recorded |
| Language → goal → policy | works end to end in the viewer; LLM parser default |
| Figures | frontier, reliability, per-kind curves, qualitative before/after |
| Preference collection | running at `127.0.0.1:8000/collect` |
| **Clean preference judgments** | **0** — the 54 collected are quarantined as a pilot |
| Tests | 602 passing |

## Results (re-measured 2026-09-17)

200 instructions, six unseen patients, median over three seeds:

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

**The claim:** the policy answers with no evaluations at the level of a
hill-climber allowed ten, and beats every non-search baseline at p < 0.001.
Search remains more reliable (93 % vs 73 % of instructions improved). Per seed
the paired test against cheap search splits — seed 0 favours search
(p = 3.0e-04), seeds 1 and 2 favour the policy (p = 0.054, p = 0.029) — so
"matches" is defensible and "beats" is not.

Per instruction kind (median, seed 1): absolute +0.50, brightness +0.48,
relative +0.30, show-only +0.20, **compound +0.09**. Compound instructions are
25 % of the mix and the policy essentially fails them — the clearest capability
gap, and the one a user notices first.

## What happened yesterday, and what it cost

The published figures came from an evaluation batch that **started before the
ceiling fix in `62b5720` landed and wrote its files six hours after it**, so it
scored with the code it had imported. Confirmed by checking out the pre-fix
commit and reproducing the stored numbers exactly.

Two published conclusions were wrong:

1. Policy attainment was +0.169; it is +0.275. Every baseline moved up with it,
   which is what identifies the fault as the ruler rather than the method.
2. "Retraining changed nothing, p = 0.85" is reversed: v2 +0.201 against v3
   +0.286 seed-averaged median, **paired Wilcoxon p = 0.0064**, v3 ahead on 59 %
   of episodes, with the gain concentrated in absolute instructions (+0.122) —
   the ones that read the ceiling channel `62b5720` fixed. The experiment doc's
   conclusion that the channel "earns less of its place than assumed" is
   withdrawn in an addendum.

`provenance.py` now records the git commit and a fingerprint of the *imported*
scoring modules in every result file, captured at import time. `rl.vis_eval
--show` prints a staleness banner when that fingerprint no longer matches disk.

## Today

### 1. Figures are done — review them

```bash
python -m plots.held_out_results      # frontier, reliability, per-kind curves
python -m plots.qualitative           # before/after renders, VTK
```

`plots/output/` is gitignored; regenerate rather than commit. The per-kind
curves are the most informative: relative and compound start deeply negative and
are **still climbing at 150k steps**, which is the visual evidence that no run
has converged.

### 2. Write-up

`docs/rl-v2-pipeline.typ`, the README, this file and the slides all carry the
corrected numbers. `docs/REPRODUCE.md` has every command from raw data to the
table. The methods chapter has a real story: four defects found by the
validation apparatus, plus a measurement incident the apparatus eventually
caught itself.

### 3. Rate, whenever there is a spare twenty minutes

The clean evaluation set starts at zero. Nothing about the MVP depends on it,
but every judgment shortens the improvement month.

## Cheapest improvements, in order

1. **Train longer.** No run has converged. No new code, ~3 h per seed.
2. **Fix compound instructions**, or train the multi-step formulation where two
   constraints do not have to be satisfied in one shot.
3. **Distil search into the policy** — supervised pretraining on hill-climb
   solutions before RL. Standard way to close an amortisation gap.
4. **Make `visibility.py` differentiable.** The index cube is constant with
   respect to the transfer function; only `transfer_tables` needs porting to
   torch. Gives an analytic ceiling, a teacher for distillation, and an honest
   new baseline.

## Known issues

- **`plots/training_curves.py` defaults to `out/rl_logs`**, the retired
  pipeline's directory. Pass `--run-dir out/rl_v2/oneshot_v3_seed0` or fix the
  default before regenerating that figure for the thesis.
- **The "policy, no ceiling input" ablation row** was dropped from the pipeline
  document: it was measured in the stale batch and has not been re-run.
- **`evaluate.objective`** has branches for camera, compound, width, brightness
  and show-only that no production path reaches — only the opacity branch is
  live. Deliberate forward contract or dead code; needs a decision.
- **Anchor-pool and image caches** under `out/cache/` are serialisation
  contracts across raters. Do not change their key formats without invalidating
  them deliberately.
