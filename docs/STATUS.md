# Status — 2026-10-19, updated 2026-10-23

Snapshot after the switch to model-based RL, task v2 and v3, the instruction
parser work, planning through the learned model, and the second VR lab
session. Submission: January 2027. The previous version of this file
(2026-10-14, SAC era, E1–E5) is in git history (`git show ab5be0b:docs/STATUS.md`);
its open problems are resolved or superseded as listed under "Since 10-14".

## In one paragraph

A model-based RL policy edits a label-scoped transfer function from a spoken
or typed instruction. On six held-out test patients (task v3, fixed
ceilings, 3 seeds) it closes about half the distance to the goal alone
(0.47–0.48), 0.62–0.65 when its answer is refined by planning through its
own learned model (no renders, ~0.2 s), 0.77–0.78 with 30 renders of search
on top — level with 200 renders of search from scratch (0.70, p = 0.14–0.9)
— and 0.90–0.92 with 200. Model-free SAC never got past 0.09. The viewer
runs the same pipeline on desktop and in a standalone VR headset at 57–58 fps
head-on.

## Since 10-14

| Date | What | Write-up |
|---|---|---|
| 10-15 | VR lab session 1: custom WebGL2 ray marcher, vtk.js WebXR bugs fixed, 27–33 fps lit | `experiments/2026-10-15-vr-lab-session.md` |
| 10-15 | Diagnosis of the `relative` regression: 55 % of relative instructions unsolvable (class invisible at the start) | `experiments/2026-10-15-relative-diagnosis.md` |
| 10-16 | **Model-based RL replaces SAC**: learned renderer ensemble, policy trained through it, Dyna rounds; v1 0.23 vs SAC ≤ 0.09 (p = 7e-14) | `experiments/2026-10-16-overnight-results.md` |
| 10-17 | **Task v2**: visible-organ start, reachable targets, occluders may fade; 500-eval search ceiling 0.49 → 0.82. Policy 0.50 test, policy + 200 0.90. Viewer bug: residual actions had been applied as absolute | `experiments/2026-10-17-task-v2.md` |
| 10-18 | Instruction parser: rule grammar 40 → 100 % (dev), LLM 66 → 88 % exact on held-out phrases | `experiments/2026-10-18-instruction-parser.md` |
| 10-19 | **Task v3**: unlabelled shell at 0.2 opacity, brightness only for visible classes, show-only hides fully, renders were upside down (camera). 2–3× the on-screen change of v2 | `experiments/2026-10-19-task-v3.md` |
| 10-19 | **Ceiling fix** (found by rendering): soft tissue's ceiling was probed at full peak height only (0.2–2.3 % instead of 4–22 %); retrained (`out/rl_mb_v3c`); the val→test drop disappeared | same, "Debugging with renders" |
| 10-19 | **Planning through the learned model** at use time; viewer default (`POLICY_PLAN=1`); seed 1 of v3c ships | same, "Results after the ceiling fix" |
| 10-19 | VR lab session 2: 21–22 → 57–58 fps head-on, 84–88 rotating, same resolution; volume placed in front of the viewer; VR panel lists per-class changes | `experiments/2026-10-19-vr-lab-session.md` |
| 10-21 | VR lab session 3: 1 Hz panel upload was the head-motion stutter; adaptive controller fixed; surface-weighted lighting; dynamic volume resolution, sharp panel at `xrscale=1` | `experiments/2026-10-21-vr-lab-session.md` |
| 10-21 | **Voice lexicon** (`data/lexicon.json`): one vocabulary for Whisper's prompt, normalization of ASR errors, rule parser and LLM prompt/schema; hybrid parser (rule when every word is covered, else LLM); Whisper hallucination filter; per-class reset. Realistic set: rule parser held-out 58 → 94 % (phrases written by the developer) | `superpowers/specs/2026-10-21-voice-lexicon-design.md`, `superpowers/plans/2026-10-21-voice-lexicon.md` |
| 10-22 | Voice end to end over 36 scripted commands × 2 synthetic voices: 29 / 30 of 36 right; latency dominated by the 200-evaluation refine (3 s) | `experiments/2026-10-22-voice-e2e-run.md` |
| 10-23 | **Keep repair** + **two-stage answer**: unmentioned-class brightness drift 0.25 → 0.10 at equal attainment (plan + 200 + keep 0.89–0.91); first image after ~0.7 s (plan + keep, 0.66–0.71), final after ~3.8 s; "no visible change" hint from the render | `experiments/2026-10-23-keep-repair-and-refine-budget.md` |
| 10-23 | Occluders as keep-repair candidates: never changes the final answer — the goal itself asks for the occluder fade; left off | `experiments/2026-10-23-keep-repair-occluders.md` |

The 10-14 TODO list, item by item:

1. *Diagnose `relative`* — done (10-15); resolved by model-based RL and task v2 (relative, test, seed 1: policy 0.44, planning + 30 0.82).
2. *Hybrid routing* — superseded: every command runs policy → planning → search.
3. *VR adaptive sampling* — superseded by the custom ray marcher and the 10-19 levers.
4. *Scale RLHF collection* — **not done**; folded into the user study below.
5. *Held-out test split* — done for every result since 10-17.
6. *GPU / MPS* — MPS measured slower than CPU on the M1; everything stays on CPU (a full v3 pipeline is ~4 h). Cluster access exists (since 10-03) for multi-seed runs.

## Current setup

**Task (v3, `goals.use_task_version("v3")`).** Eight TotalSegmentator classes
plus unlabelled "other"; one peak per class, label-scoped shading,
unlabelled tissue at 0.2 opacity. Instructions: relative, absolute,
compound, show only, brightness. Per-class visibility measured on a fixed
grid (`visibility.py`); goals in log-visibility; attainment = share of the
start distance closed; class ceilings probed over peak heights 0.05–1.0;
occluders of a target may fade.

**Policy (`rl/model_based.py`, `rl/closed_loop.py`).** Five-MLP dynamics
ensemble on renders of random and policy edits (3000 + 3 × 1500 starts, 90
training patients); policy trained by backpropagating the exact goal
distance through it; three Dyna rounds, balanced closed-loop fine-tuning.
At use: planning (100 Adam steps, 16 restarts, β = 1) through the ensemble.
Checkpoint `out/rl_mb_v3c/clb_r3_seed1/checkpoint.pt`.

**Viewer (`server.py`, `static/`).** Voice (faster-whisper) or text → rule
or LLM parser → policy → planning → 200-evaluation search (~5 s) → vtk.js
on desktop, custom WebGL2 ray marcher in WebXR. Headset defaults: custom
renderer, `down=2`, `xrscale=0.8`, `fovstep=2`.

## Results (task v3, fixed ceilings, 300 episodes per split, seeds 0/1/2)

| | val | test |
|---|---|---|
| policy alone | 0.50 / 0.49 / 0.49 | 0.48 / 0.47 / 0.47 |
| + planning (0 renders) | 0.62 / 0.63 / 0.63 | 0.62 / 0.65 / 0.64 |
| + planning + 30 renders | 0.75 / 0.77 / 0.75 | 0.77 / 0.78 / 0.77 |
| + planning + 200 renders (viewer) | 0.89 / 0.90 / 0.90 | 0.91 / 0.92 / 0.90 |
| search from scratch, 200 | 0.75 | 0.70 |

Paired Wilcoxon: planning > policy (p ≤ 2e-7), planning + 30 > policy + 30
(p ≤ 3e-5), planning + 30 = search 200 from scratch (p = 0.14–0.9). Weakest
kind: show only (planning + 30: 0.48–0.56). Earlier tasks (v1, v2) are
reported side by side in the experiment docs, never as one number improving
on another.

## Strengths

- **The core question has a clear answer:** RL can learn this — model-based
  yes, model-free no — on patients it has never seen, with paired tests over
  three seeds and a test split evaluated once.
- **Every task change is motivated by a measurement** (search ceilings, the
  image share of "other", the ceiling probe), and renders are inspected,
  which is how the ceiling bug was found.
- **A working end-to-end system**, voice to VR, with the frame rate measured
  on the headset rather than estimated.
- **Honest claims:** "level, not better" where the test says level.

## Weaknesses and risks

- **The objective is self-designed and not yet validated by people.** No
  user study or preference data on the current rendering; the 54 pilot
  pairs from 09-17 predate label-scoped rendering. This is the largest gap
  for the thesis.
- **Small evaluation set:** 4 val / 6 test patients, CT only, one scanner
  source. The task was revised several times while looking at val.
- **Weak spots that renders show:** "increase X strongly" can stay timid
  (policy alone 0.14 on one liver case); "a bit more" can overshoot when the
  search clears an occluder; brightness on organs below ~2 % of the image is
  invisible even when scored 0.9+; occluders fade as a side effect.
- **Parser held-out sets written by the developer** (base and realistic,
  `data/parser_eval_realistic.json`) — optimistic until phrases from other
  people are tested; the LLM model choice (qwen2.5:7b vs qwen3.5:4b,
  gemma4:e4b, granite4:3b, compared 10-22 without them) waits on those.
- **Nothing from 10-21 to 10-23 has been tried in the headset**: two-stage
  answer, "Understood: …" / "no visible change" panel lines (long lines may
  be cut), voice path latency (extrapolated from loaded runs).
- **VR numbers come from one scan, one sitting**, and depend ~3× on the TF.
- **Drift accumulates over a session** (found 2026-10-22, live VR session on
  ts_s0477). Each policy step keeps unmentioned classes within
  `KEEP_TOLERANCE` (0.15) of *that step's* start, so a single step looks
  clean, but replaying seven commands drifted never-mentioned classes far
  beyond it: heart brightness −0.36 (renders black), vessels +0.46, the
  unlabelled shell +0.37 log-visibility (a bluish box with vertical streaks).
  The metric has no anchor to the session's origin, and later steps start
  from states the one-shot policy never trained on. Mitigation in progress:
  an inference-time keep repair (see Next).

## Next, in order

1. **User study / preference collection** (needs people, best in the lab):
   rate before/after renders for policy + planning + search vs. search vs.
   rule executor; report agreement and the correlation with attainment.
   `/collect` is ready for it (v3 renders, these conditions as candidate
   sources, rows in `out/vis_preferences_v3.jsonl`); open: items take
   ~5–9 s, so prefetching the next item would help raters.
   Frame it as a human evaluation of the objective, not as RLHF.
2. **Preference model on the study data** (pending, needs 1's data): logistic
   regression on the judged pairs (equal/skip dropped, sides as recorded):
   P(rater picks A) from the per-class differences A − B in log-visibility
   and brightness relative to the goal (all in the rows' `features`), plus
   the instruction kind. Baseline: attainment difference alone. Compare
   held-out log-loss and accuracy (cross-validated by rater, and by patient),
   against the inter-rater agreement on the anchors as the ceiling. If the
   per-class model beats attainment, its weights show where the objective
   diverges from people (e.g. keep-term weight, small organs, occluder
   fading) → discussion chapter, and the bridge to RLHF as future work. If
   not, that supports the objective as designed.
3. **Scripted VR benchmark** over 2–3 patients × 2 TFs with fixed placement.
4. **Missing evaluation pieces for the thesis:** render-budget curve
   (10/30/100/200) on v3c with and without planning; per-stage training
   ablation; per-patient table; planning ablation re-run on v3c. Plus the
   two ablations that answer "why not supervised / why not just gradients"
   (pending):
   - **Planning from zero vs from the policy** (~1 h): `rl.model_based.plan`
     started from a zero residual instead of the policy's action, same steps
     and restarts, same 300 val/test episodes, paired Wilcoxon. If planning
     from the policy wins clearly, the policy earns its place even in a pure
     gradient-planning pipeline.
   - **Behaviour cloning on v3c** (training run): same network and data
     budget, labels from 200-eval search, same eval protocol. The only BC
     evidence is from the SAC era on v1 (negative even as MSE fell -- several
     valid answers averaged); this makes it a current ablation.
   Also: report wall-clock time next to render counts (planning costs ~0.2 s,
   about 8 renders), and cite differentiable volume rendering (DiffDVR, Weiss
   & Westermann 2021) in related work as the per-query gradient alternative.
5. **Multi-step sessions** (keep repair done 10-23, retraining optional):
   - *Keep repair (done):* the viewer default; report "policy + keep" and
     "plan + 200 + keep" next to the other arms
     (`experiments/2026-10-23-keep-repair-and-refine-budget.md`).
   - *Open thesis decisions* (each changes every reported number):
     occluders at weight 0 let "more lungs" erase soft tissue completely —
     a reduced occluder weight instead
     (`experiments/2026-10-23-keep-repair-occluders.md`); vessel goals exist
     only on "angiography" scans although e.g. `ts_s0477` is contrast-enhanced
     (vessels median 127 HU) — define contrast by measured vessel HU.
   - *Retraining (possible feature):* train on multi-step episodes (starts
     drawn from the policy's own earlier outputs, not only the task's start
     TF) and charge drift against the episode origin, with a smaller
     `KEEP_TOLERANCE` and a brightness keep term for the unlabelled shell
     ("other" is charged for visibility only today). Only if the keep repair
     does not close the gap -- it changes the objective, so every reported
     number would have to be re-run.
6. **Write the thesis** — skeleton with tables and figure briefs exists
   (Overleaf), plus an architecture overview figure (`docs/figures/arch.pdf`, made by `docs/figures/make_arch.py`).
7. Optional: start a ceiling-fix-style render audit for the remaining weak
   cases (strong increases, small-organ brightness).
