# Status — 2026-10-14

Comprehensive project snapshot after the 8-class / label-scoped (v9) transfer
function, the E1–E5 experiment series, the WebXR/VR integration, and a long
methodology review session. Presented to the thesis committee 2026-10-14 —
meeting went well. This document is the planning baseline for the remaining
~3 weeks in the Ohio lab (VR hardware access) before returning to Leipzig,
within an overall submission timeline of January 2027.

Superseded by this doc: the 2026-09-21 version of this file (pre-8-class,
pre-E1–E5, pre-VR-measurement). That version's "four classes" results are
kept below for historical/regression comparison only — they are not
re-confirmed and should not be read as current.

## Project setup

**Live app (`server.py`, FastAPI):**
Voice (faster-whisper, local) or text → rule-based or local-LLM (Ollama)
parser → `commands.apply_command` (deterministic, default path) → new
transfer-function params → client-side rendering (vtk.js/WebGL), flat
desktop and WebXR VR, same renderer. A separate `/api/compare` /
`/api/compare/sweep` path additionally runs the trained SAC policy and
`hill_climb` search for demo/comparison; a small offscreen VTK renderer
produces thumbnails for that panel only. `POLICY_PATH` env var swaps
checkpoints without a code change.

**Offline RL training:** TotalSegmentator-labelled CT volumes →
`visibility.py` (measurement, torch-based but currently CPU-only — no
`.to(device)` calls anywhere) + `transfer.py` (8-peak, label-scoped TF) +
`goals.py` (exact distance/attainment metric, instruction sampling,
per-scan `solo_max` ceiling) → `rl.oneshot_env.OneShotEnv` (Gymnasium,
97-dim observation, 24-dim action) → standard `stable-baselines3` SAC
(`SAC("MlpPolicy", ...)`, no custom `net_arch`/`policy_kwargs` — the
project's novelty is in the environment/reward/pretraining design, not the
algorithm) with hindsight relabeling + search-distilled pretraining
(`hill_climb` solutions distilled into the actor before RL starts) →
`rl.vis_eval` / `tools.per_class_eval` / `tools.baseline_report` → 3-seed
adoption gate (median attainment > 0 AND > 50% episodes improved, all 3
seeds) → `out/rl_v4/<run>/best.zip`.

**Transfer function:** 8 Gaussian peaks (one per class: lungs, soft,
liver, kidneys, spleen, heart, vessels, skeleton), 6 params each
(center, width, height, r, g, b) = 48 values. Centers fixed at calibrated
HU values; policy/search only move width, height, colour (24 controllable
groups = 8 × {width, height, brightness}). Label-scoped shading (v9): a
voxel is shaded only by its own segmented class's peak, resolving the
previous confusion of 5 classes (soft, kidneys, heart, spleen, liver) that
all sat at 40–65 HU under the old shared-HU-lookup rendering.

**Goal definition, per instruction kind** (`goals.py`, hand-set constants,
not learned):
- `relative`: target = current value × 10^(±strength), strength ∈
  {slightly: 0.15, moderately: 0.3, strongly: 0.6} (log10 units; e.g.
  "strongly less" ≈ ÷4).
- `absolute`: target = {low: 0.1, medium: 0.4, high: 0.8} × `solo_max`
  (the live-measured, per-volume ceiling of what that class can reach
  under a single-peak TF — makes "high" mean the same relative level on
  every patient regardless of raw HU/anatomy differences).
- `show_only`: named classes → max, all others → min.
- Instructions below `VISIBLE_CEILING` (0.005, i.e. unanswerable even in
  the best case) are not sampled at all.

**Reward:** `R = (progress − λ·drift) / d_start`, where `progress` is the
share of `start_distance` closed and `drift` sums unrequested changes on
every *unmentioned* class. Normalized by `start_distance` so easy and hard
goals are comparable. Action mode is residual (`action = offset from the
episode's own start`, `v9 = v7 + a load-time transfer_mode contract
check`), not absolute.

**Important caveat, discussed and worth keeping in mind:** the exact
metric is a human-designed proxy, not ground truth — the strength
constants above were picked by the developers, not fit to real human
judgment. `tools/preference_agreement.py` exists specifically to check how
often the metric's pick matches a real rater's pick. The drift penalty in
particular is a flat linear sum with no perceptual weighting, so a render
that looks correct to a human can still score low if invisible-scale drift
on unmentioned classes accumulates — a concrete, named risk, not just a
hypothetical.

## Strengths

- **Controlled ablation methodology.** E1→E2→E2+Search→E3 isolates
  exactly what hindsight pretraining and search-distilled pretraining each
  contribute, one variable at a time, 3-seed confirmed.
- **A self-caught and transparently fixed measurement bug.** The
  `show_only` reward-weighting investigation (E3-old-metric → E3-rescored
  → E5-retrained) shows the old win (+0.061) was mostly a scoring
  artifact, not a real capability — found, diagnosed, and reported
  honestly rather than left standing.
- **A real, working end-to-end system**, not a notebook experiment:
  voice → RL/rules → render, on desktop and in an actual VR headset
  (tested live, 2026-10-13/14, HTC Vive Focus 3).
- **Standard RL algorithm (SAC via SB3), unmodified.** Keeps the thesis's
  novelty claim focused on the problem formulation (one-shot amortization,
  label-scoped TF, per-scan-normalized goals, hindsight + search-distilled
  pretraining) rather than an unproven custom algorithm.
- **A genuine, evidence-derived architectural insight**: `hill_climb`
  (no learning) beats the trained policy on `show_only` (+0.196 vs
  -0.037) — motivates routing well-defined-target instructions to search
  and reserving RL for genuinely ambiguous ones (`relative`, `compound`,
  later RLHF-driven preference).
- **RLHF scaling infrastructure already exists**, not just planned:
  `collect.html`/`collect.js` (web UI, servable like the main app),
  `rater_id` already in the data schema, `tools/merge_preferences.py`
  (Cohen's kappa *and* Krippendorff's alpha, hand-implemented since scipy
  isn't available). Only the data volume is missing, not the tooling.
- **VR performance is measured, not guessed.** Live logcat data
  (`VRPerfMgr`) from the actual headset: 0.5×/fine sampling stable at
  88–90 fps / 60% GPU; 0.75× GPU-bound at 99% / 6–7 fps; 1.0×/2× coarse
  sampling 64–69 fps / 92% GPU but blurry. A real hardware ceiling for
  stereo volume ray-marching on an Adreno 650, not a misconfiguration.

## Weaknesses

- **`relative` (40% of all instructions) is negative in every experiment
  of the current 8-class generation (E1–E5), and never diagnosed.**
  Important: it was *not* always bad — the old 4-class system (v4) scored
  +0.354 on `relative`, third-best of five kinds. This is a regression
  tied to the 8-class expansion, not an inherent property of the
  instruction kind. Leading candidates, untested: more drift surface
  (8 vs 4–5 classes to accidentally disturb), or a hindsight-data-coverage
  shift analogous to the brightness blind spot in E2.
- **`show_only` still misses its own adoption gate** after the
  group-balanced reward-weighting fix (E5: 3/3 seeds improved, median
  still -0.037, below zero).
- **`compound` stays weak (+0.088) and got worse under E4** (show_only-
  anchored hindsight diluted its training signal).
- **No held-out test-split evaluation exists for the current (v9/E3/E5)
  checkpoints** — all current numbers are validation-split only, flagged
  explicitly as "not yet done" in `docs/experiments/2026-10-07-current-
  setup-and-baselines.typ`. The old v4 numbers below *were* measured on a
  real held-out test split; today's aren't directly comparable to them for
  that reason too.
- **RLHF data is not usable as-is**: 54 pairs, single rater (`Kevin`),
  collected 2026-09-17 — before the 8-class/label-scoped rendering even
  existed, so the images judged don't reflect current rendering quality.
  Already self-quarantined as "pilot" in the prior status doc.
- **Training currently runs entirely on CPU**, confirmed directly
  (`SAC("MlpPolicy", ...)` has no `device=`; `visibility.py`'s torch calls
  have no `.to(device)`/`.cuda()` anywhere). Not a weak-GPU problem, a
  no-GPU-used-at-all problem. (MPS is available and the specific ops
  `visibility.py` needs — 3D `grid_sample`, `cumprod`, `where` — were
  smoke-tested working on this machine's PyTorch 2.14.0.)
- **Out-of-source generalization was never (re-)confirmed** under the
  current 8-class system. The 2026-09-21 status doc recorded the v4 policy
  at or below "do nothing" on an out-of-source volume (`ct_skull`); that
  finding predates the 8-class work and hasn't been re-run.
- **The drift penalty is a rigid, unweighted linear formula** — it can
  penalize a render that looks correct to a human if small, imperceptible
  changes accumulate across unmentioned classes. A concrete, named
  candidate for what real preference data would likely correct.
- **The goal-definition constants themselves are hand-set, not learned
  or validated**: `VISIBILITY_STRENGTH` (0.15/0.3/0.6) and
  `ABSOLUTE_LEVEL` (0.1/0.4/0.8) were picked by the developers, not fit to
  real human judgment of what "slightly"/"strongly" mean.

## v1–v9, for the record

| Version | What changed |
|---|---|
| v1 | First one-shot checkpoint, replacing a ten-step formulation that never learned (abandoned within a day, no held-out eval ever run — amortizing a single forward pass over (instruction, start state) → TF works where learning a blind iterative search procedure did not). |
| v2 | Added the reachable-ceiling (`solo_max`) observation channel. Evaluated under two undetected bugs. |
| v3 | Both bugs fixed (ceiling probed at the wrong peak centre; a colour action collapsed r=g=b). Headline checkpoint through v4. |
| v4 | `goals.distance` now charges for hiding behind unclassified ("other") tissue — closes a reward-hacking exploit. **4-class era, see table below.** |
| v7 | Residual action mode + `target` reward mode introduced. |
| v8 | Old TF + layer actions (intermediate, superseded by v9). |
| v9 | Label-scoped TF (8 classes, voxel shaded only by its own segmented class's peak) + hindsight pretraining + brightness hindsight + a load-time `transfer_mode` contract check. **Current generation (E1–E5).** |

### v4 results, 4-class era (for historical/regression comparison only — not re-confirmed, different task difficulty, real held-out test split)

| Method | Median attainment | Improved |
|---|---|---|
| hill-climb (thorough, 200 evals) | +0.692 | 100% |
| policy + 3 refinements | +0.371 | 80% |
| policy alone | +0.331 | 78% |
| hill-climb (cheap, 10 evals) | +0.258 | 93% |
| do nothing | 0.000 | — |
| rule executor | -0.022 | 37% |
| random | -0.040 | 39% |
| occlusion heuristic | -0.198 | 30% |

Per-kind (v4): absolute +0.511, brightness +0.561, **relative +0.354**,
show only +0.166, compound +0.117.

### Current (E3/E5, 8-class era, validation split only)

| Instruction kind | Share | Attainment | Status |
|---|---|---|---|
| brightness | 10% | +0.246 | solid |
| absolute | 10% | +0.176 | solid |
| show_only | 15% | +0.096 (E3) / **-0.037** (E5, current) | gate not cleared |
| compound | 25% | +0.088 | weak, worse under E4 |
| **relative** | **40%** | **-0.068** | negative, undiagnosed |

Overall (median across kinds, 3 seeds): **+0.153**, 70.8% episodes
improved — clears the project's own gate only because the easier 60% of
instructions carries the weighted-but-worst-performing `relative`.

## TODO, prioritized for the Ohio window (lab/VR/compute access is the
scarce resource here, not time in general — defer anything that doesn't
need it)

1. **Diagnose the `relative` regression.** Compare v4 vs v9 hindsight-data
   coverage and drift-penalty surface for `relative` specifically, without
   a full retrain first — cheaper than a class-count sweep (which would
   cost ~60 compute-hours for 5 points × 3 seeds × ~4h, not worth it
   before a cheaper code-level diff investigation).
2. **Hybrid routing, quick win, no retraining needed:** route `show_only`/
   `absolute` to `hill_climb` search (already beats the policy there,
   ~2–3s latency acceptable for "apply once"); route `relative` to the
   rule-based executor until it's actually fixed, so the live demo never
   shows the policy's known-negative kind.
3. **[Done/superseded 2026-10-15 — see `docs/experiments/2026-10-15-vr-lab-session.md`: custom WebGL2 ray marcher, 27–33 fps lit, vtk.js XR bugs fixed; opt-in `?adapt=1` controller exists.]**
   **VR adaptive sample-distance controller** (a few hours, not
   multi-day): react to real measured frame time instead of trusting
   vtk.js's built-in `AutoAdjustSampleDistances`, which doesn't engage
   usefully inside the custom `session.requestAnimationFrame` XR loop.
   Needs the physical headset to validate — do this before leaving Ohio.
4. **Scale RLHF data collection while people are physically in the lab**
   — this is as location-bound as the headset. Multiple raters via
   `collect.html`, merge with `tools/merge_preferences.py`, report
   Krippendorff's alpha (not Cohen's kappa, since >2 raters is likely).
   Target checkpoints on the learning curve: ~100, ~200, ~400 pairs
   (median judgment time measured at 31s, so 400 pairs ≈ 3.5 labeling-
   hours total, trivially split across several people).
5. **Held-out test-split evaluation** for the current E3/E5 checkpoints —
   straightforward, no new training, just the missing eval run.
6. **Wire `device=` through `visibility.py` and the SAC trainer.** MPS
   already verified working locally (3D `grid_sample` included); test a
   real before/after training-speed comparison on the M1 Pro before
   assuming the Leipzig 2080 Super is needed at all. If MPS doesn't scale
   to full-size volumes, CUDA on the 2080 Super is the fallback (more
   mature PyTorch support than MPS).

### Deferred (real ideas, not realistic for this window)

- Empty-space skipping / bricking and foveated rendering for VR
  (multi-day engine work; the `FOVED` capability flag the headset reports
  is currently unused).
- Server-side VR rendering with the headset's existing Asynchronous Time
  Warp compositor (`RENDER_ATWC`/`RENDER_ATW`, observed live in logcat)
  smoothing the gaps — a legitimate pattern (cf. NVIDIA CloudXR), but a
  real video-streaming/WebRTC pipeline, bigger than this window.
- RLHF stage 2 (fine-tuning against a learned reward model, KL-anchored
  to stage 1) — blocked until enough preference data exists; realistic
  after Ohio, not during it.
- Expanding beyond 8 classes — explicitly *not* recommended; 8 already
  resolves the real HU-overlap problem, more would mostly add compute
  cost without new research insight. Better use of time: make the
  existing 8 actually work.

## What we still want to achieve

**By the end of the Ohio window (~3 weeks):** a prototype that feels
solid end-to-end, not just on the easy instruction kinds — hybrid
routing live, VR smooth-and-sharp-enough via the adaptive controller, a
`relative` root-cause investigation at least started, and a meaningfully
larger, multi-rater, Krippendorff's-alpha-validated preference dataset
banked for later RLHF work.

**By January 2027 submission:** every instruction kind individually
clearing its own 3-seed adoption gate (not just the aggregate median —
the point made above about `relative` dragging the weighted average down
while technically "passing" should not survive to the final thesis
framing), a real held-out test-split number, and an honest, evidence-
backed account of where RL helps, where search wins instead (the hybrid
architecture), and why — which is itself the more defensible thesis claim
than "RL solves everything."

**Longer-term / possible PhD continuation:** the deferred items above
(engine-level VR optimization, full RLHF with a calibrated reward model,
out-of-source generalization work) form a natural, already-scoped
research agenda rather than loose ends.
