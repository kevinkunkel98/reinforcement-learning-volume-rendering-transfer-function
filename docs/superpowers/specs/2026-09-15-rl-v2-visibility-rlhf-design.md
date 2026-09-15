# RL v2: Visibility-Grounded Transfer-Function Agent With Human Preferences

## Goal

Redesign the RL pipeline so it can actually answer the thesis question — *can a
transfer function (TF) be learned with RL?* — before any preference data is
collected. One week (until ~2026-09-22) delivers a working pipeline, a tier-1
result, and collected preference data; the following month extends it.

## Why the current pipeline cannot answer the question

Measured on 2026-09-15 with a numpy front-to-back compositing prototype:

| ct_chest change | Bone share of visible contribution | `mass_fraction(bone)` (current reward) |
|---|---|---|
| default | 0.3% | 0.26 |
| bone peak height → max (the only action the current agent has) | 0.3% | 0.43 |
| fat + soft heights → 0 | 4.5% | 0.26 |

- `mass_fraction` never reads the volume; it rewards changes that do not change
  what is visible.
- The effective strategy (clear occluders) is outside the 1-D action space.
- The reward model sees four global grayscale statistics that cannot tell
  tissues apart.
- The policy observes no information about the volume.

A single fixed rule ("target peak up, other peaks down") raised the target
tissue's visibility on all 4 CT volumes for all 4 tissues (16/16, 1.2×–8.2×), so
a transferable strategy exists. With a pure target-visibility reward, that rule
is close to optimal — which is why the reward below includes a context term.

## Success criteria

**Tier 1 (must-have, no human data):** the SAC policy (10 steps) beats baselines
B1, B2, B3 and B5 on the TotalSegmentator test subjects in mean episode
objective, paired Wilcoxon p < 0.05, for each of 3 seeds. Also reported, not
gating: training volumes (new episodes), out-of-source test volumes, and the
ratio to B4.

**Tier 2 (stretch):** on validation-volume preference pairs, the fine-tuned
reward model (RM) predicts human choices more accurately than the visibility
objective, and on the subset where human and objective disagree the accuracy
difference has a bootstrap 95% CI above 0. Then the RLHF policy is compared with
its control in a blind A/B test.

## Scope

In scope this week: Phase 0 cleanup, TotalSegmentator data, visibility score,
new environment, SAC training and evaluation, embeddings, preference model,
`/collect` page, RLHF fine-tuning, blind comparison.

Out of scope (month): anatomy-based goals from segmentation masks, the full
TotalSegmentator dataset, leave-one-out folds, best-of-4 gallery and iterative
RLHF rounds, color/center actions, multi-turn episodes, command-text goal
embeddings, the new policy in the chat UI, CLIP comparison, MRI.

## Phase 0: Cleanup

Done first, in its own commits.

**Remove**

- All old RL modules in `rl/`: `camera_env.py`, `camera_eval.py`,
  `camera_train.py`, `collect_preferences.py`, `env.py`, `eval.py`,
  `eval_reward.py`, `extract_pairs.py`, `finetune_reward.py`,
  `ingest_feedback.py`, `online_train.py`, `pretrain_reward.py`,
  `reward_model.py`, `reward_model_env.py`, `reward_model_train.py`,
  `serve.py`, `train.py`. The `rl/` package stays for the new modules.
- Their tests: `tests/test_camera_env.py`, `tests/test_camera_eval.py`, and
  every `tests/test_rl_*.py`.
- Old-RL plots: `plots/online_eval_curve.py`,
  `plots/reward_model_eval_curve.py` and their tests. `plots/style.py` and
  `plots/read_progress.py` stay. `plots/training_curves.py` stays only if it is
  a generic SB3 `progress.csv` plotter; the implementation plan checks this.
- `stats.py`, `mvp.py`, and the tracked `.$architecture-*.drawio.bkp` files.
- `evaluate.human()` (console judge, only used by `mvp.py`). `evaluate.objective()`
  stays for the chat UI's hill-climb search.
- Server and UI: `/api/judge`, `/api/feedback`, the `policy` evaluator, the
  pending-judgment state (`Session.judge`, `_next_pending_pair`,
  `_start_human_search`, `Session.feedback`), `PREF_PATH`, `FEEDBACK_PATH`, the
  judge view and thumbs buttons in `static/`, and their server tests.
- Design docs in `docs/superpowers/{specs,plans}/` for removed features:
  `2026-09-04-rl-implementation`, `2026-09-05-training-curve-plots`,
  `2026-09-06-camera-viewpoint-rl`, `2026-09-07-opacity-rl-live-wiring`,
  `2026-09-09-rl-online-learning`, `2026-09-09-rlhf-reward-model`,
  `2026-09-11-goal-conditioned-reward`, `2026-09-11-ingest-feedback`.

**Keep:** renderer, transfer functions, rule/LLM parser, voice (`asr.py`),
camera commands (`camera.py`), datasets, phantom (test fixture), scene schema,
3D viewer, command logging to `out/log.jsonl`, hill-climb search,
`eval_parsers.py`, `tools/`, slides, thesis docs (`docs/*.typ`; they describe
the old pipeline and will be updated by the author).

The README's RL sections are rewritten for v2. The user's uncommitted
`datasets.py`, `.gitignore` and `.dockerignore` changes are built on, not
discarded.

## Data

**Source:** TotalSegmentator small subset v2.0.1 (Zenodo record 10047263,
`Totalsegmentator_dataset_small_v201.zip`, 3.2 GB, 102 subjects, CC-BY-4.0,
NIfTI, Hounsfield units, with 117 structure masks per subject).

**Selection (`tools/select_totalseg.py`):** ~30 subjects, stratified by body
region. Region comes from the dataset's metadata if present, otherwise it is
inferred from which masks are non-empty (brain/skull → head, lungs → thorax,
liver → abdomen, hip/femur → pelvis). Quality filter: slice thickness ≤ 3 mm,
superior–inferior extent ≥ 100 mm, in-plane size ≥ 256². Split by subject:
20 train / 4 validation / 6 test. Output: committed manifest
`data/totalseg_manifest.json` (subject ID, region, split, sha256).

**Loading:** `datasets.py` gains a NIfTI loader (`nibabel`) that reorients every
volume to canonical RAS (`nib.as_closest_canonical`) and returns float32 HU plus
spacing. Registry names are `ts_<subject>`. `.gitignore` adds `data/*.nii.gz` and
the extracted TotalSegmentator directory.

**Out-of-source test set:** the four Slicer CTs (`ct_chest`, `ct_skull`,
`ct_cardio`, `ct_abdomen`). Each gets a registry entry mapping it to canonical
orientation (axis permutation and flips), verified visually once.

MRI and uncalibrated volumes are excluded from v2.

## Architecture

```text
                   ┌──────────── Tier 1 (no human data) ─────────────┐
volume ─► views.py (6 cameras per volume)                            │
       └► visibility.py ─► per-tissue visibility + coverage          │
                 │                                                   │
                 ▼                                                   │
          rl/vis_env.py ─► rl/vis_train.py (SAC) ─► rl/vis_eval.py   │
                 ▲                                    ▲              │
                 └──────── rl/baselines.py ───────────┘              │
                   └─────────────────────────────────────────────────┘
                   ┌──────────── Tier 2 (human data) ────────────────┐
rl/candidates.py ─► collect.py (routes) ─► static/collect.{html,js}   │
                          │ choice + TF + cameras seen                │
                          ▼                                           │
                 out/vis_preferences.jsonl                            │
                          ▼                                           │
     embed.py (VTK, 6 views, 224 px ─► DINOv2-S ─► mean, 384-D)       │
                          ▼                                           │
                 rl/pref_model.py (PCA + Bradley–Terry ensemble)      │
                          ▼                                           │
                 rl/rlhf_finetune.py ─► rl/blind_eval.py              │
                   └─────────────────────────────────────────────────┘
```

### Units

| Unit | Responsibility | Interface |
|---|---|---|
| `views.py` | The 6 standard cameras for a volume | `views_for(volume_id) -> list[Camera]`; `Camera` is renderer-neutral `{position, focal_point, view_up}` |
| `visibility.py` | Per-tissue visibility estimate | `VisibilityModel(volume_id).features(params) -> dict` with keys `fat, soft, spongy, bone, coverage` |
| `embed.py` | Multi-view image embedding with disk cache | `embed_state(volume_id, params) -> np.ndarray[384]` |
| `rl/vis_env.py` | Gymnasium environment | `VisibilityTFEnv(volume_ids, reward_fn=None, seed=None)` |
| `rl/baselines.py` | B1–B5 as functions of `(env state) -> final params` | one function per baseline |
| `rl/vis_train.py` | SAC training CLI | `python -m rl.vis_train --seed N --lam 0.3` |
| `rl/vis_eval.py` | Fixed-episode evaluation against baselines | writes JSON report + CSV per episode |
| `rl/candidates.py` | Start state, goal and candidate pair sampling, near-duplicate filter | `sample_item(volume_id, rng, policy) -> Item` |
| `collect.py` | FastAPI router for `/collect` and `/api/collect/*`, mounted by `server.py` | see Collection |
| `rl/pref_model.py` | PCA, RM ensemble, pretraining and fine-tuning CLIs | `score(start_emb, final_emb, goal) -> (mean, std)` |
| `rl/rlhf_finetune.py` | Fine-tune a tier-1 policy with the RM | CLI with `--alpha` |
| `rl/blind_eval.py` | Build blind episodes, analyze `out/blind_eval.jsonl` | CLI |
| `tools/validate_visibility.py` | Proxy vs VTK validation report | CLI |
| `tools/smoke_rl_v2.py` | End-to-end smoke run | CLI, < 5 min |

`render.render()` gains support for the renderer-neutral camera form so
`embed.py` and `views.py` share one camera definition; the existing
azimuth/elevation form keeps working for the chat UI.

## Tier 1

### Views

The volume is in canonical RAS. View 0 looks at the volume center from anterior,
with superior as view-up, at a distance that fits the volume bounds. Views 1–5
rotate view 0 about the superior axis in 60° steps. The same 6 cameras are used
by the visibility score, the embeddings, and (view 0) the `/collect` start
camera.

### Visibility score

Precomputed once per volume and cached under `out/cache/visibility/`:

1. Resample to isotropic spacing, longest axis 96 voxels.
2. For each of the 6 views, rotate the grid so the view direction is axis 0
   (`torch.nn.functional.affine_grid` + `grid_sample`, trilinear).
3. Label each voxel by `TISSUE_BANDS` (air, fat, soft, spongy, bone).

Per call, on all 6 grids as one batched torch operation:

- Opacity per voxel from the TF's 256-entry lookup table, corrected for step
  length: `α = 1 − (1 − α_TF)^(Δs / 1 mm)`, with `Δs` the resampled voxel size.
- Front-to-back transmittance `T = cumprod(1 − α)` (exclusive).
- Contribution `w = T · α`.
- `V_t` = mean over rays of the summed contribution of voxels labeled `t`,
  averaged over the 6 views.
- `coverage` = share of rays whose accumulated opacity is ≥ 0.3, averaged over
  views.

Budget: ≤ 30 ms per call. If exceeded: longest axis 64, then 3 views.

**Validation gate** (`tools/validate_visibility.py`): 50 random TFs per volume
on 3 volumes; Spearman correlation between proxy coverage and coverage of the
real VTK render (same camera) ≥ 0.8, plus a side-by-side figure. Environment
work does not start until the gate passes.

### Environment (`VisibilityTFEnv`)

- **Reset:** training volume chosen uniformly; goal = one of 8
  (`fat, soft, spongy, bone` × `increase, decrease`), uniform; start TF =
  `default_params()` plus uniform noise ±0.3 on heights and ±0.2 on widths
  (normalized units, clipped to [−1, 1]).
- **Observation (35):** 8 controllable values (height, width of each peak), goal
  one-hot (4), direction (±1), `log10(V_t + 1e-3)` for 4 tissues, coverage,
  16-bin intensity histogram of the full volume over `CENTER_RANGE`
  (normalized to sum 1, precomputed), step fraction `t / 10`.
- **Action:** 8 values in [−1, 1], scaled to ±0.1 change of each height and
  width per step. Centers and colors stay fixed.
- **Episode:** 10 steps, then truncation.
- **Reward per step:**
  `r_t = s · Δ log10(V_target + ε) + λ · Δ log10(V_context + ε) − 1[coverage < 0.01]`,
  with `s = +1` for increase and `−1` for decrease, `V_context` = sum of the
  other three tissues, `ε = 1e-3`, `λ = 0.3` (ablation: `λ = 0`).
- **info:** visibility before/after, objective components.

### Training

SAC (stable-baselines3, default hyperparameters), 100k steps, seeds 0–2, on the
20 training volumes. Checkpoints every 10k steps; the best checkpoint is chosen
by mean objective on the validation volumes. Runs go to `out/rl_v2/<run>/`.

### Evaluation

Fixed episode sets (seeded) of 200 episodes each: training volumes (new
episodes), TotalSegmentator test subjects (primary), out-of-source Slicer CTs.

| ID | Baseline |
|---|---|
| B1 | Old behavior: target peak height to 0.9 (increase) or 0.02 (decrease), nothing else |
| B2 | Random policy, 10 steps |
| B3 | Coordinate hill-climber on the episode objective, 10 evaluations |
| B4 | Same hill-climber, 200 evaluations (reference, not gating) |
| B5 | Rule: increase → target 0.9, others 0.02; decrease → target 0.02, others unchanged |

Metrics: episode objective (sum of `r_t`), final target visibility gain, share of
improved episodes, empty-image rate, per-goal breakdown, evaluations used.
Paired Wilcoxon per baseline. Also reports `λ = 0` runs against the same
baselines.

## Tier 2

### Collection (`/collect`)

**Item generation (`rl/candidates.py`):**

- Volume from the training and validation splits only, in blocks of 10 items per
  volume.
- Start TF as in the environment; goals weighted 2 (increase) : 1 (decrease).
- Candidate pair: 50% two stochastic samples of the tier-1 policy
  (`deterministic=False`, 10 steps); 50% one policy sample vs one of B5, B3, or
  a random perturbation (start ± uniform 0.3 on the 8 values), chosen uniformly.
- Near-duplicate filter: the pair is regenerated unless at least one tissue
  differs by ≥ 0.1 in `log10(V + ε)` between A and B.
- The objective's preferred side is stored with the item.

**Repeats:** 10% of items are repeats of an earlier item by the same rater,
shown at least 20 items later with A/B sides swapped.

**Page (`static/collect.html`, `static/collect.js`):**

- Goal sentence ("Make the bone more visible").
- Two vtk.js viewports sharing one loaded volume, with linked cameras (moving
  one moves both), starting at view 0.
- "Show start" toggle sets both viewports to the start TF while held.
- Keys: `A`, `B`, `E` (equal), `S` (skip).
- Rater ID entered once, kept in `localStorage`.
- Camera state sampled every 500 ms during interaction into `views_seen`.

`static/viewer.js` is refactored if needed so a viewport can be created for a
given container and share already-loaded image data.

**Routes (`collect.py`):** `GET /collect`, `POST /api/collect/next`
(`{rater_id, mode}` → item without source labels), `POST /api/collect/judge`
(`{pair_id, choice, views_seen, decision_ms}`). `mode` is `collect` or
`blind`. Writes are append-only under a lock, reusing the scene-log pattern.

**Row format (`out/vis_preferences.jsonl`, read directly by training):**

```json
{"pair_id": "...", "timestamp": "...", "rater_id": "kk",
 "volume_id": "ts_s0123", "volume_version": "sha256:...",
 "goal": {"tissue": "bone", "direction": "increase"},
 "start_params": [24 floats],
 "a": {"params": [24 floats], "source": "policy"},
 "b": {"params": [24 floats], "source": "rule"},
 "choice": "a", "objective_choice": "b",
 "visibility": {"start": {}, "a": {}, "b": {}},
 "views_seen": [{"t": 0.0, "camera": {}}],
 "decision_ms": 4200, "repeat_of": null}
```

`choice` ∈ `a | b | equal | skip`. Images and embeddings are not stored; they
are recomputed from `(volume_id, params)`.

**Target:** ≥ 300 decisive pairs.

### Embeddings (`embed.py`)

- VTK renders of the 6 views at 224×224 in a dedicated offscreen window.
- DINOv2 ViT-S/14 via `torch.hub.load("facebookresearch/dinov2",
  "dinov2_vits14")`, ImageNet normalization done in torch (no `torchvision`),
  CLS token per view, batched on MPS when available.
- Mean over views, L2-normalized, 384-D.
- Disk cache `out/cache/embeddings/` keyed by sha256 of
  `(volume_version, params rounded to 1e-4, view set id, model id)`.
- Fallback if `torch.hub` DINOv2 fails on Python 3.14: `transformers`
  `facebook/dinov2-small`.

### Preference model (`rl/pref_model.py`)

- **PCA to 64 dims**, fitted on ~5,000 unlabeled final states from tier-1
  rollouts on training volumes; stored with the checkpoint.
- **Input (133):** `pca(final)` (64), `pca(final) − pca(start)` (64), goal
  one-hot (4), direction (1).
- **Network:** 133 → 64 → 1, ReLU, dropout 0.2, weight decay 1e-3.
- **Ensemble:** 5 members, bootstrap-resampled training pairs.
- **Loss:** Bradley–Terry; `equal` uses soft target 0.5; `skip` is ignored.
- **Pretraining:** ~4,000 pairs of two final states from the same start and
  goal (tier-1 policy samples and baselines), labeled by the tier-1 episode
  objective, on training volumes.
- **Fine-tuning:** human pairs from training volumes, early stopping on
  validation-volume pairs. Pretrained and fine-tuned checkpoints are separate.
- **Score mapping:** member score `2·sigmoid(s) − 1`; ensemble returns mean and
  std.

**Evaluation** on validation-volume pairs: visibility objective (λ = 0.3),
pretrained RM, fine-tuned RM, rater self-consistency from repeats. Each overall,
on the human–objective disagreement subset, and per goal. Bootstrap 95% CIs.

### RLHF fine-tuning (`rl/rlhf_finetune.py`)

- Starts from the best tier-1 checkpoint (λ = 0.3).
- Reward: `r_t = (1 − α) · r_obj,t − 1[coverage < 0.01]`, plus at the last step
  `α · (μ − σ)` of the RM ensemble scoring start → final.
- 20k steps per run, α ∈ {0, 0.7, 1.0}, seeds 0–2. α = 0 is the control for
  extra training steps.
- Logged every 1k steps: RM mean, RM std, objective. If the ensemble std on
  rollouts exceeds 1.5× its value at the start of fine-tuning, training stops
  and the last checkpoint below that threshold is kept.

### Blind comparison (`rl/blind_eval.py`)

- 40 fixed episodes from TotalSegmentator test subjects and the Slicer CTs,
  goals uniform over 8.
- RLHF policy (α = 0.7, deterministic) vs α = 0 control (deterministic), shown
  on `/collect` in `blind` mode with sources hidden and order randomized.
- Written to `out/blind_eval.jsonl`, never read by training.
- Preferably rated by a second person who did not label training pairs.
- Report: win rate excluding ties, Wilson 95% CI, binomial test; gallery figure.

## Error handling

- Missing dataset files or model weights: clear error naming the command that
  fetches them.
- Visibility: NaN/inf guard; all-transparent TF returns zeros, not NaN.
- Collection endpoints validate `choice`, `pair_id`, rater ID; unknown pair →
  400.
- Training CLIs refuse to overwrite existing run directories.

## Testing

Unit tests (fast, no weights):

- Visibility: front slab occluding back slab (clearing the front raises the back
  tissue's `V`); transparent TF → zero coverage; asymmetric volume checks view
  rotations; step-length correction.
- Views: 6 distinct cameras, view 0 anterior with superior up on a synthetic
  RAS volume.
- NIfTI loader: synthetic affine → canonical orientation and spacing.
- Environment: observation/action shapes, bounds, reward sign per direction,
  empty-image penalty, determinism with a seed.
- Baselines: each returns valid params; B1 changes only the target height.
- Candidates: near-duplicate filter, repeat scheduling, goal weights.
- Preference model: Bradley–Terry, soft ties, PCA round-trip, ensemble shape.
- Collection routes: row schema, blind mode hides sources, append-only writes.
- `embed.py` with an injected stub embedder and renderer.

Marked `slow`: real DINOv2 load, real VTK multi-view rendering.

`tools/smoke_rl_v2.py`: 2 volumes, 500 SAC steps, 20 pairs from a scripted
rater, PCA + RM + 200 RLHF steps, under 5 minutes. Must pass before collecting.

## Schedule

| Day | Work |
|---|---|
| 1 | Phase 0 cleanup; check `nibabel` and DINOv2 on Python 3.14; download and select TotalSegmentator; NIfTI loader |
| 2 | `views.py`, `visibility.py`, validation gate; environment; tier-1 training overnight |
| 3 | Baselines and evaluation (tier-1 result); `embed.py`, PCA, RM pretraining overnight |
| 4 | `/collect` page and routes; smoke test; start collecting |
| 5 | Collect ≥ 300 pairs; RM fine-tuning and evaluation (RM result) |
| 6 | RLHF α ablation; blind comparison |
| 7 | Buffer; if behind, day 6 moves into the month |

## Risks

- `nibabel` or DINOv2 not working on Python 3.14 → fallbacks: `transformers`,
  or a minimal NIfTI-1 reader in numpy.
- Visibility proxy diverges from VTK → validation gate blocks environment work
  until the view math is fixed.
- Inconsistent human choices → measured by repeats; reported as a result.
- Time → tier 2 is the stretch; RLHF and blind comparison may move to the month.

## Decisions

| Decision | Choice |
|---|---|
| Success bar | Two tiers: objective result must succeed, human-preference result is the stretch |
| Goals | fat, soft, spongy, bone × increase/decrease, CT only |
| Action | height + width of all 4 peaks |
| Tier-1 reward | numpy visibility with context term λ = 0.3 (λ = 0 ablation) |
| RM input | DINOv2 ViT-S/14 embeddings of 6 server-rendered views |
| Judging | A/B from the same start, two linked 3D viewports |
| Data | TotalSegmentator small subset, ~30 subjects, split by subject; Slicer CTs as out-of-source test |
| Old code | Old RL pipeline, camera RL, old feedback paths and `mvp.py` removed |
