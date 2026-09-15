# RL v2: Instruction-Following Transfer-Function Agent With Human Preferences

## Goal

Redesign the RL pipeline so it can answer the thesis question — *can a
transfer function (TF) be learned with RL?* — for the instructions a user will
later speak in VR. One goal-conditioned policy learns a diverse set of
perceptual instructions ("more bone", "more bone, a bit less spongy", "show only
bone", "high opacity spongy", "brighten fat") on real CT volumes, first from a
visibility-based objective, then refined with human preferences.

One week (until ~2026-09-22) delivers: cleanup, data, the instruction-following
policy with its evaluation (tier 1), the collection page, and collected
preference data. Reward-model fine-tuning, RLHF, and the blind comparison start
the following month.

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
- The goal is one tissue plus a direction; compound or graded instructions
  cannot be expressed.

A single fixed rule ("target peak up, other peaks down") raised the target
tissue's visibility on all 4 CT volumes for all 4 tissues (16/16, 1.2×–8.2×), so
a transferable strategy exists. With requested amounts and "keep the rest"
constraints, that rule overshoots and the policy must learn how much to change
per volume. Tissues also interfere: the default bone peak (center 900 HU,
width 280 HU) still reaches 56% of its height at the spongy/bone boundary
(600 HU), so "more bone, less spongy" requires narrowing the bone peak.

## Instruction split

| Instruction | Example | Handled by |
|---|---|---|
| Relative visibility, graded | "more bone", "a bit less fat" | policy |
| Compound relative | "more bone, a bit less spongy" | policy |
| Absolute level | "high opacity bone" | policy |
| Show only | "show only bone and spongy" | policy |
| Brightness, graded | "brighten bone", "darken fat" | policy |
| Sharpen/soften, center shift | "sharpen bone", "shift fat down" | exact (`apply_command`) |
| Camera, reset | "rotate left", "reset" | exact |

Exact instructions have a closed-form mapping to parameters; learning adds
nothing. Perceptual instructions depend on the volume and on occlusion; that is
where RL is used.

## Success criteria

**Goal attainment** per episode: `A = 1 − D_final / D_start`, with `D` the goal
distance defined under Reward (1 = goal reached, 0 = no progress, < 0 = worse).

**Tier 1 (must-have, no human data):** on the TotalSegmentator test subjects,
the policy's mean attainment exceeds B1, B2, B3 and B5 (paired Wilcoxon,
p < 0.05) for each of 3 seeds. Reported, not gating: attainment per instruction
type, training volumes (new episodes), out-of-source test volumes, ratio to B4.

**Tier 2 (month):** on validation-volume preference pairs, the fine-tuned reward
model (RM) predicts human choices more accurately than the objective, and on the
subset where human and objective disagree the accuracy difference has a
bootstrap 95% CI above 0. Then the RLHF policy is compared with its control in a
blind A/B test.

## Scope

**This week:** Phase 0 cleanup; TotalSegmentator data; visibility and
brightness estimate; goal representation, sampler and parser mapping; parser
extension; environment; SAC training and evaluation; `/collect` page and
preference collection; `embed.py` and PCA if time remains.

**Month:** RM pretraining/fine-tuning, RLHF, blind comparison; the policy in the
chat UI and VR; anatomy-based goals from segmentation masks; learned
sharpen/center; learned viewpoints; full TotalSegmentator and leave-one-out;
best-of-4 gallery and iterative RLHF rounds; command-text goal embeddings; CLIP
comparison; MRI.

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
  `plots/read_progress.py` stay. `plots/training_curves.py` stays (generic SB3
  `progress.csv` plotter).
- `stats.py`, `mvp.py`, and the tracked `.$architecture-*.drawio.bkp` files.
- `evaluate.human()` (console judge, only used by `mvp.py`). `evaluate.objective()`
  stays for the chat UI's hill-climb search.
- Server and UI: `/api/judge`, `/api/feedback`, the `policy` evaluator, the
  pending-judgment state (`Session.judge`, `_next_pending_pair`,
  `_start_human_search`, `Session.feedback`), `PREF_PATH`, `FEEDBACK_PATH`, the
  judge view and thumbs buttons in `static/`, and their server tests.
- Design docs in `docs/superpowers/{specs,plans}/` for removed features:
  `2026-09-04-rl-implementation`, `2026-09-06-camera-viewpoint-rl`,
  `2026-09-07-opacity-rl-live-wiring`, `2026-09-09-rl-online-learning`,
  `2026-09-09-rlhf-reward-model`, `2026-09-11-goal-conditioned-reward`,
  `2026-09-11-ingest-feedback`. (`2026-09-05-training-curve-plots` stays: the
  plotter is generic and kept.)

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
text ─► parser (rule/LLM) ─► goals.py ─► goal vector                 │
                                  ▲          │                        │
                         sampler ─┘          ▼                        │
volume ─► views.py ─► visibility.py ─► rl/vis_env.py ─► rl/vis_train │
                                             ▲              │         │
                              rl/baselines ──┴── rl/vis_eval ◄┘       │
                   └─────────────────────────────────────────────────┘
                   ┌──────────── Tier 2 (human data) ────────────────┐
rl/candidates.py ─► collect.py ─► static/collect.{html,js}            │
                          │ choice + TF + cameras seen                │
                          ▼                                           │
                 out/vis_preferences.jsonl                            │
                          ▼                                           │
     embed.py (VTK, 6 views, 224 px ─► DINOv2-S ─► mean, 384-D)       │
                          ▼                                           │
     rl/pref_model.py ─► rl/rlhf_finetune.py ─► rl/blind_eval.py      │
                   └─────────────────────────────────────────────────┘
```

### Units

| Unit | Responsibility | Interface |
|---|---|---|
| `views.py` | The 6 standard cameras for a volume | `views_for(volume_id) -> list[Camera]`; `Camera` is renderer-neutral `{position, focal_point, view_up}` |
| `visibility.py` | Per-tissue visibility and brightness estimate | `VisibilityModel(volume_id).features(params) -> Features` with `vis[4]`, `bright[4]`, `coverage`; `.solo_max(tissue)` |
| `goals.py` | Goal vector, goal distance, sampler, command → goal, goal → text | see Goals |
| `embed.py` | Multi-view image embedding with disk cache | `embed_state(volume_id, params) -> np.ndarray[384]` |
| `rl/vis_env.py` | Gymnasium environment | `VisibilityTFEnv(volume_ids, reward_fn=None, seed=None)` |
| `rl/baselines.py` | B1–B5 as functions `(volume_id, start_params, goal) -> final params` | one function per baseline |
| `rl/vis_train.py` | SAC training CLI | `python -m rl.vis_train --seed N --lam 0.3` |
| `rl/vis_eval.py` | Fixed-episode evaluation against baselines | JSON report + per-episode CSV |
| `rl/candidates.py` | Item sampling for collection, near-duplicate filter | `sample_item(volume_id, rng, policy) -> Item` |
| `collect.py` | FastAPI router for `/collect` and `/api/collect/*`, mounted by `server.py` | see Collection |
| `rl/pref_model.py` | PCA, RM ensemble, pretraining and fine-tuning CLIs | `score(start_emb, final_emb, goal) -> (mean, std)` |
| `rl/rlhf_finetune.py` | Fine-tune a tier-1 policy with the RM (month) | CLI with `--alpha` |
| `rl/blind_eval.py` | Blind episodes and analysis (month) | CLI |
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
by the visibility estimate, the embeddings, and (view 0) the `/collect` start
camera.

### Visibility and brightness estimate

Precomputed once per volume and cached under `out/cache/visibility/`:

1. Resample to isotropic spacing, longest axis 96 voxels.
2. For each of the 6 views, rotate the grid so the view direction is axis 0
   (`torch.nn.functional.affine_grid` + `grid_sample`, trilinear).
3. Label each voxel by `TISSUE_BANDS` (air, fat, soft, spongy, bone).
4. 16-bin intensity histogram of the full volume over `CENTER_RANGE`,
   normalized to sum 1.

Per call, on all 6 grids as one batched torch operation:

- Opacity and RGB per voxel from the TF's 256-entry lookup tables; opacity
  corrected for step length: `α = 1 − (1 − α_TF)^(Δs / 1 mm)`, with `Δs` the
  resampled voxel size.
- Front-to-back transmittance `T = cumprod(1 − α)` (exclusive), contribution
  `w = T · α`.
- `vis_t` = mean over rays of the summed `w` of voxels labeled `t`.
- `bright_t` = `Σ w · lum(rgb) / Σ w` over voxels labeled `t`, with
  `lum = 0.2126 R + 0.7152 G + 0.0722 B`; `0` when `Σ w < 1e-4`.
- `coverage` = share of rays whose accumulated opacity is ≥ 0.3.
- All quantities averaged over the 6 views.

`solo_max(t)` = `vis_t` with tissue `t`'s peak at height 1 and all other heights
0 (default widths), computed once per volume and tissue and cached.

Budget: ≤ 30 ms per call. If exceeded: longest axis 64, then 3 views.

**Validation gate** (`tools/validate_visibility.py`): 50 random TFs per volume
on 3 volumes; Spearman correlation between proxy coverage and coverage of the
real VTK render (same camera) ≥ 0.8, plus a side-by-side figure. Environment
work does not start until the gate passes.

### Goals (`goals.py`)

Tissues `(fat, soft, spongy, bone)`. A goal vector has 16 values:

- `d[4]`: requested change of `log10(vis_t + ε)` relative to the episode start
- `m[4]`: 1 if the tissue's visibility is part of the instruction, else 0
- `e[4]`: requested change of `bright_t` relative to the episode start
- `n[4]`: 1 if the tissue's brightness is part of the instruction, else 0

`ε = 1e-3`.

**Command → goal** (`goal_from_command(cmd, vis_model, start_features)`):

| Command | Goal entries |
|---|---|
| opacity increase/decrease X, strength | `d_X = ±{slightly 0.15, moderately 0.3, strongly 0.6}`, `m_X = 1` |
| compound | union of its sub-commands |
| opacity set X level | `d_X = log10(L · solo_max(X) + ε) − log10(vis_X,start + ε)`, `L = {low 0.1, medium 0.4, high 0.8}`, `m_X = 1` |
| show only X (, Y) | named: `d = max(0, log10(0.8 · solo_max + ε) − log10(vis_start + ε))`; others: `d = log10(ε) − log10(vis_start + ε)` (hide); all `m = 1` |
| brightness increase/decrease X, strength | `e_X = ±{slightly 0.1, moderately 0.2, strongly 0.4}`, `n_X = 1` |

Other commands (width, center, camera, reset) are not goals; they go to
`apply_command` or the camera code as today.

**Goal distance** at state `s`, with `c_t = log10(vis_t + ε) − log10(vis_t,start + ε)`
and `b_t = bright_t − bright_t,start`:

```
D(s) = Σ_t m_t |c_t − d_t|  +  κ Σ_t n_t |b_t − e_t|
     + λ Σ_t (1 − m_t) |c_t|  +  λ κ Σ_t (1 − n_t) |b_t|
```

`κ = 1.5` (a brightness change of 0.2 weighs like a visibility change of 0.3),
`λ = 0.3` (weight of "keep the unmentioned tissues as they are"; ablation
`λ = 0`).

**Sampler** (`sample_goal(rng)`) returns `(command dict, text, goal)` with the
following mix; tissues, strengths and levels uniform:

| Type | Share |
|---|---|
| single relative visibility | 40% |
| compound relative (2 tissues) | 25% |
| show only (1–2 tissues) | 15% |
| absolute level (1–2 tissues) | 10% |
| brightness | 10% |

The sampler never hides all four tissues. Text comes from templates with
synonyms ("more bone", "increase opacity for bone strongly", "a bit less
spongy") and is used on the collect page and as parser test phrases.

### Parser extension

- Rule parser: short forms "more/less X [a bit/much/slightly/strongly]";
  compound relative commands split on "," / "and" into a `compound` of relative
  sub-commands.
- Fix: a sentence with several relative clauses currently returns only the
  first clause; it must return all of them or raise.
- LLM parser prompt: `compound` may contain relative sub-commands with
  `strength`, not only `set` sub-commands; validation updated.
- `data/parser_eval_phrases.json` gains phrases for every instruction type;
  `eval_parsers.py` reports accuracy per type.

`apply_command` already folds compound sub-commands of any kind and needs no
change.

### Environment (`VisibilityTFEnv`)

- **Reset:** training volume chosen uniformly; goal from `sample_goal`; start TF
  = `default_params()` plus uniform noise ±0.3 on heights and ±0.2 on widths
  (normalized units, clipped to [−1, 1]), and one uniform ±0.2 per peak added
  to its r, g, b in unit space (clipped to [0, 1]).
- **Action (12):** per peak: height, width, brightness, each in [−1, 1], scaled
  to ±0.1 change per step. Brightness adds the same amount to the peak's r, g, b
  in unit space, clipped to [0, 1]. Centers and hue stay fixed.
- **Observation (62):** 12 controllable values; goal (16); current
  `log10(vis + ε)` (4) and `bright` (4); progress `c` (4) and `b` (4);
  coverage (1); volume histogram (16); step fraction `t / 10` (1).
- **Episode:** 10 steps, then truncation.
- **Reward:** `r_t = D(s_{t−1}) − D(s_t) − 1[coverage < 0.01]`. Returns
  telescope to `D_start − D_final` minus penalties.
- **info:** features before/after, `D`, goal type.

### Training

SAC (stable-baselines3, default hyperparameters), 200k steps, seeds 0–2, on the
20 training volumes. Checkpoints every 20k steps; the best checkpoint is chosen
by mean attainment on the validation volumes. Runs go to `out/rl_v2/<run>/`.

### Evaluation

Fixed seeded sets of 300 episodes each (sampler mix as in training): training
volumes (new episodes), TotalSegmentator test subjects (primary), out-of-source
Slicer CTs.

| ID | Baseline |
|---|---|
| B1 | Current system: `apply_command(cmd, start_params)` for the episode's command |
| B2 | Random policy, 10 steps |
| B3 | Coordinate hill-climber on `D`, 10 evaluations |
| B4 | Same hill-climber, 200 evaluations (reference, not gating) |
| B5 | Occlusion rule: tissues to increase or show → height 0.9, hidden or decreased tissues → 0.02, all other tissues → 0.02 if any tissue is increased or shown; brightness as B1 |

Metrics: attainment (overall and per instruction type), `D_final`, empty-image
rate, evaluations used. Paired Wilcoxon per baseline. The `λ = 0` runs are
reported against the same baselines.

## Tier 2

### Collection (`/collect`)

**Item generation (`rl/candidates.py`):**

- Volume from the training and validation splits only, in blocks of 10 items per
  volume.
- Start TF as in the environment; goal and text from `sample_goal`.
- Candidate pair: 50% two stochastic samples of the tier-1 policy
  (`deterministic=False`, 10 steps); 50% one policy sample vs one of B1, B5, B3,
  or a random perturbation (start ± uniform 0.3 on the 12 values), chosen
  uniformly.
- Near-duplicate filter: the pair is regenerated unless at least one tissue
  differs by ≥ 0.1 in `log10(vis + ε)` or by ≥ 0.05 in `bright`.
- The objective's preferred side (lower `D`) is stored with the item.

**Repeats:** 10% of items are repeats of an earlier item by the same rater,
shown at least 20 items later with A/B sides swapped.

**Page (`static/collect.html`, `static/collect.js`):**

- The instruction text.
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
 "command": {}, "text": "more bone, a bit less spongy", "goal": [16 floats],
 "start_params": [24 floats],
 "a": {"params": [24 floats], "source": "policy"},
 "b": {"params": [24 floats], "source": "rule"},
 "choice": "a", "objective_choice": "b",
 "features": {"start": {}, "a": {}, "b": {}},
 "views_seen": [{"t": 0.0, "camera": {}}],
 "decision_ms": 4200, "repeat_of": null}
```

`choice` ∈ `a | b | equal | skip`. Images and embeddings are not stored; they
are recomputed from `(volume_id, params)`.

**Target:** ≥ 300 decisive pairs this week.

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
- **Input (144):** `pca(final)` (64), `pca(final) − pca(start)` (64), goal (16).
- **Network:** 144 → 64 → 1, ReLU, dropout 0.2, weight decay 1e-3.
- **Ensemble:** 5 members, bootstrap-resampled training pairs.
- **Loss:** Bradley–Terry; `equal` uses soft target 0.5; `skip` is ignored.
- **Pretraining:** ~4,000 pairs of two final states from the same start and
  goal (tier-1 policy samples and baselines), labeled by lower `D`, on training
  volumes.
- **Fine-tuning:** human pairs from training volumes, early stopping on
  validation-volume pairs. Pretrained and fine-tuned checkpoints are separate.
- **Score mapping:** member score `2·sigmoid(s) − 1`; ensemble returns mean and
  std.

**Evaluation** on validation-volume pairs: objective (lower `D`), pretrained RM,
fine-tuned RM, rater self-consistency from repeats. Each overall, on the
human–objective disagreement subset, per instruction type. Bootstrap 95% CIs.

### RLHF fine-tuning (`rl/rlhf_finetune.py`)

- Starts from the best tier-1 checkpoint (`λ = 0.3`).
- Reward: `r_t = (1 − α) · (D(s_{t−1}) − D(s_t)) − 1[coverage < 0.01]`, plus at
  the last step `α · (μ − σ)` of the RM ensemble scoring start → final.
- 20k steps per run, α ∈ {0, 0.7, 1.0}, seeds 0–2. α = 0 is the control for
  extra training steps.
- Logged every 1k steps: RM mean, RM std, attainment. If the ensemble std on
  rollouts exceeds 1.5× its value at the start of fine-tuning, training stops
  and the last checkpoint below that threshold is kept.

### Blind comparison (`rl/blind_eval.py`)

- 40 fixed episodes from TotalSegmentator test subjects and the Slicer CTs,
  goals from the sampler.
- RLHF policy (α = 0.7, deterministic) vs α = 0 control (deterministic), shown
  on `/collect` in `blind` mode with sources hidden and order randomized.
- Written to `out/blind_eval.jsonl`, never read by training.
- Preferably rated by a second person who did not label training pairs.
- Report: win rate excluding ties, Wilson 95% CI, binomial test; gallery figure.

## Error handling

- Missing dataset files or model weights: clear error naming the command that
  fetches them.
- Visibility: NaN/inf guard; an all-transparent TF returns zeros, not NaN.
- `goal_from_command` raises for commands that are not goals; the caller routes
  them to `apply_command`.
- Collection endpoints validate `choice`, `pair_id`, rater ID; unknown pair →
  400.
- Training CLIs refuse to overwrite existing run directories.

## Testing

Unit tests (fast, no weights):

- Visibility: front slab occluding back slab (clearing the front raises the back
  tissue's `vis`); transparent TF → zero coverage; brightness follows peak RGB;
  asymmetric volume checks view rotations; step-length correction; `solo_max`.
- Views: 6 distinct cameras, view 0 anterior with superior up on a synthetic
  RAS volume.
- NIfTI loader: synthetic affine → canonical orientation and spacing.
- Goals: each command type maps to the documented entries; `D = 0` at a state
  matching the goal; sampler mix and "never hide all" rule; text round-trips
  through the rule parser for every template.
- Parser: short forms, compound relative, no silently dropped clauses.
- Environment: observation/action shapes, bounds, reward = decrease in `D`,
  empty-image penalty, determinism with a seed.
- Baselines: each returns valid params; B1 equals `apply_command`.
- Candidates: near-duplicate filter, repeat scheduling.
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
| 2 | `views.py`, `visibility.py` (vis + brightness), validation gate; `goals.py`; parser extension |
| 3 | Environment, baselines; tier-1 training overnight (3 seeds) |
| 4 | Evaluation (tier-1 result, per instruction type); `/collect` page and routes; smoke test |
| 5 | Collect preference pairs; `embed.py` and PCA |
| 6 | Collect until ≥ 300 decisive pairs; README for v2 |
| 7 | Buffer, writing |
| Month start | RM pretraining and fine-tuning, RLHF α ablation, blind comparison |

## Risks

- `nibabel` or DINOv2 not working on Python 3.14 → fallbacks: `transformers`,
  or a minimal NIfTI-1 reader in numpy.
- Visibility proxy diverges from VTK → validation gate blocks environment work
  until the view math is fixed.
- Some instruction types unreachable on some volumes (e.g. strongly more soft
  tissue when it already dominates) → attainment reported per type; the sampler
  can be reweighted after the first training run.
- Inconsistent human choices → measured by repeats; reported as a result.
- Time → collection is the last hard deliverable of the week; everything in
  tier 2 after collection is month work.

## Decisions

| Decision | Choice |
|---|---|
| Success bar | Tier 1 (objective) must succeed this week; tier 2 (human preferences) in the month |
| Instructions | Policy: relative, compound, absolute, show only, brightness. Exact: sharpen/center, camera, reset |
| Goal | 16-value vector: requested visibility and brightness change per tissue + masks |
| Action | Height, width, brightness of all 4 peaks (12-D) |
| Tier-1 reward | Decrease of goal distance `D` from the visibility/brightness estimate; `λ = 0.3` keep term |
| Language | Parser (rule/LLM) → command → goal vector; policy is language-free |
| RM input | DINOv2 ViT-S/14 embeddings of 6 server-rendered views |
| Judging | A/B from the same start, two linked 3D viewports |
| Data | TotalSegmentator small subset, ~30 subjects, split by subject; Slicer CTs as out-of-source test |
| Old code | Old RL pipeline, camera RL, old feedback paths and `mvp.py` removed |
