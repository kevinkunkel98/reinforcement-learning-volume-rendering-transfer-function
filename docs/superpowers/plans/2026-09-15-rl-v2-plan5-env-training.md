# RL v2 Plan 5: Environment, Training and Evaluation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one goal-conditioned policy that follows visibility instructions on CT volumes, and measure it against the baselines on volumes it never saw — the tier-1 result of RL v2.

**Architecture:** `rl/vis_env.py` is a Gymnasium environment: each episode picks a training volume and an instruction, and the agent edits the transfer function for 10 steps while the reward is the decrease in goal distance. `rl/vis_train.py` trains SAC across the 20 training volumes with periodic evaluation on the validation volumes. `rl/vis_eval.py` runs fixed episode sets against every baseline and writes the report the thesis quotes.

**Tech Stack:** Python 3.14 (`.venv`), Gymnasium, stable-baselines3 (SAC), numpy, torch, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-rl-v2-visibility-rlhf-design.md` (sections "Environment", "Training", "Evaluation", "Success criteria").

**Established facts:**
- `goals.py`: `starting_params()`, `sample_instruction(name, model, start_features, rng)` → `{"kind", "text", "targets", "goal"}` (goal vector, 16 values), `aggregate(features)`, `distance(goal, start, current)`, `attainment(goal, start, final)`, `is_useless(features)`, `GOAL_CLASSES = ("skeleton", "lungs", "soft", "vessels")`, `KEEP_TOLERANCE`.
- `visibility.for_volume(name)`: `features(params)` in ~14 ms, `solo_max(class)`, `histogram` (16 bins), `label_source`.
- `datasets.volumes_for_split("train"|"val"|"test"|"out_of_source")`: 20 / 4 / 6 / 4 volumes.
- `rl/baselines.py`: `BASELINES` maps name → `(model, start_params, instruction) -> params`, including `B0_do_nothing`, `B1_current_executor`, `B2_random_policy`, `B3_hill_climb_10`, `B4_hill_climb_200`, `B5_occlusion_rule`.
- Baseline attainment measured on 3 volumes × 30 instructions (tolerance 0.15): do-nothing 0.0 by construction, today's executor ≈ −0.40, occlusion rule ≈ −0.86, 10-evaluation hill-climb ≈ +0.24, 200-evaluation hill-climb ≈ +0.63.
- The action space is 12 values (height, width, brightness per peak); peak centres are fixed, so the starting transfer function must already have a peak per goal class (`goals.starting_params()`).

**Repo rules:** run from the repository root with `.venv/bin/python`; plain commit messages with **no** trailers; stage files explicitly by path; nothing under `out/` is committed; leave untracked `docs/prompts_report.pdf` and `docs/vr-integration.md` alone; branch `rl-v2`.

---

### Task 1: The environment

**Files:** create `rl/vis_env.py`, `tests/test_vis_env.py`.

```python
OBSERVATION_SIZE = 62      # 12 params + 16 goal + 4 log vis + 4 bright + 4 c + 4 b + 1 coverage + 16 histogram + 1 progress
ACTION_SIZE = 12
MAX_STEPS = 10
STEP_SCALE = 0.1           # per action unit, in normalized parameter units
USELESS_PENALTY = 1.0
```

- **`reset()`**: pick a volume uniformly from the env's list; load its cached `VisibilityModel`; start from `goals.starting_params()` plus uniform noise (±0.3 heights, ±0.2 widths, ±0.2 on each peak's r/g/b, clipped); sample an instruction; remember the start features and the start distance.
- **`step(action)`**: clip the action to [−1, 1], scale by `STEP_SCALE`, apply to the 12 controllable values (heights, widths, and a single brightness offset added to each peak's r, g, b), clip parameters to their bounds; recompute features; reward = `previous_distance − current_distance`, minus `USELESS_PENALTY` when `goals.is_useless(features)`; truncate after `MAX_STEPS`.
- **`info`**: `attainment`, `distance`, `kind`, `volume`, `text`, `useless`.
- The observation packs: the 12 controllable values, the goal vector, `log10(vis + EPSILON)` and brightness per goal class, progress `c` and `b` per goal class, coverage, the volume's histogram, and `step / MAX_STEPS`.

- [ ] **Step 1: Write failing tests** (`tests/test_vis_env.py`), using a stub model/volume list so they run in milliseconds:
  - shapes and bounds of observation and action spaces; observation is finite and inside the declared space after reset and after a step;
  - `step` reward equals the drop in `goals.distance` (compute it independently in the test);
  - the useless penalty is applied when a state is useless (force it with an all-transparent transfer function);
  - an episode truncates after exactly `MAX_STEPS` and never terminates early;
  - `reset(seed=...)` twice gives the same volume, instruction and start parameters;
  - actions are clipped: an action of +5 moves parameters no further than an action of +1;
  - parameters stay within [−1, 1] after many large actions;
  - `info["attainment"]` at the final step equals `goals.attainment(goal, start_features, final_features)`;
  - the sampled instruction only targets classes the chosen volume supports.
- [ ] **Step 2–4:** fail, implement, pass.
- [ ] **Step 5: Speed check.** Time 200 environment steps on a real volume: expect ≈ 15 ms per step (accept ≤ 30 ms). Print the measured figure.
- [ ] **Step 6:** Commit `rl/vis_env.py tests/test_vis_env.py` — `feat(rl): instruction-following environment`.

---

### Task 2: Training

**Files:** create `rl/vis_train.py`, `tests/test_vis_train.py`.

- SAC (`stable_baselines3`, `MlpPolicy`, default hyperparameters) on a `VisibilityTFEnv` over `datasets.volumes_for_split("train")`.
- CLI: `--timesteps` (default 200000), `--seed`, `--eval-interval` (default 20000), `--out` (default `out/rl_v2/sac_seed<seed>`).
- Every `eval-interval`: run 40 fixed validation episodes (seeded, from the 4 validation volumes) with the deterministic policy, log mean attainment overall and per instruction kind to `progress.csv` in the run directory, and save a checkpoint. Keep the checkpoint with the best validation attainment as `best.zip`.
- Log to stdout in the same shape the existing `plots/read_progress.py` expects (SB3 CSV logger plus our own `eval_progress.csv`).
- Refuse to overwrite an existing run directory.

- [ ] **Step 1: Write failing tests:** `build_env` returns an env whose volume list is the training split; `evaluate_policy_on(episodes, model)` returns per-episode attainment using the deterministic action; the run directory guard raises on an existing directory; a 200-step smoke run with a stub env produces `eval_progress.csv` with the expected columns.
- [ ] **Step 2–4:** fail, implement, pass.
- [ ] **Step 5: Smoke run.** `.venv/bin/python -m rl.vis_train --timesteps 2000 --eval-interval 1000 --seed 0 --out out/rl_v2/smoke` completes, writes `eval_progress.csv` with 2 rows and a `best.zip`. Report the wall-clock time and the two attainment numbers.
- [ ] **Step 6:** Commit `rl/vis_train.py tests/test_vis_train.py` — `feat(rl): SAC training across the training volumes`.

---

### Task 3: Evaluation against the baselines

**Files:** create `rl/vis_eval.py`, `tests/test_vis_eval.py`.

- `fixed_episodes(split, count, seed)`: a deterministic list of `(volume, start params, instruction)`.
- `run_policy(model_path, episodes)` and `run_baseline(name, episodes)` → attainment per episode.
- `compare(results)`: mean attainment per method overall and per instruction kind, plus a paired Wilcoxon signed-rank test of the policy against each baseline (implement the test directly — `scipy` is not installed — with the normal approximation and a note that ties are dropped; verify against a hand-computed example in the tests).
- CLI writes `out/rl_v2/eval_<split>.json` and prints a table.

- [ ] **Step 1: Write failing tests:** `fixed_episodes` is deterministic and only uses volumes of the requested split; `wilcoxon` matches a hand-computed statistic and p-value on a small example, and returns p = 1.0 when the two series are identical; `compare` aggregates per kind correctly and handles a baseline that fails on an episode.
- [ ] **Step 2–4:** fail, implement, pass.
- [ ] **Step 5:** Run against the smoke policy on 20 validation episodes to prove the path works end to end (the numbers will be poor — a 2000-step policy is untrained).
- [ ] **Step 6:** Commit `rl/vis_eval.py tests/test_vis_eval.py` — `feat(rl): evaluate the policy against the baselines`.

---

### Task 4: The real training run

- [ ] **Step 1:** Launch three seeds in the background, one after another (they compete for CPU otherwise):
  `.venv/bin/python -m rl.vis_train --timesteps 200000 --seed 0 --out out/rl_v2/seed0` (then seeds 1 and 2).
  Expect several hours per seed. Report the wall-clock time of the first seed and its validation attainment curve (the `eval_progress.csv` rows).
- [ ] **Step 2:** If validation attainment is still ≤ 0 after 60000 steps on the first seed, STOP and report: that means the policy is not learning and the controller must look at the reward before burning more compute.
- [ ] **Step 3:** Evaluate every seed's `best.zip` on the TotalSegmentator test subjects and on the out-of-source Slicer CTs, 200 episodes each:
  `.venv/bin/python -m rl.vis_eval --policy out/rl_v2/seed0/best.zip --split test --episodes 200`
- [ ] **Step 4:** Report the full table: mean attainment for the policy and every baseline, overall and per instruction kind, with the Wilcoxon p-values against each baseline, for each seed and each evaluation set.

**Tier-1 success:** the policy beats `B0_do_nothing`, `B1_current_executor`, `B2_random_policy`, `B3_hill_climb_10` and `B5_occlusion_rule` on the test subjects, paired Wilcoxon p < 0.05, for all three seeds. The ratio to `B4_hill_climb_200` is reported, not required — that baseline uses 20× the evaluations the policy gets.

- [ ] **Step 5:** Commit nothing from `out/`; commit only any code fixes made along the way.

---

### Task 5: Curves, figures and write-up

**Files:** modify `plots/training_curves.py` if needed; create `tools/rl_v2_figures.py`; modify `README.md`.

- [ ] **Step 1:** A learning-curve figure (validation attainment against training steps, one line per seed) and a per-instruction-kind bar chart of the final policy against the baselines, written to `plots/output/`.
- [ ] **Step 2:** A qualitative figure: for three test episodes, the rendered start state, the policy's result and `B1`'s result side by side, with the instruction as the caption.
- [ ] **Step 3:** README section reporting the tier-1 result with the real numbers, stating the evaluation protocol (held-out subjects, fixed episodes, paired test) and the honest caveats (out-of-source volumes use intensity labels; organs and muscle are one target; vessels only on contrast scans).
- [ ] **Step 4:** Commit — `docs: report the RL v2 tier-1 result`.
