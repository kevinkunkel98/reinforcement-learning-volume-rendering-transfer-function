# RL v2 Plan 6: One-Shot Transfer-Function Policy

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A policy that reads an instruction, a volume's intensity histogram and its current visibility, and outputs a transfer function **in one step** — trained with RL, evaluated against the search baselines on volumes it never saw.

**Architecture:** `rl/oneshot_env.py` is a one-step Gymnasium environment: reset samples a volume, a start transfer function and an instruction; the action *is* the new transfer function; the reward is the attainment of that output. `rl/oneshot_train.py` trains SAC over the 20 training volumes with validation checkpoints; evaluation reuses `rl/vis_eval.py`'s comparison machinery.

**Why one shot (measured, 2026-09-15/16):** the ten-step formulation does not learn. Across 20 volumes it stayed flat (median −0.13 to −0.28 through 120k steps). Reduced to one volume and one instruction kind it got *worse* with training (−0.15 at 20k → −0.98 at 100k), and neither more gradient steps, lower exploration nor a 10× reward scale fixed it. The reason is structural: a ten-step policy must commit to blind parameter changes, while the hill-climber it is measured against may try a change, measure it and reject it — so the policy was being asked to learn an optimisation procedure rather than the mapping the thesis asks about. In one shot the same SAC improves monotonically on one volume: median −9.77 → +0.08 and share-positive 0 → 60% over 20k steps.

**The claim this supports is stronger than the original:** at inference the policy performs **no** visibility evaluations (it needs the start features, which are one evaluation, and then outputs parameters), while `B3_hill_climb_10` needs 10 and `B4_hill_climb_200` needs 200. Matching B3 with a single forward pass is amortised optimisation; matching B4 would be a strong result.

**Established facts:**
- `goals.py`: `starting_params()`, `sample_instruction`, `aggregate`, `distance`, `attainment`, `is_useless`, `summarise_attainment`, `GOAL_CLASSES`, `EPSILON`.
- `visibility.for_volume(name)`: `features(params)` ≈ 14 ms, `histogram`, `solo_max`.
- `rl/baselines.py`: `CONTROLLABLE` (the 12 controllable parameter groups: height, width and colour per peak) and `BASELINES` (B0…B5). Baseline medians on 90 instructions: B0 0.00, B1 −0.02, B2 −0.23, B3 +0.29, B4 +0.71, B5 −0.64.
- `rl/vis_eval.py`: `fixed_episodes`, `run_baseline`, `wilcoxon`, `compare`.
- One-shot episodes cost ~2 visibility evaluations (reset + step), ≈ 30 ms.

**Repo rules:** run from the repository root with `.venv/bin/python`; plain commit messages with **no** trailers; stage files explicitly by path; nothing under `out/` is committed; leave untracked `docs/prompts_report.pdf` and `docs/vr-integration.md` alone; branch `rl-v2`.

---

### Task 1: One-shot environment

**Files:** create `rl/oneshot_env.py`, `tests/test_oneshot_env.py`.

```python
OBSERVATION_SIZE = 53   # 16 goal + 16 histogram + 4 log10 visibility + 4 brightness + 12 start parameters + 1 coverage
ACTION_SIZE = 12        # the new value of each controllable parameter group, in [-1, 1]
```

- **`reset()`**: volume uniformly from the env's list; start parameters = `goals.starting_params()` with uniform ±0.3 noise per controllable group (clipped); instruction from `goals.sample_instruction`; store the start aggregate.
- **`step(action)`**: the action *is* the new value of each controllable group (not a delta): write it into the transfer function, clipped to [−1, 1]; compute features; `reward = clip(goals.attainment(goal, start, result), -1, 1)`, minus `USELESS_PENALTY = 1.0` when `goals.is_useless(features)`; `terminated = True` always.
- **`info`**: `attainment` (unclipped), `kind`, `volume`, `text`, `useless`.

- [ ] **Step 1: Write failing tests** with a stub model: observation shape/bounds/finiteness; the action writes parameters directly (an action of all zeros yields the mid-range parameter value, not the start value); episodes always terminate after one step; `reward` equals clipped attainment minus any useless penalty; `info["attainment"]` is the unclipped value; `reset(seed=…)` is reproducible; instructions only target classes the volume supports.
- [ ] **Step 2–4:** fail, implement, pass.
- [ ] **Step 5:** Commit — `feat(rl): one-shot transfer-function environment`.

---

### Task 2: Training

**Files:** create `rl/oneshot_train.py`, `tests/test_oneshot_train.py`.

Mirror `rl/vis_train.py`: SAC (`MlpPolicy`, `learning_starts=500`), CLI `--timesteps` (default 150000), `--seed`, `--eval-interval` (default 10000), `--out` (default `out/rl_v2/oneshot_seed<seed>`); validation on 40 fixed episodes from the validation split; `eval_progress.csv` with `timesteps, mean_attainment, median_attainment, mean_clipped_attainment, share_positive` plus per-kind columns; `best.zip` by median; refuse to overwrite a run directory.

- [ ] **Step 1: Write failing tests** (same shape as `tests/test_vis_train.py`: env construction uses the training split, evaluation is deterministic, the run-directory guard, a stub-env smoke run writing the expected CSV columns).
- [ ] **Step 2–4:** fail, implement, pass.
- [ ] **Step 5:** Smoke run: `--timesteps 2000 --eval-interval 1000 --out out/rl_v2/oneshot_smoke`. Report wall-clock and the two rows.
- [ ] **Step 6:** Commit — `feat(rl): train the one-shot policy`.

---

### Task 3: The real run and evaluation

- [ ] **Step 1:** Train seed 0: `.venv/bin/python -m rl.oneshot_train --timesteps 150000 --seed 0 --out out/rl_v2/oneshot_seed0` (≈ 75 min at ~30 ms/episode). Report the validation curve.
- [ ] **Step 2: Decision point.** If validation median has not exceeded **+0.10** by 60000 steps, STOP and report — do not run further seeds. (The single-volume prototype reached +0.08 at 20k while still climbing, so a multi-volume run that is still ≤ 0 at 60k means the mapping is not generalising and the controller must decide.)
- [ ] **Step 3:** If it passes, train seeds 1 and 2 (sequentially — they compete for CPU).
- [ ] **Step 4:** Evaluate every seed's `best.zip` with `rl/vis_eval.py` (extended to drive `OneShotEnv`) on 200 fixed episodes from the TotalSegmentator test subjects and from the out-of-source Slicer CTs, against all six baselines, reporting median, clipped mean, raw mean, share-positive, the per-kind breakdown, and paired Wilcoxon tests.
- [ ] **Step 5:** Report the table. **Tier-1 success:** the policy beats B0, B1, B2 and B5 on the test subjects (p < 0.05, all three seeds), and its relationship to B3 (10 evaluations) and B4 (200 evaluations) is reported honestly, noting that the policy uses one evaluation for its input features and none for search.

---

### Task 4: Figures and write-up

- [ ] **Step 1:** Learning curves (validation median per seed), a bar chart of policy against baselines per instruction kind, and a qualitative figure: for three test episodes, rendered start / policy result / B1 result with the instruction as caption.
- [ ] **Step 2:** README section with the real numbers, the evaluation protocol, and the caveats (organs and muscle are one target; vessels only on contrast scans; out-of-source volumes use intensity labels).
- [ ] **Step 3:** A short "what did not work" section: the ten-step formulation, with the measured curves. This belongs in the thesis — it is the evidence that the one-shot framing is a finding rather than a convenience.
- [ ] **Step 4:** Commit — `docs: report the one-shot policy result`.
