# RL Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve zero-search policy reliability and small-budget hybrid quality using validation-safe residual actions, search-teacher distillation, and correctly anchored refinement.

**Architecture:** Preserve the existing 57-value observation and 12-group action contract for the primary policy. Add residual behavior as an explicit environment/training variant, add a reproducible teacher dataset and supervised warm-start utility, and extend evaluation to report policy-only and hybrid candidates against the original episode start. Keep test subjects and existing checkpoints unchanged until validation selects a candidate.

**Tech Stack:** Python, NumPy, Gymnasium, Stable-Baselines3 SAC, pytest, existing visibility/goals pipeline, JSONL/JSON experiment artifacts.

---

## File Map

- Modify `rl/baselines.py`: add explicit search-result accounting needed for candidate comparison; retain fixed-order hill-climb as the labeled baseline.
- Modify `rl/oneshot_env.py`: add residual-action decoding as an explicit variant without changing default absolute-action behavior.
- Modify `rl/oneshot_train.py`: support selectable action mode, validation reliability/tail metrics, and optional policy initialization for fine-tuning.
- Modify `rl/vis_eval.py`: add original-start anchored refinement and richer per-episode hybrid details.
- Create `rl/teacher_data.py`: generate training-split-only hill-climb targets with provenance and compound-instruction oversampling.
- Create `rl/distill.py`: train a supervised one-shot policy initialization from teacher observations/actions, then export a checkpoint compatible with SAC fine-tuning.
- Create `tools/rl_experiment_report.py`: compare validation runs using the approved lexicographic selection rule and produce JSON/console reports.
- Modify `tests/test_oneshot_env.py`: residual action behavior and preservation tests.
- Modify `tests/test_oneshot_train.py`: metric logging, action-mode selection, and warm-start tests.
- Modify `tests/test_vis_eval.py`: original-start refinement and evaluation-budget tests.
- Create `tests/test_teacher_data.py`: leakage, deterministic generation, compound coverage, and provenance tests.
- Create `tests/test_distill.py`: observation/action dataset and supervised initialization tests.
- Create `tests/test_rl_experiment_report.py`: selection ordering and metric calculations.

## Task 1: Freeze Validation Baseline

**Files:**
- Create: `tools/rl_experiment_report.py`
- Modify: `tests/test_rl_experiment_report.py` (create if absent)

- [ ] **Step 1: Write failing metric tests**

Test a report helper with synthetic per-episode rows. Require median, improvement rate, clipped mean, worst decile, compound median, compound improvement rate, granted/actual evaluation counts, and latency fields. Assert selection is lexicographic: improvement rate first, median second, worst-tail third, cost fourth.

```python
def test_select_variant_prioritizes_reliability_then_quality():
    candidates = [
        {"name": "quality", "share_positive": .90, "median": .20,
         "worst_decile": -.8, "latency_ms": 10},
        {"name": "reliable", "share_positive": .92, "median": .10,
         "worst_decile": -1.5, "latency_ms": 20},
    ]
    assert select_variant(candidates)["name"] == "reliable"
```

- [ ] **Step 2: Run tests and confirm RED**

```bash
.venv/bin/python -m pytest -q tests/test_rl_experiment_report.py
```

Expected: import failure for the new report helpers.

- [ ] **Step 3: Implement report helpers**

Implement `summarise_episode_rows(rows, refinement_budget=None)` using existing `goals.summarise_attainment`, `np.median`, and the 10th percentile. Compute per-kind summaries from `kind`, and preserve `evaluations_granted`, `evaluations_used`, and `elapsed_ms` when present. Implement `select_variant(rows)` with the exact ordering in the approved spec.

- [ ] **Step 4: Add baseline CLI**

Add a CLI that reads stored JSON result files and writes a report with provenance, validation split, episode seed/count, policy name, and all metrics. It must refuse files whose provenance or split metadata is absent rather than silently comparing incompatible runs.

- [ ] **Step 5: Run tests and freeze baseline artifact**

```bash
.venv/bin/python -m pytest -q tests/test_rl_experiment_report.py
.venv/bin/python -m rl.vis_eval --policy out/rl_v2/oneshot_v3_seed0/best.zip --split val --episodes 200 --seed 0 --formulation one_shot --refine 3 --out out/rl_v2/optimization_baseline_seed0.json
```

Store baseline reports under `out/rl_v2/optimization_baseline_*`; do not alter existing test result files.

## Task 2: Correct Original-Start Hybrid Refinement

**Files:**
- Modify: `rl/vis_eval.py`
- Modify: `tests/test_vis_eval.py`

- [ ] **Step 1: Write failing refinement test**

Construct a fixed episode where the proposal already improves the goal, and a refinement candidate overshoots when the search measures relative to the proposal. Assert the selected output is the better candidate when both are scored against the original start aggregate.

- [ ] **Step 2: Run focused test and confirm RED**

```bash
.venv/bin/python -m pytest -q tests/test_vis_eval.py -k original_start_refinement
```

- [ ] **Step 3: Implement shared candidate scoring**

Add a helper that takes `model`, original `start_params`, original `start_agg`, instruction, proposal params, and refinement budget. It must:

- Score the proposal against original start.
- Run `hill_climb` from proposal with the requested budget.
- Score the refined params against original start.
- Include original start as a candidate.
- Return the highest-attainment candidate plus `evaluations_granted`, `evaluations_used`, and `selected_source`.

Do not change `hill_climb` order in this task. Record its fixed coordinate order in result metadata.

- [ ] **Step 4: Update `run_policy_with_refinement` and tests**

Preserve the existing return shape for callers while adding optional detail output or a parallel detailed helper. Assert budgets 0, 1, 3, and 4 behave deterministically and no refinement can reduce the chosen analytical attainment below the proposal/original-start candidates.

- [ ] **Step 5: Verify**

```bash
.venv/bin/python -m pytest -q tests/test_vis_eval.py
```

## Task 3: Add Residual One-Shot Variant

**Files:**
- Modify: `rl/oneshot_env.py`
- Modify: `rl/oneshot_train.py`
- Modify: `tests/test_oneshot_env.py`
- Modify: `tests/test_oneshot_train.py`

- [ ] **Step 1: Write failing residual-action tests**

Add tests that an action of zero preserves the start transfer function in residual mode, nonzero action changes only controllable group means, colour offsets remain intact, and the residual range can express large `show_only` changes through a configurable bound.

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
.venv/bin/python -m pytest -q tests/test_oneshot_env.py -k residual
```

- [ ] **Step 3: Implement explicit action mode**

Add `action_mode="absolute"` as the default and `action_mode="residual"` as an explicit option. In residual mode decode the policy action as a bounded adjustment from `_start_params`, then call the existing colour-preserving group application. Keep action/observation sizes unchanged and preserve the absolute mode exactly.

- [ ] **Step 4: Thread mode through training and evaluation**

Add `--action-mode` to `rl.oneshot_train`; persist it in run metadata and evaluation output. Add matching mode handling to policy evaluation so residual checkpoints cannot be evaluated under the absolute decoder accidentally.

- [ ] **Step 5: Verify**

```bash
.venv/bin/python -m pytest -q tests/test_oneshot_env.py tests/test_oneshot_train.py tests/test_vis_eval.py
```

## Task 4: Generate Search-Teacher Dataset

**Files:**
- Create: `rl/teacher_data.py`
- Create: `tests/test_teacher_data.py`

- [ ] **Step 1: Write leakage and determinism tests**

Test that generated records use only `datasets.volumes_for_split("train")`, contain no validation/test IDs, reproduce with the same seed, include original start parameters, instruction, teacher parameters, attainment, budget, instruction kind, and provenance.

- [ ] **Step 2: Run tests and confirm RED**

```bash
.venv/bin/python -m pytest -q tests/test_teacher_data.py
```

- [ ] **Step 3: Implement deterministic teacher generation**

Implement a CLI accepting `--episodes`, `--seed`, `--evaluations`, `--compound-ratio`, and `--out`. Generate normal instruction episodes from training volumes, oversample compound instructions by filtering/resampling the instruction stream, run existing `hill_climb`, and store JSONL records:

```json
{
  "volume": "...",
  "start_params": [...],
  "instruction": {...},
  "teacher_params": [...],
  "teacher_action": [...],
  "attainment": 0.0,
  "evaluations_granted": 200,
  "instruction_kind": "compound",
  "split": "train",
  "provenance": {...}
}
```

`teacher_action` must be derived through the shared controllable-group encoding, not duplicated parameter math. Refuse non-train splits.

- [ ] **Step 4: Verify a small dataset**

```bash
.venv/bin/python -m rl.teacher_data --episodes 20 --seed 0 --evaluations 200 --compound-ratio 0.5 --out out/rl_v2/teacher_smoke.jsonl
.venv/bin/python -m pytest -q tests/test_teacher_data.py
```

## Task 5: Supervised Warm-Start and SAC Fine-Tuning

**Files:**
- Create: `rl/distill.py`
- Modify: `rl/oneshot_train.py`
- Create: `tests/test_distill.py`
- Modify: `tests/test_oneshot_train.py`

- [ ] **Step 1: Write failing dataset/training tests**

Test loading teacher JSONL into finite observation/action arrays, rejecting split leakage and malformed action lengths, and invoking a supplied policy initialization before SAC fine-tuning.

- [ ] **Step 2: Run tests and confirm RED**

```bash
.venv/bin/python -m pytest -q tests/test_distill.py
```

- [ ] **Step 3: Implement teacher-to-observation conversion**

Reconstruct the exact one-shot observation with `build_observation`, stored start features, histogram, reachable ceiling, and controllable values. Validate action shape equals `ACTION_SIZE` and teacher split equals `train`.

- [ ] **Step 4: Implement supervised warm-start**

Use a small MLP matching SB3 `MlpPolicy` actor input/output dimensions. Train action regression against teacher actions, save a compatible initialization artifact, and report train loss plus teacher attainment. Do not select using test episodes.

- [ ] **Step 5: Add SAC fine-tuning hook**

Add `--init-policy` to `rl.oneshot_train`. Load the supervised actor parameters when supplied, continue with the existing SAC environment/reward, and preserve checkpoint selection by validation reliability first, then median. Record initialization path and action mode in run metadata.

- [ ] **Step 6: Verify smoke path**

```bash
.venv/bin/python -m pytest -q tests/test_distill.py tests/test_oneshot_train.py
.venv/bin/python -m rl.distill --data out/rl_v2/teacher_smoke.jsonl --out out/rl_v2/distill_smoke
```

## Task 6: Reliability-Aware Validation and Hybrid Evaluation

**Files:**
- Modify: `rl/oneshot_train.py`
- Modify: `rl/vis_eval.py`
- Modify: `tests/test_oneshot_train.py`
- Modify: `tests/test_vis_eval.py`

- [ ] **Step 1: Add validation metrics**

Log overall median, share positive, clipped mean, 10th percentile, compound median, compound share positive, and per-volume summaries at each validation checkpoint. Keep existing CSV columns backward-compatible and add new columns explicitly.

- [ ] **Step 2: Select checkpoint by approved rule**

Use validation improvement rate first, median second, worst decile third. Record the selected checkpoint reason. Do not use final timestep or test results for selection.

- [ ] **Step 3: Add hybrid budgets 1/3/4**

Expose `--refine 1`, `--refine 3`, and `--refine 4`. Store granted and actual evaluations, source selected, elapsed solve time, and instruction kind per episode. Keep fixed coordinate order labeled as `hill_climb_fixed_order`.

- [ ] **Step 4: Verify**

```bash
.venv/bin/python -m pytest -q tests/test_oneshot_train.py tests/test_vis_eval.py tests/test_rl_experiment_report.py
```

## Task 7: Run Validation Experiments

**Files:**
- Generated: `out/rl_v2/optimization_*`
- Generated: `plots/output/optimization_*` if figures are added

- [ ] **Step 1: Run current-policy baseline on validation**

Run fixed validation episodes for existing v3 seed checkpoints. Record policy-only and hybrid budgets 1, 3, and 4. Do not overwrite held-out artifacts.

- [ ] **Step 2: Run short residual ablation**

Train a short residual variant with the same training subjects, seed set, evaluation episodes, and checkpoint interval. Compare reliability first.

- [ ] **Step 3: Generate teacher data and distill**

Generate teacher records from training subjects, warm-start a policy, fine-tune with SAC, and evaluate on validation only.

- [ ] **Step 4: Select candidate**

Apply the lexicographic selection rule. A useful zero-search candidate must reach at least 80% positive episodes without median below 0.275. A useful hybrid must reach at least 90% positive episodes at no more than four evaluations. Treat these as targets, not proof.

## Task 8: Final Held-Out Confirmation

**Files:**
- Generated: `out/rl_v2/optimization_selected_test_*.json`
- Modify: `docs/STATUS.md` and thesis/report only after results exist

- [ ] **Step 1: Freeze selected variant**

Record selected checkpoint, action mode, teacher-data provenance, training seed, validation report, and selection rule before touching test subjects.

- [ ] **Step 2: Run existing 200-episode six-subject test protocol**

Use identical episodes and compare current v3, selected policy, and selected hybrid budgets. Store per-episode details and provenance.

- [ ] **Step 3: Verify and report honestly**

Run the relevant test suites and report whether thresholds were met. If not met, retain the experiment as a negative result and do not replace the current thesis headline.

```bash
.venv/bin/python -m pytest -q tests/test_oneshot_env.py tests/test_oneshot_train.py tests/test_vis_eval.py tests/test_teacher_data.py tests/test_distill.py tests/test_rl_experiment_report.py
```
