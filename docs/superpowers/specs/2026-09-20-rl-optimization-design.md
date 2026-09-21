# RL Optimization Experiment Design

## Goal

Improve both the zero-search one-shot policy and the delivered policy-plus-small-search result. Optimize reliability first: the primary metric is the share of fixed episodes with positive attainment, followed by median attainment, then latency and worst-tail regression.

The existing checkpoints and held-out test episodes remain untouched until a variant is selected.

## Scope

One evening of experiments. The work is an evaluation and training experiment, not a redesign of the application or a change to the thesis test protocol.

In scope:

- Baseline freeze on fixed validation episodes.
- Short residual-action policy ablation.
- Search-teacher data generated only from training subjects.
- Supervised policy warm-start followed by existing SAC fine-tuning.
- Hybrid evaluation with 1, 3, and 4 refinement evaluations.
- Reliability, attainment, cost, and tail metrics.

Out of scope:

- Selecting variants using held-out test subjects.
- Human-preference training.
- Changing the observation semantics without a separate documented ablation.
- Claiming RL superiority over search.

## Experiment Sequence

### 1. Freeze baseline

Preserve current v3 checkpoints and record policy-alone plus hybrid results on deterministic validation episodes. Store per-episode rows and provenance. Report:

- Improvement rate.
- Median attainment.
- Clipped mean attainment.
- Worst 10% attainment.
- Compound-instruction median and improvement rate.
- Actual visibility evaluations and solve latency.

The current test set is not used for selection.

### 2. Residual one-shot ablation

Add a variant whose action predicts a bounded adjustment from the current transfer function. The final parameters are derived from the start parameters plus the predicted adjustment, while preserving the existing controllable groups and colour-offset handling.

The variant must remain one-step at inference. Large instructions must retain enough action range to express `show only` goals. Compare against the absolute-action policy using the same validation episodes, training budget, and evaluation protocol.

### 3. Search-teacher data

Generate teacher solutions with the existing hill-climber on training subjects only. Store the original start state, instruction, teacher parameters, search budget, scoring code provenance, and per-episode attainment.

Teacher data must include extra compound instructions so the dataset directly targets the known failure mode. No validation or test subject may enter teacher generation.

### 4. Distillation and SAC fine-tuning

Warm-start a one-shot policy with supervised teacher-action training, then fine-tune using the existing SAC environment and reward. Keep the observation and action contracts compatible with the application unless an ablation explicitly records a changed contract.

Select checkpoints on validation reliability, not training reward or final timestep. Compare supervised-only, fine-tuned, and current-policy baselines where compute allows.

### 5. Hybrid refinement

Evaluate each selected policy with 1, 3, and 4 refinement evaluations. Every candidate must be scored against the original start aggregate and original instruction goal. Select the best of the original start, policy proposal, and refined candidates under the shared objective.

The result must report both budget granted and evaluations actually spent. Search order must be recorded; any change from the existing fixed coordinate order is a separate baseline, not an invisible implementation detail.

## Selection Rule

Choose a variant by this lexicographic rule:

1. Highest validation improvement rate.
2. Highest validation median attainment.
3. Lowest worst-tail regression.
4. Lowest solve cost and latency.

A policy-only variant is useful only if improvement rate reaches at least 80% without median attainment falling below the current 0.275 reference. A hybrid result is useful if it reaches at least 90% improvement with no more than four visibility evaluations. These are targets, not claims of equivalence or statistical proof.

## Reporting

Every run records:

- Git commit and scoring fingerprint.
- Training subjects and split.
- Random seeds.
- Training timesteps and checkpoint-selection rule.
- Fixed episode seed and episode count.
- Instruction-kind and per-volume breakdowns.
- Policy-only and hybrid metrics.
- Actual and granted evaluation budgets.

Final held-out evaluation is run only after selecting the variant. It uses the existing 200-episode, six-subject protocol and is reported separately from validation selection results.

## Risks and Controls

- Teacher action non-uniqueness: judge rendered attainment, not action loss alone.
- Search-order bias: retain current order as baseline and label alternative ordering.
- Distribution leakage: generate teacher data only from training subjects.
- Overfitting validation episodes: use fixed validation episodes for comparison, but avoid repeated hyperparameter fishing; retain the test set for final confirmation.
- Policy regression on large goals: evaluate relative, absolute, show-only, brightness, and compound kinds separately.
- Hybrid overclaiming: report hybrid as a separate method, never as zero-search policy quality.
