# Thesis-Clean Training Curves

## Goal

Upgrade the RL training figure from a single-run diagnostic plot into a publication-ready thesis figure that answers three questions:

1. Did validation performance improve?
2. When was the best checkpoint selected?
3. How stable was training across the three seeds?

The figure must remain grounded in stored SB3 logs and must not imply that the final checkpoint is the selected checkpoint or that validation performance estimates held-out performance.

## Visual Design

Use the existing thesis palette and serif styling from `plots.style`:

- Light background.
- Navy title and structural rules.
- Blue, sage, and amber for the three seed curves.
- Rust marker for selected checkpoints.
- Thin grey gridlines and restrained decoration.

The main figure is a two-column layout:

- Left, larger panel: validation attainment over timesteps, one line per seed, with a marker at each seed's best validation checkpoint.
- Right, smaller panel: compact run summary showing best validation median, selected timestep, final validation median, and seed spread.

The figure caption or subtitle must state that validation episodes are fixed and that checkpoints are selected by best validation median. It must not call the validation curves test performance.

## Data

Read existing `eval_progress.csv` files from `oneshot_v3_seed*` run directories. The loader must tolerate normal SB3 CSV formatting and missing values.

Expected validation columns should be discovered by column name rather than positional order. Support the existing per-kind validation columns where present, but do not fabricate values when a run lacks them.

The summary panel derives:

- Best validation median per seed.
- Timestep of that best value.
- Final validation median.
- Seed spread, defined as the range or equivalent clearly-labelled spread of best validation medians.

If an input run lacks required validation data, fail with a clear error naming the run and missing columns.

## Outputs

Preserve the existing command shape:

```bash
python -m plots.training_curves
```

Preserve `--run-dir` for single-run diagnostics. Add an option or automatic mode for the three-seed thesis figure without breaking the existing single-run output. Write publication exports as PDF and PNG under the requested output directory.

Keep the existing detailed single-run diagnostics available as a secondary output containing reward, actor loss, critic loss, and entropy coefficient. Do not make the thesis figure depend on training-time TensorBoard state.

## Testing

Add tests that:

- Render the multi-seed figure from small synthetic `eval_progress.csv` fixtures.
- Verify PDF and PNG output exists and is non-empty.
- Verify best checkpoint selection uses the validation maximum, not the final row.
- Verify seed ordering is deterministic.
- Verify missing required columns produce an actionable error.
- Preserve existing single-run plot tests.

Tests should inspect structured helper results for selection logic and file existence, not pixel values.

## Scope Limits

- No change to RL training or checkpoint files.
- No browser UI integration in this change.
- No new plotting dependency beyond the current matplotlib/numpy stack.
- No smoothing that could hide instability; if curves are smoothed for readability, retain raw checkpoint markers or disclose the smoothing explicitly. Prefer raw curves initially.

## Acceptance Criteria

- Default command produces the thesis-clean three-seed figure from current v3 runs.
- Figure visibly foregrounds validation attainment and selected checkpoints.
- Seed spread is readable without a separate manual calculation.
- Secondary diagnostic plot remains available.
- Existing plotting tests and new tests pass.
