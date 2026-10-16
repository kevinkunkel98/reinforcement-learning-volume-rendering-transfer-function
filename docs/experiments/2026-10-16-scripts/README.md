# Experiment scripts, 2026-10-16

One-off probes behind `../2026-10-16-overnight-results.md`, kept so every
number there can be re-run. Run from the repository root
(`.venv/bin/python docs/experiments/2026-10-16-scripts/<script> ...`); each
script's docstring has its usage. They are not part of the package or the
test suite. Anything that proved itself was ported: `mb_collect.py` and
`mb_train.py` became `rl/model_based.py`.

| script | result section |
|---|---|
| `start_variety.py` | start-variety probe (organ visibility per start) |
| `scale_probe.py` | probe 1: scaling the SAC action |
| `bc_curve.py`, `bc_eval.py` | probe 2: search-distilled pretraining alone |
| `expert_iter.py` | probe 3: expert iteration |
| `refine_probe.py` | probe 4: SAC proposal + warm search |
| `closed_loop_probe.py`, `closed_loop_train.py` | closed-loop SAC (negative) |
| `mb_collect.py`, `mb_train.py` | model-based RL, first version (now `rl/model_based.py`) |
| `paired_eval.py` | model-based vs SAC E3 on the same 200 val episodes |
| `warm_compare.py` | model-based proposal + warm search (uses `mb_train.py`'s policy.pt format) |
| `ceiling.py` | attainment ceiling (1000-evaluation search) |
| `closed_loop_mb.py` | zero-shot closed loop with a model-based policy |
| `mb_multistep.py` | trained K-step closed-loop model-based policy |
