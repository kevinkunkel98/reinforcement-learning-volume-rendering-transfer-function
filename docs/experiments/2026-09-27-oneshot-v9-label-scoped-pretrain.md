# One-Shot v9: Label-Scoped TF + Hindsight Pretraining

## Purpose

Find out whether the eight-class one-shot policy can beat doing nothing, and
which of two changes gets it there:

1. **Label-scoped transfer function** (`oneshot-v9`, done): a voxel segmented
   as class c is shaded by c's own peak only. Unlabelled voxels see only the
   skeleton, lungs and soft peaks. Soft, kidneys, heart, spleen and liver all
   sit at 40–65 HU, so the old shared HU lookup could not tell them apart.
2. **Hindsight pretraining** (`--pretrain-data`, new): fit the SAC actor to
   the oracle actions of hindsight episodes before SAC starts. The same
   transitions are also added to the replay buffer.
3. **Brightness hindsight** (`--hindsight-brightness`, new): some hindsight
   goals change one class's brightness rather than its visibility, so
   brightness instructions (the weakest kind) get demonstrations too.

## Results so far (seed 0, 40 validation episodes, not held out)

| | search ceiling (hill-climb 200) | policy, final median | share improved |
|---|---|---|---|
| v8 (old TF + layer actions), 50k | +0.123 (old TF) | −0.371 | 25 % |
| **v9 (label-scoped TF), 50k** | **+0.342** | **−0.011** | **47.5 %** |

v9 was still improving at 50k (mean −0.47 → −0.29 → −0.21 over the last three
evaluations). That is why the runs below use 150k steps.

## Setup on the other machine

The code for this experiment must be on the remote first (`git push` from the
machine that has it).

```bash
git clone https://github.com/kevinkunkel98/reinforcement-learning-volume-rendering-transfer-function.git
cd reinforcement-learning-volume-rendering-transfer-function
git checkout <commit containing this file>
python3.14 -m venv .venv            # this project runs on Python 3.14
.venv/bin/pip install -r requirements.txt
```

**Data is not in git.** Copy these from the original machine, keeping the same
paths:

| path | size | needed |
|---|---|---|
| `data/totalseg/` (30 subjects, `ct.nii.gz` + `labels.nii.gz`) | ~1 GB | **required** |
| `out/cache/visibility/` | ~200 MB | optional; otherwise rebuilt on first use (slow) |
| `out/rl_v4/hindsight_v9_seed0.npz` | small | optional; skips collection for seed 0 **only if** it was collected with the same `--pretrain-episodes` and `--hindsight-brightness` |

```bash
rsync -a --relative data/totalseg out/cache/visibility user@other:path/to/repo/
```

`data/totalseg_manifest.json` is committed. It records each volume's sha256,
so a changed or missing file fails loudly rather than silently.

Sanity check before launching anything (run each command from the repo root;
`PYTHONPATH=.` is required):

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_visibility.py tests/test_oneshot_env.py \
    tests/test_oneshot_train.py tests/test_pretrain.py
PYTHONPATH=. .venv/bin/python -m rl.oneshot_train --short --policy-version oneshot-v9 \
    --action-mode residual --reward-mode target --out out/rl_v4/smoke_v9
```

## Runs

The shared flags are identical in every run. Only the lines marked **differs**
change.

```bash
COMMON="--timesteps 150000 --eval-interval 10000 --eval-episodes 200 \
  --policy-version oneshot-v9 --action-mode residual --reward-mode target \
  --hindsight-ratio 0.25 --balance-classes"
```

| id | what it isolates | differs |
|---|---|---|
| E1 | control: label-scoped TF, no pretraining | — |
| E2 | + hindsight pretraining | `--pretrain-data …` |
| E3 | + brightness hindsight | `--pretrain-data … --hindsight-brightness 0.1` |
| E4 | seeds 1, 2 of the best of E1–E3 | `--seed 1`, `--seed 2` |

```bash
# E1
PYTHONPATH=. nohup .venv/bin/python -m rl.oneshot_train $COMMON --seed 0 \
  --out out/rl_v4/e1_v9_150k_seed0 > out/rl_v4/e1_v9_150k_seed0.log 2>&1 &

# E2
PYTHONPATH=. nohup .venv/bin/python -m rl.oneshot_train $COMMON --seed 0 \
  --pretrain-data out/rl_v4/hindsight_v9_seed0.npz --pretrain-episodes 4000 --pretrain-workers 6 \
  --out out/rl_v4/e2_v9_bc_150k_seed0 > out/rl_v4/e2_v9_bc_150k_seed0.log 2>&1 &

# E3 (its own dataset file: brightness changes what gets collected)
PYTHONPATH=. nohup .venv/bin/python -m rl.oneshot_train $COMMON --seed 0 \
  --hindsight-brightness 0.1 \
  --pretrain-data out/rl_v4/hindsight_v9_bright_seed0.npz --pretrain-episodes 4000 --pretrain-workers 6 \
  --out out/rl_v4/e3_v9_bcbright_150k_seed0 > out/rl_v4/e3_v9_bcbright_150k_seed0.log 2>&1 &

# E4: rerun the winning command with --seed 1 and --seed 2, and give each seed
# its own --out and its own --pretrain-data file (datasets are tied to the seed).
```

Notes:

- **The dataset is cached and checked.** The first E2/E3 run collects the
  hindsight dataset and saves it. Later runs load it, and refuse it if it was
  collected under a different contract, seed, episode count or brightness
  share. Don't start E2 and E3 at the same time on a new file path.
- **Running in parallel:** each run uses all cores for torch by default. With
  several runs at once, prefix each with `OMP_NUM_THREADS=2`, and lower
  `--pretrain-workers` so collection doesn't oversubscribe the machine.
- **Time:** on the original machine, 50k v8 steps took ~1.5 h at ~10 steps/s,
  so plan ~4.5 h per 150k run. Collection time depends on the core count.
  It crawled when the machine was busy.
- **Starting reward** of about −1.3 is normal for v9 (random action −1.37,
  zero action 0). Judge a run by validation median attainment in
  `eval_progress.csv`, not by `ep_rew_mean`.
- With pretraining, `eval_progress.csv` has a **timestep 0 row**: the
  pretrained actor before any SAC step. Also check `metadata.json` →
  `pretrain.val_mse` against `pretrain.zero_mse` (the error of doing nothing).
  If val_mse isn't clearly below zero_mse, the fit learned nothing.

## Evaluation (after each run)

Held-out test split, 200 episodes, against the baselines:

```bash
RUN=out/rl_v4/e2_v9_bc_150k_seed0
PYTHONPATH=. .venv/bin/python -m rl.vis_eval --policy $RUN/best.zip \
  --split test --episodes 200 --seed 0 --formulation one_shot --out $RUN/eval_test.json
PYTHONPATH=. .venv/bin/python -m tools.per_class_eval $RUN/best.zip \
  --split test --episodes 200 --seed 0 --baseline expanded_hill_climb \
  --out $RUN/per_class.json
```

Compare seeds with `tools.compare_runs` once E4 exists.

## Decision gates

- **E2 vs E1:** pretraining is worth keeping if the final validation median
  is higher and the timestep-0 row is already above −0.1. If the timestep-0
  row is good but later rows fall, SAC's early critic updates are undoing
  the pretrained actor. The next step then is to delay actor updates, not to
  drop pretraining.
- **E3 vs E2:** keep brightness hindsight if `attainment_brightness`
  improves and no other kind gets worse.
- **Worth a thesis claim:** held-out median > 0 with share improved > 50 %
  across 3 seeds, reported next to hill-climb 10 and 200.

## Bring back

For each run, copy the whole `out/rl_v4/<run>/` directory. It holds
`metadata.json`, `eval_progress.csv`, `progress.csv`, `best.zip`, the
checkpoints, `eval_test.json` and `per_class.json`. Also copy the matching
`.log` file.
