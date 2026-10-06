#set page(numbering: "1", margin: (x: 2.5cm, y: 2.5cm))
#set text(size: 10.5pt, lang: "en")
#set heading(numbering: "1.")
#set par(justify: true, leading: 0.65em)
#show raw: set text(size: 9pt)
#show raw.where(block: true): block.with(
  fill: rgb("#f4f6f9"), inset: 8pt, radius: 3pt, width: 100%,
)
#show link: underline

#let navy = rgb("#1c3a5e")
#let amber = rgb("#b7770d")
#let sage = rgb("#1e6b3c")

#let note(body, title: "Note", color: amber) = block(
  width: 100%, inset: (x: 0.9em, y: 0.6em), radius: 3pt, fill: rgb("#fef9ec"),
  stroke: (left: 3pt + color),
)[#text(weight: "bold", fill: color, size: 0.9em)[#title:] #h(0.4em) #body]

#let finding(body) = note(body, title: "Current default", color: sage)

#align(center)[
  #text(size: 18pt, weight: "bold", fill: navy)[RL Training Setup] \
  #v(5pt)
  #text(size: 11.5pt)[Transfer function, task, reward, and policy versions as of 2026-10-06] \
  #v(9pt)
  #text(size: 9.5pt, fill: luma(90))[
    Kevin Kunkel #sym.dot.c Master's thesis #sym.dot.c Technical note
  ]
]

#v(12pt)
#note[This is a snapshot of the current one-shot RL setup (`rl/oneshot_env.py`,
`rl/oneshot_train.py`, `transfer.py`, `goals.py`). It supersedes the TF
description in `transfer-function.typ` (written for an earlier, four-peak,
non-label-scoped version) wherever the two disagree. Code is the source of
truth; this note just collects it in one place.]

= The transfer function

Eight Gaussian peaks, one per anatomical class, each with 6 parameters
(center, width, height, r, g, b) $=$ 48 values total (`transfer.py`,
`N_PEAKS = 8`, `PARAMS_PER_PEAK = 6`). Peak centers are fixed at calibrated
HU values; the policy only ever changes width, height, and colour:

#table(
  columns: (2.2cm, 1.6cm, 1.6cm, 1.6cm, 1fr),
  align: (left, right, right, right, left),
  table.header([*Class*], [*Center (HU)*], [*Width (HU)*], [*Height*], [*Notes*]),
  [lungs], [-800], [60], [0.05], [],
  [soft], [40], [80], [0.15], [generic soft tissue / muscle],
  [liver], [65], [45], [0.22], [promoted organ],
  [kidneys], [45], [45], [0.24], [promoted organ],
  [spleen], [55], [45], [0.20], [promoted organ],
  [heart], [50], [50], [0.25], [promoted organ],
  [vessels], [300], [80], [0.30], [only a goal on contrast volumes],
  [skeleton], [900], [280], [0.60], [width capped at 150 HU for "show only"],
)

*Controllable action space:* each peak contributes 3 groups -- width,
height, and brightness (r, g, b moved together) -- for $8 times 3 = 24$
controllable groups (`rl/baselines.py:CONTROLLABLE`). An action group's value
is the *mean* of its channels; only the (r, g, b) group keeps internal
offsets (hue) across an edit.

*Shading mode:* `transfer.DEFAULT_TRANSFER_MODE = "label_scoped"`. A voxel
segmented as class $c$ is shaded by $c$'s own peak only; unlabeled voxels see
only the skeleton/lungs/soft peaks. This replaced an earlier shared-HU-lookup
mode under which soft tissue, kidneys, heart, spleen, and liver (all
40--65 HU) could not be told apart -- the change that produced `oneshot-v9`.

= The task

One instruction, one action, done -- every episode is exactly one `reset()`
followed by one `step()` (`terminated = True` unconditionally). An
instruction names 1--2 of the 8 canonical classes and asks for a visibility
and/or brightness change on each (`goals.sample_instruction`); the mix is

#table(
  columns: (3.2cm, 1fr),
  table.header([*kind*], [*share*]),
  [relative ("more/less X")], [40%],
  [compound (2 classes)], [25%],
  [show\_only], [15%],
  [absolute ("X at medium level")], [10%],
  [brightness], [10%],
)

The *goal vector* is 32 values: requested visibility change, a
visibility-mentioned flag, requested brightness change, and a
brightness-mentioned flag, each per class (4 $times$ 8). The *observation*
(`build_observation`) packs goal (32) + intensity histogram (16) + log10
visibility (8) + brightness (8) + log10 solo-max ceiling (8) + the 24
controllable start values + coverage (1) $=$ *97* values (105 for v8, which
appends 8 anatomy-layer opacities).

== Reward

Two modes:

- *`attainment`* (v6 only): $1 - "distance"("final") / "distance"("start")$.
  1 = goal reached, 0 = no better than doing nothing, $< 0$ = worse.
- *`target`* (v7/v8/v9): the fraction of `start_distance` closed by the
  action, minus a drift penalty on unmentioned classes, both divided by the
  same `start_distance` so a goal's own difficulty doesn't change how much
  of the $[-1, 1]$ reward budget a given fractional outcome spends. (Fixed
  2026-10-03 -- the drift term used to be charged in absolute units while
  progress was already a fraction, which let a `show_only` goal's large
  `start_distance` absorb drift almost for free while squeezing small-goal
  instructions like `brightness` toward $-1$.)

`distance()` itself is a weighted sum over mentioned-class error plus a
"keep" penalty (`LAMBDA_KEEP`, tolerance `KEEP_TOLERANCE`) on unmentioned
classes and on unlabeled ("other") tissue, so hiding behind unclassified
material or wrecking an unmentioned organ both cost something even when the
named instruction is satisfied.

== Hindsight episodes

A share of episodes (`--hindsight-ratio`) skip instruction sampling: draw a
*reachable* target transfer function by perturbing the start state, measure
what it actually changed, and make that the instruction -- so the episode
always has a demonstrated solution (`OneShotEnv.hindsight_policy_action()`).
A further share (`--hindsight-brightness`) make that demonstration a
brightness-only change on one visible class, since brightness instructions
are otherwise the weakest-supervised kind.

= Policy versions

#table(
  columns: (1.6cm, 2cm, 2cm, 1.6cm, 1.6cm, 1fr),
  align: (left, center, center, right, right, left),
  table.header([*version*], [*action mode*], [*reward*], [*obs*], [*action*], [*adds*]),
  [v6], [absolute], [attainment], [97], [24], [baseline; action = the TF directly],
  [v7], [residual], [target], [97], [24], [action = offset from episode start],
  [v8], [residual], [target], [105], [32], [per-class anatomy-layer opacity actions],
  [v9], [residual], [target], [97], [24], [label-scoped shading (contract-checked at load)],
)

"residual" actions are offsets from the episode's own (noised) start state,
not the absolute TF -- `apply_controllable(start_params, start_controllable + action)`.
v8 additionally controls an opacity multiplier per anatomy layer, independent
of the TF peaks. v9 is architecturally identical to v7, but
`resolve_policy_metadata` refuses to load a v9 checkpoint whose recorded
`transfer_mode` doesn't match the code's current `DEFAULT_TRANSFER_MODE` --
the label-scoped shading is a contract, not a hint.

#finding[Current experiments and `main` training runs target
*`oneshot-v9`*, `--action-mode residual`, `--reward-mode target`.]

= Training loop

`rl/oneshot_train.py`: Stable-Baselines3 `SAC("MlpPolicy", ...)`,
`learning_starts = 500` (SB3's own default is 100 -- a few more random
one-step episodes before learning from a near-empty replay buffer),
evaluated every `--eval-interval` steps on a fixed, seeded set of validation
episodes (median attainment selects `best.zip`, robust to single-episode
outliers a mean is not).

== Supervised warm start (`rl/pretrain.py`)

With `--pretrain-data`, hindsight episodes are collected up front
(`pretrain.collect_hindsight`, cached to `.npz`, multiprocess), then:

- `pretrain_actor`: regress SAC's own actor (not a separate network) onto
  the hindsight oracle actions by MSE, before any SAC step.
- `prefill_replay_buffer`: add the same transitions, with their true reward,
  to the replay buffer, so the critic's first updates see good actions
  rather than only the random warm-up.

An eval row at timestep 0 records what the warm start alone achieves,
against `val_mse`/`zero_mse` in `metadata.json` (if `val_mse` isn't clearly
below "always predict zero", the fit learned nothing).

== A representative run

```bash
.venv/bin/python -m rl.oneshot_train \
  --timesteps 150000 --eval-interval 10000 --eval-episodes 200 \
  --policy-version oneshot-v9 --action-mode residual --reward-mode target \
  --hindsight-ratio 0.25 --balance-classes \
  --pretrain-data out/rl_v4/hindsight_v9_seed0.npz \
  --pretrain-episodes 4000 --pretrain-workers 6 \
  --seed 0 --out out/rl_v4/e1_v9_150k_seed0
```

`--balance-classes` biases real-instruction sampling toward under-trained
goal classes (`OneShotEnv._class_counts`). `--hindsight-brightness 0.1`
mixes in brightness-only hindsight demonstrations. Env-interaction throughput
(not counting SAC's own gradient updates) is currently
~20--27 episodes/s on an M1 Pro, after the 2026-10-06 `visibility.py`
performance pass (see
`docs/superpowers/plans/2026-10-06-visibility-features-throughput.md`).

#note[For the fuller experimental narrative (why v9, hindsight pretraining
results so far, per-run setup for a second machine) see
`docs/experiments/2026-09-27-oneshot-v9-label-scoped-pretrain.md`. For known
capability gaps (`show_only`, compound instructions, out-of-source
generalization) see `docs/STATUS.md` -- written mid-way through the v4/v8
work, so read it as directional, not current numbers.]
