# Voice-Driven Transfer Function MVP

**RLHF for speech-controlled volume rendering.** Say "show only bone" or "make
the skeleton pop" and a real CT/MRI scan re-renders live — parsed by a rule
engine or a local LLM, optimized by hill-climbing or a trained RL policy, and,
as a research pilot, steered by a reward model learned from human preferences
instead of a hand-coded metric.

<p align="center">
  <img src="docs/screenshots/architecture-highlevel-dark.png" width="85%" alt="Architecture: command layer, transfer-function/render pipeline, RL sub-projects" />
</p>

For the full technical writeup (rendering pipeline, command layer,
search/evaluation, camera control, all RL sub-projects, server session model),
see `docs/architecture.typ` (compiled: `docs/architecture.pdf`) and
`docs/rl-write-test.typ` (math reference, compiled: `docs/rl-math.pdf`).

## Setup

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt

Whisper model downloads once on first use and is cached afterward. The LLM parser
needs a local Ollama server (`ollama serve`, `ollama pull qwen2.5:7b`) — without it,
`--parser llm` automatically falls back to the rule parser and says so.

## Usage

    python mvp.py                                            # render current state -> out/start.png
    python mvp.py --cmd "increase opacity for bone strongly"  # single apply -> before/after.png
    python mvp.py --cmd "..." --learn --steps 15              # hill-climb, objective evaluator
    python mvp.py --cmd "..." --learn --human --steps 10      # hill-climb, human evaluator (logs preferences)
    python mvp.py --listen                                    # push-to-talk -> parse -> apply
    python mvp.py --wav out/audio/xyz.wav                     # replay a prior recording
    python mvp.py --cmd "make the skeleton pop" --parser llm --llm-model qwen2.5:7b

    python eval_parsers.py            # rule vs LLM parser accuracy table
    python stats.py                   # preferences.jsonl agreement analysis

State (the current 24-float transfer-function vector) persists in `out/state.json`
across invocations, so commands compose: run several `--cmd` calls in a row to build
up a view. `reset` (or deleting `out/state.json`) returns to the default vector.

## Commands

Tissues: `bone`, `spongy`, `soft`, `fat`, `air` (each with synonyms both parsers
understand — "cortical"/"skeleton" for bone, "cancellous"/"trabecular" for spongy,
etc. — see `TISSUE_SYNONYMS` in `commands.py`).

```
increase|decrease opacity for <tissue> [slightly|moderately|strongly]
show only <tissue> [and <tissue> ...]              # multi-target isolates all named tissues
<low|medium|high> opacity for <tissue> [, <low|medium|high> opacity for <tissue> ...]
sharpen|soften <tissue>                            # width
brighten|darken <tissue>                           # brightness
shift <tissue>'s center up|down                    # center position, clamped to its own HU band
rotate left|right | tilt up|down | zoom in|out [slightly|moderately|strongly]
reset
```

The third form is an *absolute* level (distinct from increase/decrease's relative
delta) — "high opacity spongy, low opacity bone" sets both in one command. Two or
more in one sentence become a compound command; it's a one-shot assignment, not a
hill-climb search target (there's no single peak, or "wrong direction", to search).

Camera commands (`rotate`/`tilt`/`zoom`) are a fully separate shape from every
other command above — they adjust viewing angle/zoom only and never touch the
transfer function. See `camera.py`.

`--parser llm` understands considerably more free phrasing than this rigid grammar
(e.g. "make the skeleton pop") — see `eval_parsers.py` for a measured comparison.

## Real CT/MRI data

By default everything runs on `mri_head`, a real, de-identified T1 MRI brain
scan (public 3D Slicer test data) — no other code changes needed, the tissue
table and transfer function already cover its rescaled range. Pass
`--dataset ct_chest` (or `ct_skull`, `ct_cardio`, `ct_abdomen`) for a real CT
scan instead, or `--dataset synthetic` for the old fast/offline phantom
(still used internally by the RL training scripts — see
[Reinforcement learning](#reinforcement-learning)):

    python mvp.py --dataset ct_chest --cmd "show only bone"
    python server.py --dataset ct_chest              # chat UI on a real CT scan

Each scan (6-60MB depending on dataset) downloads once into `data/`
(checksum-verified) and is cached afterward. See `datasets.py` for the
registry.

**MRI caveat:** MRI has no calibrated Hounsfield-unit scale, and T1's own
brightness ordering differs from CT's (skull is *dark* in T1, not the
brightest structure as in CT). `mri_head`'s intensities are linearly
rescaled onto the same numeric range the tissue bands already use, so every
command mechanically works and the render looks correct — but tissue
*labels* ("bone", "fat", ...) aren't radiologically accurate for this
dataset (e.g. "show only bone" surfaces bright skin/fat signal, not the
actual skull). See `datasets.py`'s module docstring.

## Chat UI

    python server.py                    # http://127.0.0.1:8000, real MRI scan
    python server.py --dataset ct_chest # same UI, real CT chest scan

Chat-driven version of the CLI: type or speak (🎤, via the browser mic + the same
Whisper wrapper) a command, watch it render, step back/forward through the
session's history, and judge hill-climb steps as Better/Worse when search is
toggled on with the human evaluator. Camera state (azimuth/elevation/zoom)
travels alongside the transfer function in each history step, so back/forward
navigation restores both together. Session state persists in
`out/ui_session.json`.

<p align="center">
  <img src="docs/screenshots/chat-ui-ct-skull.png" width="70%" alt="Chat UI on a real CT skull scan, bone tissue isolated" />
</p>

### Running in the background, and checking it's actually up

    .venv/bin/python server.py > /tmp/server.log 2>&1 &   # background, logs to a file
    curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/   # expect 200
    lsof -i :8000 -sTCP:LISTEN                              # confirm something's listening
    tail -f /tmp/server.log                                 # watch logs live

**Must be started from the repo root.** `server.py` uses relative paths (`out/`,
`static/`) — if it's started from anywhere else (e.g. a git worktree that later
gets removed), every request 500s with `FileNotFoundError: 'out'` or
`RuntimeError: File at path static/index.html does not exist`. This can look
like a browser/mic problem in the UI (push-to-talk silently "does nothing")
when the real issue is the server itself is down — always `curl` the root path
first before assuming it's a client-side issue.

To stop a running (or broken) server:

    pkill -f "server\.py"        # or: lsof -i :8000, then kill <pid>

### Debugging the LLM parser (Ollama)

    ollama serve                                # start Ollama if not already running
    curl -s http://localhost:11434/api/tags     # confirm it's reachable
    ollama ps                                   # which model is loaded, and for how long
    ollama list                                 # installed models

`parse_command_llm` sets `keep_alive: "30m"` so the model stays resident
between commands — but the *first* command of a session (or the first after
30 minutes idle) still pays an ~8-10s cold-start reload; that's expected, not
a bug. `ollama stop <model>` unloads a resident model immediately (e.g. to
free RAM) at the cost of the next command paying that reload again.

    python eval_parsers.py --llm-model qwen2.5:7b   # rule vs LLM parser accuracy,
                                                       # data/parser_eval_phrases.json

## Reinforcement learning

Two Gymnasium environments, trained offline with `stable-baselines3` SAC against
exact metrics computed directly on the data (no rendering, no human labels), each
compared against the hand-coded hill-climbing baseline they're meant to replace.
**The camera-viewpoint policy is not wired into the live loop yet** — the
opacity policy now is (chat UI only, see below); both remain trained and
evaluated offline first. Two further pieces build on the opacity agent: a
continual **online-learning** variant, and an **RLHF** pilot that replaces the
automatic metric with a reward model learned from human judgments.

- **`rl/env.py` + `rl/train.py` + `rl/eval.py`** — transfer-function opacity.
  One episode = one (target tissue, direction) goal; the agent moves the
  resolved peak's height; reward = Δ`mass_fraction`. Result (200k timesteps,
  20 held-out episodes): policy `0.344` vs. hill-climb `0.347` mean final mass
  fraction, `3.3` vs `2.7` steps to 90% of best — near-parity, a well-tuned
  hand controller is a strong baseline on this low-dimensional problem.

      python -m rl.train --timesteps 200000
      python -m rl.eval

  The trained policy is also wired live into the chat UI's search toggle
  (`server.py`, evaluator = policy — no CLI equivalent) — see `rl/serve.py`.

- **`rl/camera_env.py` + `rl/camera_train.py` + `rl/camera_eval.py`** — camera
  viewpoint. One episode = one target tissue on a real CT volume (`ct_skull`);
  the agent moves azimuth/elevation to maximize alignment with that tissue's
  centroid direction from the volume center (a directional proxy — no
  occlusion model). Result: policy `0.997` vs. hill-climb `1.000` mean final
  alignment, `3.05` vs `4.05` steps to 90% — both reach near-perfect alignment,
  the policy converges faster.

      python -m rl.camera_train --timesteps 200000
      python -m rl.camera_eval

See `docs/architecture.typ` §6-7 for full detail on both, including two real
bugs found and fixed during development (a `_resolve_target` fallback that
could silently return no direction, and a shared step-resizing utility whose
opacity-domain-tuned growth cap crippled the camera hill-climb baseline until
it was made domain-aware).

### Online learning

`rl/online_train.py` trains a *fresh* opacity agent continuously on a single
environment instead of one 200k-step batch across 4 parallel envs — chunked
`model.learn()` calls, re-evaluated against the same 20 held-out episodes
after each chunk, so the held-out metric's improvement over training is
directly observable rather than only inferred from SAC's own reward curve.

    python -m rl.online_train --timesteps 50000 --eval-interval 5000
    python -m plots.online_eval_curve   # held-out metric vs. offline baselines

Result (50k timesteps, single env — 25% of the offline run's environment-step
budget): `mean_final_mass_fraction` climbs `0.199` (5k) → `0.347` (15k) →
`0.346` (50k), landing almost exactly between the offline policy (`0.344`)
and hill-climb (`0.347`) baselines, with steps-to-90%-of-best falling from
`17.4` to `3.2` — a clean, converging learning curve on a quarter of the
offline budget, not a coincidence.

### RLHF: learning from human feedback

`mass_fraction` measures whether a transfer-function change follows the
literal command. The goal-conditioned reward model adds human preference over
the rendered result without losing that objective signal. Each observation is
an 18-value `float32` vector:

```text
[features_after(4), features_before(4), after-before(4), target_onehot(5), direction(1)]
```

The four image features are `mean`, `std`, `coverage`, and `entropy`. Target
one-hot order is `air`, `fat`, `soft`, `spongy`, `bone`; direction is the
requested increase/decrease sign. `RewardModel` is an `18 -> 64 -> 32 -> 1`
MLP whose scalar output is a logit. Pair training uses weighted Bradley-Terry
loss, `weight * -log(sigmoid(R(preferred) - R(other)))`, rather than the old
10-input binary-cross-entropy pilot.

#### Preference data contract

Web and VR clients write canonical JSONL records. One pair record has this
shape (one JSON object per line):

```json
{
  "observation_a": {
    "before_features": {}, "after_features": {},
    "before_image": null, "after_image": null
  },
  "observation_b": {
    "before_features": {}, "after_features": {},
    "before_image": null, "after_image": null
  },
  "command": {"attribute": "opacity", "target": "bone", "direction": "increase"},
  "target_tissue": "bone",
  "direction": "increase",
  "label": 1,
  "source": "branch",
  "weight": 1.0
}
```

`label=1` means A is preferred; `label=-1` means B is preferred. Client logs
may additionally carry `session_id`, `episode_id`, `step_id`,
`parent_step_id`, `branch_id`, `carried_forward`, `accepted`, and `ended`.
Feature dictionaries are authoritative. PNG/JPEG paths are audit artifacts:
they support visual review and recovery of missing legacy features through
Pillow, but training consumes extracted features, not pixels. Missing image
artifacts are skipped and counted.

#### Extract, train, evaluate

Extract current logs, feedback, and canonical web/VR preferences without
mutating source files:

```bash
python -m rl.extract_pairs \
  --log out/rl_logs.jsonl \
  --feedback out/feedback.jsonl \
  --preferences out/web_vr_preferences.jsonl \
  --out out/pairs.jsonl
```

The extractor reports counts by source, target, skipped rows, and total. Branch
pairs require explicit parent/child and carried-forward metadata, or equivalent
history metadata. It never guesses branches from parameter similarity. If no
branch is recoverable, it reports `branch=0` and continues with valid
within-episode (`weight=0.7`) and thumbs (`weight=0.4`) pairs. Branch pairs
have `weight=1.0`.

Pretrain on synthetic rendered pairs, then fine-tune separate artifacts on a
fixed seeded human split:

```bash
python -m rl.pretrain_reward \
  --pairs 20000 --members 5 \
  --out out/rl_models/reward_pretrained.pt

python -m rl.finetune_reward \
  --preferences out/pairs.jsonl \
  --pretrained out/rl_models/reward_pretrained.pt \
  --out out/rl_models/reward_finetuned.pt

python -m rl.eval_reward \
  --preferences out/pairs.jsonl \
  --pretrained out/rl_models/reward_pretrained.pt \
  --finetuned out/rl_models/reward_finetuned.pt \
  --disagreements out/reward_disagreements.jsonl \
  --out out/reward_report.json
```

`reward_pretrained.pt` and its member checkpoints are never overwritten by
fine-tuning. Fine-tuning writes `reward_finetuned.pt` and separate member
checkpoints, trains only on the persisted train rows, and evaluates both model
families on exactly the same persisted test rows. Missing logs, checkpoints,
or render artifacts produce an explicit error or an empty summary; results
are never fabricated.

#### Reward safeguards and independent comparisons

`RewardModelTFEnv` combines ensemble reward and the automatic objective:

```text
r = alpha * (mean(model_reward) - std(model_reward))
    + (1 - alpha) * objective_reward
```

Default `alpha` is `0.7`. Run identical evaluation episodes for the anchor
settings `alpha=0.0`, `alpha=0.7`, and `alpha=1.0`; record each result
separately. The near-black coverage and near-opaque mean-opacity hard
penalties apply independently of model output. Alpha ablations are an
evaluation comparison, not new training data.

The final report keeps three results separate:

1. Pretrained versus fine-tuned reward-only accuracy, overall and by source,
   on the fixed held-out human split.
2. RLHF policy versus hill-climber objective performance, as a reward-hacking
   sanity check.
3. Blind head-to-head policy versus hill-climber rendering, with seeded random
   A/B order and verdicts written to a separate file.

Use `evaluate_rl_policy_against_hill_climb()` for the objective sanity check
and `collect_blind_head_to_head()` for the blind comparison. The latter takes
render callbacks and optional `better`/`worse`/`tie` verdicts, writes records
such as `out/blind_head_to_head.jsonl`, and never sends verdicts to extraction
or training. Store alpha results, reward metrics, disagreement records
(`out/reward_disagreements.jsonl`), and blind verdicts independently in the
final report; do not merge blind judgments back into `out/pairs.jsonl`.

## Project layout

The core pipeline (dataset loading, transfer function, VTK render, camera,
command parsing, hill-climb search, CLI/chat-UI) lives at the repo root;
`rl/` holds both offline RL sub-projects plus the online-learning and RLHF
extensions; `plots/` regenerates the training-curve figures. See
`docs/architecture.typ` for the full component-by-component breakdown.

## Tests

    .venv/bin/pytest -v
