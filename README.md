# Voice-Driven Transfer Function MVP

Local prototype: synthetic CT phantom (HU) -> 4-Gaussian-peak transfer function ->
offscreen VTK render, driven by rule-based or local-LLM-parsed text/voice commands,
searched by hill-climbing against an exact opacity-mass metric or a human judge.
Camera viewing angle is a second, independent piece of controllable state, adjusted
by the same command layer. Two offline reinforcement-learning sub-projects (see
below) each train a policy to match or beat the hand-coded hill-climbing baseline.

For a full technical writeup of the current architecture (rendering pipeline,
command layer, search/evaluation, camera control, both RL sub-projects, server
session model), see `docs/architecture.typ` (compiled: `docs/architecture.pdf`).

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

## Real CT data

By default everything runs on the synthetic phantom. Pass `--dataset ct_chest` to
use a real, de-identified chest CT (public 3D Slicer test data, genuine Hounsfield
units) instead — no other code changes needed, since the tissue table and transfer
function are already defined in true HU space:

    python mvp.py --dataset ct_chest --cmd "show only bone"
    python server.py --dataset ct_chest              # chat UI on the real scan

The scan (~40MB) downloads once into `data/` (checksum-verified) and is cached
afterward. See `datasets.py` for the registry.

## Chat UI

    python server.py                    # http://127.0.0.1:8000, synthetic phantom
    python server.py --dataset ct_chest # same UI, real CT chest scan

Chat-driven version of the CLI: type or speak (🎤, via the browser mic + the same
Whisper wrapper) a command, watch it render, step back/forward through the
session's history, and judge hill-climb steps as Better/Worse when search is
toggled on with the human evaluator. Camera state (azimuth/elevation/zoom)
travels alongside the transfer function in each history step, so back/forward
navigation restores both together. Session state persists in
`out/ui_session.json`.

## Reinforcement learning

Two Gymnasium environments, trained offline with `stable-baselines3` SAC against
exact metrics computed directly on the data (no rendering, no human labels), each
compared against the hand-coded hill-climbing baseline they're meant to replace.
**The camera-viewpoint policy is not wired into the live loop yet** — the
opacity policy now is (chat UI only, see below); both remain trained and
evaluated offline first.

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

## Files

- `phantom.py` — synthetic CT volume in Hounsfield units
- `transfer.py` — 24-float vector <-> VTK transfer functions, `opacity_mass` metric
- `render.py` — offscreen VTK render, pixel grab, image features
- `camera.py` — camera state (azimuth/elevation/zoom), relative camera commands
- `commands.py` — rule-based parser, Ollama LLM parser, `apply_command`
- `evaluate.py` — objective (`opacity_mass`) and human console evaluators, preference logging
- `asr.py` — faster-whisper push-to-talk mic capture and file transcription
- `mvp.py` — CLI, hill-climbing loop, state persistence
- `eval_parsers.py` — rule vs LLM parser accuracy comparison
- `stats.py` — `out/preferences.jsonl` agreement analysis
- `datasets.py` — real CT dataset registry, checksum-verified download + NRRD loading
- `search.py` — hill-climb search-step math, shared by `mvp.py`, `server.py`, and both RL evaluators
- `server.py` + `static/` — FastAPI chat UI (text/voice commands, history navigation, human judging, camera)
- `rl/env.py`, `rl/train.py`, `rl/eval.py`, `rl/serve.py` — SAC RL for transfer-function opacity (train/eval offline, serve live)
- `rl/camera_env.py`, `rl/camera_train.py`, `rl/camera_eval.py` — offline SAC RL for camera viewpoint
- `docs/architecture.typ` — full architecture reference (compiled: `docs/architecture.pdf`)

The hand-coded hill-climbing baseline (`search.py` + `evaluate.py`) remains
the default live-loop optimizer, and the only one `mvp.py` uses. `server.py`
additionally offers the trained opacity policy as a third search option
(`evaluator=policy`); the camera-viewpoint policy is trained and evaluated
offline only, not yet wired into either loop.
`out/log.jsonl` and `out/preferences.jsonl` are the data this baseline leaves
behind, originally intended for a later learned agent to train on directly;
both RL sub-projects instead trained against exact metrics computed straight
from the data, so that data remains unused by them so far.

## Tests

    .venv/bin/pytest -v
