# Voice-Driven Transfer Function MVP

Local prototype: synthetic CT phantom (HU) -> 4-Gaussian-peak transfer function ->
offscreen VTK render, driven by rule-based or local-LLM-parsed text/voice commands,
searched by hill-climbing against an exact opacity-mass metric or a human judge.

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
reset
```

The third form is an *absolute* level (distinct from increase/decrease's relative
delta) — "high opacity spongy, low opacity bone" sets both in one command. Two or
more in one sentence become a compound command; it's a one-shot assignment, not a
hill-climb search target (there's no single peak, or "wrong direction", to search).

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
toggled on with the human evaluator. Session state persists in
`out/ui_session.json`.

## Files

- `phantom.py` — synthetic CT volume in Hounsfield units
- `transfer.py` — 24-float vector <-> VTK transfer functions, `opacity_mass` metric
- `render.py` — offscreen VTK render, pixel grab, image features
- `commands.py` — rule-based parser, Ollama LLM parser, `apply_command`
- `evaluate.py` — objective (`opacity_mass`) and human console evaluators, preference logging
- `asr.py` — faster-whisper push-to-talk mic capture and file transcription
- `mvp.py` — CLI, hill-climbing loop, state persistence
- `eval_parsers.py` — rule vs LLM parser accuracy comparison
- `stats.py` — `out/preferences.jsonl` agreement analysis
- `datasets.py` — real CT dataset registry, checksum-verified download + NRRD loading
- `search.py` — hill-climb search-step math, shared by `mvp.py` and `server.py`
- `server.py` + `static/` — FastAPI chat UI (text/voice commands, history navigation, human judging)

No MLLM judge, no neural reward model, no torch, no RL — hill-climbing on a binary
verdict is the baseline. `out/log.jsonl` and `out/preferences.jsonl` are the data
this baseline leaves behind for a later learned agent to beat.

## Tests

    .venv/bin/pytest -v
