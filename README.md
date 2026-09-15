# Voice-Driven Transfer Function MVP

**Speech-controlled volume rendering, toward a learned transfer-function agent.**
Say "show only bone" or "increase opacity for bone strongly" and a real CT/MRI
scan re-renders live — parsed by a rule engine or a local LLM and executed by
exact commands or hill-climbing search. RL v2 (in progress) adds a
goal-conditioned policy for perceptual instructions, refined with human
preferences.

<p align="center">
  <img src="docs/screenshots/architecture-highlevel-dark.png" width="85%" alt="Project architecture" />
</p>

Detailed architecture and mathematical notes:

- `docs/architecture.typ` and `docs/architecture.pdf`
- `docs/rl-write-test.typ` and `docs/rl-math.pdf`

## Setup

Use a virtual environment. Install dependencies with the same Python that will
run the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

First Whisper use downloads and caches a model. LLM parsing requires Ollama:

```bash
ollama serve
ollama pull qwen2.5:7b
```

Without Ollama, the `llm` parser (UI parser toggle) falls back to the rule parser.

## Quick Start

```bash
python server.py
python server.py --dataset ct_chest
```

The web UI runs at `http://127.0.0.1:8000`. Start commands from repository
root because the application uses relative paths such as `out/` and `static/`.

The UI session persists in `out/ui_session.json`. Delete that file or say
`reset` to return to the default transfer function.

<p align="center">
  <img src="docs/screenshots/chat-ui-ct-skull.png" width="70%" alt="Chat UI on a real CT skull scan, bone tissue isolated" />
</p>

## Commands

```text
increase|decrease opacity for <tissue> [slightly|moderately|strongly]
show only <tissue> [and <tissue> ...]
<low|medium|high> opacity for <tissue>
sharpen|soften <tissue>
brighten|darken <tissue>
shift <tissue>'s center up|down
rotate left|right | tilt up|down | zoom in|out
reset
```

Supported tissues: `bone`, `spongy`, `soft`, `fat`, and `air`. See
`commands.py` for synonyms and the complete grammar. Camera commands change
the view and do not change the transfer function.

Useful evaluation commands:

```bash
python eval_parsers.py
python eval_parsers.py --llm-model qwen2.5:7b
```

## Data And UI

Default dataset is `mri_head`. Other datasets include `ct_chest`, `ct_skull`,
`ct_cardio`, `ct_abdomen`, and `synthetic`.

```bash
python server.py --dataset ct_chest
```

Datasets are downloaded into `data/` and cached. MRI intensities are rescaled
to the transfer-function range, so tissue names are functional labels rather
than guaranteed radiological labels. Use CT data for radiological claims.

The web UI supports:

- Text and voice commands
- Back/forward navigation through session history
- Objective hill-climb search for opacity commands
- Camera state stored with each history step

## Local 3D Viewer

The web UI uses a local `vtk.js` volume renderer when the local viewer is
available. Start the server as usual:

```bash
python server.py
```

After a dataset is selected, the browser fetches normalized dataset metadata
from `/api/datasets/<name>/metadata`, then downloads the binary `float32`
volume from `/api/datasets/<name>/chunks/<index>`. Metadata includes dimensions,
spacing, intensity range, orientation/version, byte order, storage order, and
chunk descriptors. The browser validates the descriptors and reconstructs the
volume before creating vtk.js image data. Existing NRRD files and the Python
loader remain the source of truth; the transport avoids requiring NRRD parsing
in the browser and preserves the loader's orientation and MRI rescaling rules.

Camera interaction and transfer-function changes are browser-owned. vtk.js
renders these changes locally after the initial volume upload; they do not
request a new server-rendered frame. The transfer function remains the shared
24-value vector used by the Python renderer and RL pipeline. Camera state uses
the renderer-neutral form `position`, `focal_point`, `view_up`, and `zoom`.

If metadata or chunk validation fails, or the local viewer is disabled, the UI
keeps the existing PNG renderer available through its fallback mode. PNG images
and image features are legacy compatibility, audit, blind-evaluation, and
recovery fields. They are not the target visual representation for new local
viewer preference data.

### Scene transitions

Each committed browser state is logged through `POST /api/scenes/transition`
as renderer-neutral JSON. A transition links `after.parent_scene_id` to the
previous `before.scene_id`; returning to an earlier state therefore records an
explicit branch rather than relying on similar parameters. A typical record
contains:

```json
{
  "scene_id": "web:session-123:scene:7",
  "parent_scene_id": "web:session-123:scene:6",
  "session_id": "session-123",
  "client": "web",
  "dataset": "ct_cardio",
  "dataset_version": "sha256:...",
  "volume": {
    "dimensions": [512, 512, 300],
    "spacing": [0.7, 0.7, 1.0],
    "scalar_type": "float32",
    "orientation": "dataset-normalized"
  },
  "transfer_function": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "camera": {
    "position": [0.0, 0.0, 1.0],
    "focal_point": [0.0, 0.0, 0.0],
    "view_up": [0.0, 1.0, 0.0],
    "zoom": 1.0
  },
  "goal": {"target": "bone", "direction": "increase"},
  "command": {"attribute": "opacity", "target": "bone", "direction": "increase"}
}
```

The canonical schema accepts `client` values `web` and `vrui`. Both clients
share dataset/version, volume, camera, transfer-function, goal, command, and
parent-link fields. Client-specific details belong in `client_metadata`; VR
hardware integration is not required for the current web viewer. Scene logs
are written to `out/scene_transitions.jsonl` and do not contain binary volume
chunks.

## Reinforcement Learning (v2, in progress)

The previous RL pipeline (a height-only SAC agent trained on `mass_fraction`,
camera RL, and a reward model on four global image statistics) has been
removed: its reward did not measure what is visible on screen.

RL v2 trains one goal-conditioned policy for perceptual instructions
(relative, compound, absolute, show only, brightness) against a per-tissue
visibility estimate on real CT volumes, then refines it with human A/B
preferences collected on a dedicated page. Design:
[`docs/superpowers/specs/2026-09-15-rl-v2-visibility-rlhf-design.md`](docs/superpowers/specs/2026-09-15-rl-v2-visibility-rlhf-design.md).

## Tests

```bash
python -m pytest -q -m "not slow"
python -m pytest -q
```

## Project Layout

- Root files: rendering, transfer functions, parser, and web UI server
- `static/`: web UI and local 3D viewer
- `rl/`: RL v2 (in progress)
- `plots/`: training-curve plotting
- `tools/`: maintenance scripts (e.g. `COMMANDS.md` generator)
- `data/`: datasets and parser evaluation phrases
- `docs/`, `slides/`: architecture notes, thesis material, design specs and plans
- `out/`: runtime state, logs, images, models, and reports
- `tests/`: automated tests
