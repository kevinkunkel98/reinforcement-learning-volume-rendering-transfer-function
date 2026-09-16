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

### RL v2 volumes

RL v2 trains and evaluates on 30 CT scans from the TotalSegmentator small
subset (CC-BY-4.0, Zenodo record 10047263), split by subject into 20 train,
4 validation and 6 test volumes (stratified by body region; see
`data/totalseg_manifest.json`), plus the four Slicer CTs as an out-of-source
test set. Fetch and extract once:

```bash
curl -L -o data/Totalsegmentator_dataset_small_v201.zip \
  "https://zenodo.org/records/10047263/files/Totalsegmentator_dataset_small_v201.zip?download=1"
python -m tools.select_totalseg
```

The selected volumes appear as `ts_<subject>` in the dataset list. RL v2 code
loads every volume with `load_dataset(name, canonical=True)` (RAS axis order)
and gets split members from `volumes_for_split("train" | "val" | "test" |
"out_of_source")`. `python -m tools.check_orientation <name>` writes projection
images for a visual orientation check.

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

### Visibility estimate

`visibility.py` estimates, per anatomical class, how much of a transfer
function's rendered image that class contributes (`vis`), how bright it
looks (`bright`), and how much of the frame is covered at all (`coverage`) --
without running VTK. It resamples a volume once per one of the 6 fixed views
into a front-to-back cube, then composites that cube for a given transfer
function; this is what the RL reward and observation are computed from,
since a real VTK render is too slow to call every training step.

Classes come from TotalSegmentator segmentation masks, collapsed into five
anatomical groups: `skeleton`, `lungs`, `organs`, `muscle`, `vessels` (label
ids 1-5; id 0, `other`, is everything TotalSegmentator's 117 structures don't
cover -- fat, skin, bowel contents, the scanner table -- and is the majority
of the body: only 7-20% of voxels in a typical scan carry any mask at all).
The four Slicer CTs and the synthetic phantom carry no TotalSegmentator
labels, so their samples fall back to coarse Hounsfield bands mapped onto the
same class names (skeleton >= 300 HU, lungs <= -500 HU, organs -30..300 HU);
muscle and vessels are never populated by the fallback, since they aren't
separable by intensity alone. `VisibilityModel.label_source` reports which
source (`"anatomy"` or `"intensity"`) produced a given model's estimate.
Vessels are anatomically labeled on every TotalSegmentator scan, but only
stand out as their own structure on a contrast (angiography) scan -- on a
plain scan they sit at the same HU as the soft tissue around them, so no
transfer function can visually single them out. "Show me the vessels" is
therefore only a meaningful RL goal on contrast scans.

`tools/validate_visibility.py` checks the estimate against real VTK renders.
For each class a volume's label volume carries, it sweeps 10 transfer
functions that isolate that class (the peak whose single-peak transfer
function shows the most of it, height 0.02 -> 1.0, the other peaks jittered
by a seeded generator), and compares the estimate's `vis * bright` against a
real reference: mean luminance over the 6 views, normal render minus a
render with that class's label colour blacked out and its opacity left
untouched, using VTK's label-map masking
(`vtkGPUVolumeRayCastMapper.SetMaskInput` + `SetMaskTypeToLabelMap`).
Blackening rather than deleting a class's voxels keeps occlusion the same --
deleting opens a hole that reveals whatever sits behind, which is not that
class's actual contribution to the image. Coverage is checked separately, by
rank agreement between the estimate's `coverage` and the rendered fraction of
lit pixels over a global opacity sweep (every peak's height swept together).
A class whose real contribution barely moves across the sampled transfer
functions (rendered luminance range below 0.002, the renderer's own noise
floor) can't be validated this way and is reported unvalidated rather than
failed.

Measured on three TotalSegmentator volumes (`out/visibility_validation.json`;
Pearson >= 0.7 required per validated class, rank agreement >= 0.9 for
coverage -- every validated class passed):

| volume | skeleton | organs | muscle | vessels | lungs | coverage (rank agreement) |
| --- | --- | --- | --- | --- | --- | --- |
| ts_s1379 (contrast) | 1.000 | 0.995 | 0.981 | 0.992 | 0.804, unvalidated (range 0.0001) | 1.000 |
| ts_s1337 | 0.999 | 0.994 | 0.948 | 0.989 | 0.349, unvalidated (range 0.0000) | 1.000 |
| ts_s0454 | 1.000 | 0.969 | 0.999 | 1.000, unvalidated (range 0.0009) | -- (no lungs label present) | 1.000 |

`lungs` and, on the non-contrast `ts_s0454`, `vessels` fell below the
0.002 noise floor and are unvalidated rather than failed: lungs sit near air
HU, so a transfer-function peak aimed at them barely changes mean luminance
against the scan's own dark background, and `ts_s0454`'s vessels are a small,
non-contrast structure that stays a thin sliver of the rendered image either
way.

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
