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
#let blue = rgb("#1a4f8a")
#let sky = rgb("#edf2f8")
#let sage = rgb("#1e6b3c")
#let amber = rgb("#b7770d")

#let note(body, title: "Note", color: amber) = block(
  width: 100%, inset: (x: 0.9em, y: 0.6em), radius: 3pt, fill: rgb("#fef9ec"),
  stroke: (left: 3pt + color),
)[#text(weight: "bold", fill: color, size: 0.9em)[#title:] #h(0.4em) #body]

#let pbox(title, sub) = block(
  fill: sky, inset: 8pt, radius: 4pt, width: 100%,
)[
  #align(center)[
    #text(weight: "bold", size: 8.5pt, fill: navy)[#title] \
    #v(2pt)
    #text(size: 7.5pt, fill: luma(90))[#sub]
  ]
]

// ══════════════════════════════════════════════════════════════════════════
// Title block
// ══════════════════════════════════════════════════════════════════════════
#align(center)[
  #text(size: 20pt, weight: "bold", fill: navy)[
    Voice-Driven Transfer Function \& Viewpoint Control
  ] \
  #v(0.3em)
  #text(size: 13pt, fill: luma(80))[Architecture Reference — Current State]
  #v(0.6em)
  #line(length: 40%, stroke: 1pt + navy)
  #v(0.6em)
  #text(size: 11pt)[Kevin Kunkel] \
  #text(size: 9.5pt, fill: luma(110))[
    Universität Leipzig · Masterseminar: Optimierung von Transferfunktionen \
    im Volume Rendering mit Reinforcement Learning \
    Betreuer: Prof. Gerik Scheuermann, Dr. Thomas Wischgoll, Dr. Baldwin Nsonga
  ] \
  #v(0.3em)
  #text(size: 9pt, fill: luma(130))[September 2026]
]

#v(1.5em)

This document describes the system as it stands today: a local prototype that
turns spoken or typed commands into transfer-function and camera edits on
volume-rendered CT data, plus two independent reinforcement-learning
sub-projects that each learn to replace a hand-tuned hill-climbing baseline.
It is a technical reference for the codebase, not a thesis chapter — for the
narrative/motivation framing see `slides/slides.typ`, and for a boxes-and-
arrows diagram of the same system see `architecture-mvp.drawio` in the repo
root.

#outline(title: "Contents", indent: 1.5em)

#pagebreak()

= Overview

The system has five layers, each independently swappable without touching
its neighbors:

#v(0.3em)
#grid(
  columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr, auto, 1fr),
  align: horizon + center,
  column-gutter: 4pt,
  pbox([Voice / Text], [ASR (Whisper) + chat UI]),
  text(fill: luma(140))[→],
  pbox([Parser], [rule-based or local LLM]),
  text(fill: luma(140))[→],
  pbox([Command dict], [target · attribute · direction · strength, or camera]),
  text(fill: luma(140))[→],
  pbox([TF params + camera], [24-float vector + az/el/zoom]),
  text(fill: luma(140))[→],
  pbox([VTK render], [GPU raycast → PNG]),
)
#v(0.3em)
#align(center)[
  #text(size: 8pt, fill: luma(120), style: "italic")[
    A rendered step feeds an evaluator (objective metric or human judgment),
    which drives a hill-climbing search loop back into the command layer —
    see @sec-search.
  ]
]

#v(0.5em)

Two Gymnasium environments mirror pieces of this pipeline for offline RL
training, without touching VTK rendering at all: one trains against the
transfer-function opacity metric directly (@sec-rl-tf), the other against a
camera-viewpoint alignment proxy computed straight from the volume's voxel
data (@sec-rl-camera). Both are trained, evaluated, and compared against the
hand-coded hill-climbing baseline they are meant to eventually replace —
neither is wired into the live chat/voice loop yet.

== Module map

#table(
  columns: (auto, 1fr),
  stroke: (bottom: 0.5pt + luma(200)),
  inset: (x: 6pt, y: 5pt),
  align: (left, left),
  table.header([*File*], [*Responsibility*]),
  [`phantom.py`], [Synthetic CT volume in Hounsfield units (fallback dataset)],
  [`datasets.py`], [Real CT dataset registry: checksum-verified download, NRRD loading, caching],
  [`transfer.py`], [24-float TF vector ↔ VTK transfer functions, `opacity_mass` / `mass_fraction` metrics],
  [`render.py`], [Offscreen VTK volume render, pixel grab, image features],
  [`camera.py`], [Camera state (azimuth/elevation/zoom) and relative camera commands],
  [`commands.py`], [Rule-based parser, local-LLM parser, `apply_command`, `COMMAND_REFERENCE`],
  [`search.py`], [Hill-climbing step math (`propose_step`, `resize_step`), shared across every consumer],
  [`evaluate.py`], [Objective (`opacity_mass`-based) and human (console) evaluators, preference logging],
  [`asr.py`], [`faster-whisper` push-to-talk mic capture and file transcription],
  [`mvp.py`], [CLI entry point: single apply, hill-climb loop, state persistence],
  [`server.py` + `static/`], [FastAPI chat UI: text/voice commands, history navigation, human judging],
  [`rl/env.py`, `rl/train.py`, `rl/eval.py`], [RL sub-project 1: transfer-function opacity (SAC)],
  [`rl/camera_env.py`, `rl/camera_train.py`, `rl/camera_eval.py`], [RL sub-project 2: camera viewpoint (SAC)],
  [`plots/`], [Thesis figure generation from SB3 training logs],
)

#pagebreak()

= Rendering pipeline

== Data sources

Every command ultimately acts on a 3D numpy array of Hounsfield Units (HU)
plus a physical voxel spacing. Two sources exist behind one interface,
`datasets.load_dataset(name) -> (volume, spacing)`:

- *`synthetic`* — `phantom.build_phantom()`: a fixed-seed, layered blob
  phantom (torso → fat → spongy → cortical bone, each a smooth sigmoid-edge
  blob composited additively, plus Gaussian noise). Fast, deterministic, no
  network access, but its outer tissues (air, fat, soft) are geometrically
  symmetric around the volume center — a limitation that mattered for
  camera-viewpoint RL (@sec-rl-camera).
- *Four real CT datasets* (`ct_chest`, `ct_skull`, `ct_cardio`, `ct_abdomen`)
  — de-identified public test data from the 3D Slicer project, each
  downloaded once from a content-addressed GitHub release URL, SHA-256
  checksum-verified against Slicer's own registration, and cached under
  `data/`. Loaded via `vtkNrrdReader`, with an explicit axis transpose so the
  resulting numpy array's axis 0 matches VTK's X axis exactly as `render.py`
  expects (`ravel(order="F")`).

== Transfer function representation

The transfer function is a flat 24-float vector: 4 Gaussian peaks × 6
parameters each (center, width, height, R, G, B), always stored normalized
to $[-1, 1]$ and mapped to real units on demand:

```python
N_PEAKS = 4
PARAMS_PER_PEAK = 6  # center, width, height, r, g, b
CENTER_RANGE = (-1050.0, 2000.0)   # HU
WIDTH_RANGE  = (10.0, 400.0)       # HU
```

Five tissues are recognized throughout the system by fixed HU bands, used
identically by the transfer function, the command layer, and both RL
sub-projects:

#table(
  columns: (auto, auto, auto),
  stroke: (bottom: 0.5pt + luma(200)),
  inset: (x: 6pt, y: 4pt),
  table.header([*Tissue*], [*Representative HU*], [*Band (HU)*]),
  [air], [−1000], [−1050 to −550],
  [fat], [−100], [−550 to −30],
  [soft], [40], [−30 to 170],
  [spongy (cancellous bone)], [300], [170 to 600],
  [bone (cortical)], [900], [600 to 2000],
)

At render time, `transfer.vector_to_vtk` samples 256 HU points, evaluates
each peak's Gaussian, composites total opacity (clipped to $[0,1]$) and a
weight-blended RGB, and builds a `vtkColorTransferFunction` +
`vtkPiecewiseFunction` pair. Two derived metrics matter everywhere else in
the codebase:

- *`opacity_mass(params, lo, hi)`* — the exact trapezoidal integral of
  composited opacity over an HU sub-range. This is the ground-truth signal
  the hand-coded evaluator and both RL rewards are built on; no rendering is
  needed to compute it.
- *`mass_fraction(params, tissue)`* — `opacity_mass` normalized by the
  band's width, giving an average-opacity-in-band fraction in $[0,1]$. Raw
  `opacity_mass` scales with band width (bone's band alone is 1400 HU wide),
  which would bias any learner toward wide bands; `mass_fraction` removes
  that bias and is what the TF-opacity RL environment observes and rewards.

== VTK render

`render.render(volume, params, spacing, camera)` builds a `vtkImageData`
from the HU array, attaches the transfer function via a `vtkVolumeProperty`,
and picks a GPU raycast mapper with an automatic CPU fallback
(`vtkFixedPointVolumeRayCastMapper`) when the GPU path isn't supported. The
camera is applied in a fixed order that matters: `ResetCamera()` first (so
the volume fills the frame), then relative `Azimuth()` / `Elevation()` /
`Zoom()` calls, then `ResetCameraClippingRange()` — reusing the same
relative-adjustment shape the camera command layer already uses
(@sec-camera). `render.grab()` reads back an RGB numpy array from the
offscreen window; `render.features()` computes cheap image statistics
(mean, std, coverage, entropy) that get logged alongside every command for
later analysis, independent of any evaluator.

#pagebreak()

= Command layer

Every user-facing instruction becomes one command dict before anything else
happens. There are five shapes:

```python
{"target": "bone", "attribute": "opacity", "direction": "increase", "strength": "strongly"}
{"target": "bone", "attribute": "opacity", "direction": "set", "level": "high"}       # absolute
{"compound": [ {...}, {...} ]}                                                        # multi-tissue absolute
{"target": ["bone", "spongy"], "attribute": "opacity", "direction": "show_only", "strength": None}
{"camera": {"action": "rotate", "direction": "right", "strength": "moderately"}}       # no TF target at all
```

`COMMAND_REFERENCE` (in `commands.py`) is the single source of truth for
what the system understands — it drives the rule parser's regex grammar,
the LLM's system prompt, *and* the in-UI help dialog, so the three can never
silently drift apart:

#table(
  columns: (auto, 1fr),
  stroke: (bottom: 0.5pt + luma(200)),
  inset: (x: 6pt, y: 4pt),
  table.header([*Category*], [*Example*]),
  [Opacity (relative)], [`increase opacity for bone strongly`],
  [Opacity (absolute)], [`high opacity spongy`],
  [Show only], [`show only bone and spongy`],
  [Compound], [`high opacity spongy, low opacity bones`],
  [Width / sharpness], [`sharpen the bone peak`],
  [Brightness], [`brighten bone`],
  [Center position], [`shift bone's center up`],
  [Camera], [`rotate left`, `tilt down`, `zoom in`],
  [Reset], [`reset`],
)

== Rule parser

`parse_command_rule` runs an ordered cascade of regexes (reset → show-only →
sharpen/soften/brighten/darken → shift-center → camera rotate/tilt/zoom →
absolute level(s) → relative increase/decrease), raising `ValueError` if
nothing matches. Tissue names are resolved through a synonym table
(`TISSUE_SYNONYMS`) sorted longest-phrase-first, so `"cortical bone"`
matches before the bare `"bone"` it contains.

== LLM parser

`parse_command_llm` sends the raw text plus a strict-JSON system prompt to a
local Ollama server (default model `qwen2.5:7b`), enforcing `format: "json"`.
The prompt encodes the tissue synonym table (generated from the same
`TISSUE_SYNONYMS` the rule parser uses), the full command schema including
the separate camera-command shape, and a rule for direction that follows the
user's *goal* rather than their wording (`"barely there"` → increase,
because something invisible needs more presence). Any failure — network
error, malformed JSON, or a schema the validator rejects — falls back to the
rule parser and logs the full request/response/fallback-reason to
`out/llm_requests.jsonl` for later accuracy analysis (`eval_parsers.py`).

== Applying a command

`apply_command` resolves a target tissue to a concrete peak via
`_find_or_create_peak` (nearest existing peak by HU distance, or reseed the
*weakest* peak if nothing is within 300 HU), then mutates the relevant
parameter slot(s):

```python
ATTRIBUTE_PARAM_INDEX = {"opacity": 2, "width": 1, "brightness": (3, 4, 5), "center": 0}
```

Two details prevent the naive version of this from misbehaving:

- *`_asymptotic_step`* moves a fraction of the *remaining headroom* toward
  the $±1$ bound, not a fixed absolute amount — a fixed step would saturate
  almost any starting value in one command. This generalizes a fix originally
  found for height/opacity to every attribute stored in the same normalized
  encoding.
- *`_center_band_clip`* keeps a center-shift command inside the target
  tissue's own HU band, so repeated `"shift bone's center up"` commands
  can't silently walk the bone peak into cortical-adjacent territory outside
  its own tissue's meaning.

#pagebreak()

= Search and evaluation <sec-search>

Hill-climbing is the system's baseline optimizer, used directly by the CLI
and chat UI, and later used as the comparison point for both RL
sub-projects.

== Step math (`search.py`)

```python
def propose_step(params, peak_idx, sign, step):
    ...  # mutates only the resolved peak's height slot

def resize_step(step, accepted, max_step=1.0):
    return min(step * 1.2, max_step) if accepted else step * 0.5
```

`propose_step` is opacity-only by convention — callers must gate entry on
`attribute == "opacity"`, since it always mutates the height slot regardless
of what a command dict says. `resize_step` is domain-agnostic: it grows the
step 1.2× on acceptance (capped at `max_step`) and halves it on rejection.
The `max_step` parameter was added during the camera-viewpoint RL
sub-project after a real bug: the original hardcoded `1.0` cap was tuned for
the opacity domain (values live in $[0,1]$) and silently capped the camera
hill-climb's step at 1° instead of a 30°-scale exploration budget — see
@sec-rl-camera for the numbers this cost.

== Evaluators (`evaluate.py`)

`objective(params_before, params_after, cmd)` returns $+1$/$-1$:

- Camera commands, compound/reset commands, and width/brightness/center
  commands are always accepted — there is no established exact metric for
  them the way `opacity_mass` exists for opacity, and (for camera/compound)
  no well-defined "wrong direction" either.
- `show_only` is judged by *dominance share* (target tissue's fraction of
  total opacity mass across all bands), not raw mass — crushing every other
  peak can lower the target's own mass while still making it far more
  dominant in the image.
- A relative increase/decrease is accepted only if the target band's
  `opacity_mass` moved the right way *and* no other tissue's mass moved by
  more than the target did (a simple collateral-damage check).

`human(before_png, after_png)` is a console b/s prompt; the chat UI exposes
the same judgment as Better/Worse buttons over a live-rendered image pair
(@sec-server).

#pagebreak()

= Camera control <sec-camera>

The camera is modeled as its own small piece of state, deliberately kept
independent of the transfer-function vector:

```python
DEFAULT_CAMERA = {"azimuth": 30.0, "elevation": 20.0, "zoom": 1.0}
ELEVATION_RANGE = (-85.0, 85.0)
ZOOM_RANGE = (0.3, 4.0)
ROTATE_STEP_DEGREES = {"slightly": 15.0, "moderately": 30.0, "strongly": 60.0}
ZOOM_STEP_FACTOR    = {"slightly": 1.15, "moderately": 1.35, "strongly": 1.7}
```

`apply_camera_command` applies one relative adjustment: azimuth *wraps*
modulo 360° (it has no natural bound), while elevation and zoom *clamp* to
fixed ranges (both have real physical limits — clamping elevation
specifically avoids the camera flipping through the poles). Camera commands
are a fully separate shape from transfer-function commands at every layer:
their own regex branch in the rule parser, their own schema block in the LLM
system prompt, their own `_validate_camera_cmd` check, and (critically) they
never touch the 24-float TF vector at all.

== Why camera state lives on the backend

In the chat UI, camera state is stored *per history step*
(`Session.history[i]["camera"]`), not as client-side view state. A
camera-only command creates a new history step that carries the same
transfer-function params forward but a new camera; back/forward navigation
restores both together. This was a deliberate design choice, not an
accident: it makes camera state part of the same server-owned session model
as everything else, which is the actual prerequisite for a *different*
client — a browser tab today, potentially a native Vrui/CAVE application
later — to drive or observe the same session without duplicating logic
(see @sec-limitations).

One real bug surfaced here during review: `Session.judge()`'s convergence
branch originally sourced the *final* camera from live cursor state instead
of the camera frozen at the moment a human judgment began. If a user
rotated the camera while a judgment was pending, the eventual accepted step
could silently end up with the wrong camera. The fix freezes the camera
into the pending-judgment dict itself (`p["camera"]`) and reads it back from
there, so a mid-judgment camera change updates the *live* cursor (as
expected) without corrupting the judged step.

#pagebreak()

= RL sub-project 1 — transfer-function opacity <sec-rl-tf>

The first RL sub-project asks: can a learned policy match or beat the
hand-coded hill-climber at the task it already does — nudging one tissue's
opacity up or down? Training happens entirely offline against the exact
`opacity_mass` metric; no rendering, no human data.

== Environment (`rl/env.py`)

#table(
  columns: (auto, 1fr),
  stroke: (bottom: 0.5pt + luma(200)),
  inset: (x: 6pt, y: 4pt),
  table.header([*Aspect*], [*Definition*]),
  [Episode], [One `(target_tissue, direction)` goal, sampled fresh on `reset()`, from a randomly height-perturbed default transfer function],
  [Observation], [31-dim: the 24 TF params + 5-way tissue one-hot + direction flag + current `mass_fraction`],
  [Action], [1-dim continuous delta in $[-1,1]$, scaled by `MAX_DELTA = 0.2`, applied to the resolved target peak's height via the same `search.propose_step` the hill-climber uses],
  [Reward], [Signed per-step $Delta$`mass_fraction` in the target band (positive if it moved the right way)],
  [Episode length], [20 steps, truncated (not terminated)],
)

Reusing `commands._find_or_create_peak` and `search.propose_step` means the
RL agent moves the transfer function through *exactly* the same mechanism
as a real user command — it cannot exploit some RL-only shortcut unavailable
to the baseline.

== Training and evaluation

`rl/train.py` trains a `stable-baselines3` SAC agent (`MlpPolicy`, 4 parallel
environments via `make_vec_env`) for a configurable timestep budget,
logging per-seed CSV + TensorBoard output. `rl/eval.py` then replays the
*same* held-out `(params, target_tissue, direction)` episodes through both
the trained policy (deterministic) and the hill-climbing baseline
(`propose_step` + `resize_step`, gated by `objective()`), reporting mean
final `mass_fraction` and mean steps-to-90%-of-best
(`_steps_to_90pct`) — a generic helper reused unmodified by the second RL
sub-project.

== Result (200k timesteps, 20 held-out episodes)

#table(
  columns: (auto, auto, auto),
  stroke: (bottom: 0.5pt + luma(200)),
  inset: (x: 6pt, y: 5pt),
  table.header([], [*Mean final* `mass_fraction`], [*Mean steps to 90% of best*]),
  [Trained SAC policy], [0.344], [3.3],
  [Hill-climb baseline], [0.347], [2.7],
)

#note(title: "Interpretation")[
  Near-parity, with the hand-tuned hill-climber converging slightly faster.
  This is a legitimate finding, not a failed experiment: a well-tuned
  hand-designed controller is a strong baseline on a low-dimensional,
  single-scalar-action problem like this one. The value of the RL approach
  is expected to grow once the action space widens (full multi-peak control
  rather than one resolved peak's height) or the reward incorporates signal
  a hand rule can't easily express (e.g. learned human preference). The
  agent is trained and evaluated, but *not* wired into the live chat/voice
  loop.
]

#pagebreak()

= RL sub-project 2 — camera viewpoint <sec-rl-camera>

The second RL sub-project takes inspiration from natural-language
viewpoint-navigation work (`literature/ZhaoTao2025_NL-Viewpoint-Navigation-RL.pdf`)
but scopes down sharply: instead of language-conditioned navigation against
a learned reward, it trains a continuous-control agent to orient the camera
toward a target tissue's centroid direction, against a cheap, exact
geometric proxy computed directly from voxel data — no rendering, no
learned reward model.

== Why a real CT dataset, not the synthetic phantom

The phantom's outer tissues (air, fat, soft) were assumed, from reading
`phantom.py`'s blob-center coordinates, to be spatially symmetric around the
volume center. Direct measurement during design contradicted this: `soft`
tissue's HU-band mask inherits an off-center bias from the bone/spongy
blobs' smooth sigmoid-edge transitions overlapping into its own HU range, so
`soft` actually clusters with `spongy`/`bone` (approx. 5% relative centroid
offset) rather than with `air`/`fat` (approx. 1%). Since a genuinely-centered tissue has no
meaningful "direction" to align a camera toward, the phantom does not
reliably exercise this environment across all five tissues — the design was
changed to train on a real CT scan (`ct_skull`) instead, where anatomical
asymmetry is real rather than a modeling artifact.

== Environment (`rl/camera_env.py`)

#table(
  columns: (auto, 1fr),
  stroke: (bottom: 0.5pt + luma(200)),
  inset: (x: 6pt, y: 4pt),
  table.header([*Aspect*], [*Definition*]),
  [Episode], [One target tissue, resolved on `reset()` from a random starting azimuth/elevation],
  [Observation], [9-dim: $sin$/$cos$ of azimuth, normalized elevation, 5-way tissue one-hot, current alignment],
  [Action], [2-dim continuous $(Delta"azimuth", Delta"elevation")$ in $[-1,1]$, scaled by `MAX_DELTA_DEGREES = 30`°],
  [Reward], [Per-step increase in *alignment* — the dot product of the camera's view direction and the target tissue's centroid direction from the volume center],
  [Episode length], [20 steps, truncated (not terminated)],
)

A target tissue's centroid direction is computed once per episode:

```python
def _tissue_centroid_direction(volume, spacing, tissue):
    mask = (volume >= lo) & (volume < hi)          # tissue's HU band
    centroid = voxel_mean(mask, spacing)            # physical-space centroid
    offset = centroid - volume_center
    if norm(offset) < MIN_RELATIVE_CENTROID_OFFSET * volume_diagonal:
        return None   # too close to center to have a meaningful direction
    return offset / norm(offset)
```

`_resolve_target` shuffles the five tissues and picks the first with a
meaningful direction, falling back to `bone` — and raises a clear
`ValueError` (rather than silently propagating `None` into a downstream
`TypeError`) if even that fallback has no valid direction. This was itself a
bug found and fixed during implementation: the original fallback could
return `("bone", None)`.

== Training and evaluation

`rl/camera_train.py` mirrors `rl/train.py`'s structure exactly (same SAC/SB3
recipe, same per-seed logging convention, disjoint model/log paths so the
two agents never collide on disk), parameterized by which real dataset to
load. `rl/camera_eval.py` compares the trained policy against a 2D
coordinate hill-climbing baseline that probes four cardinal
azimuth/elevation directions per step, reusing `search.resize_step` and
`rl.eval._steps_to_90pct` directly rather than reimplementing either.

#note(title: "A real bug caught during this sub-project", color: rgb("#8a2e2e"))[
  The hill-climb baseline seeds its step at `MAX_DELTA_DEGREES` (30°), but
  `search.resize_step`'s growth cap was hardcoded to `1.0` — correct for
  every *other* caller, all of which operate on an opacity parameter in
  $[0,1]$. The very first accepted move collapsed the camera baseline's step
  from 30° to 1°, and it could never grow back. A smoke-test run with the
  bug present reached only *0.658* mean final alignment; after adding an
  optional `max_step` parameter (defaulting to `1.0`, so the three
  pre-existing opacity-scale callers are completely unaffected) and passing
  `max_step=MAX_DELTA_DEGREES` at the camera call site, the same setup
  reached *0.99998*. A regression test now locks in the correct wiring at
  that call site specifically, since none of the pre-existing tests would
  have caught a future regression of it.
]

== Result (200k timesteps, `ct_skull`, 20 held-out episodes)

#table(
  columns: (auto, auto, auto),
  stroke: (bottom: 0.5pt + luma(200)),
  inset: (x: 6pt, y: 5pt),
  table.header([], [*Mean final alignment*], [*Mean steps to 90% of best*]),
  [Trained SAC policy], [0.997], [3.05],
  [Hill-climb baseline], [1.000], [4.05],
)

#note(title: "Interpretation")[
  Both methods reach near-perfect alignment — expected, since the alignment
  landscape here is smooth and unimodal (a dot product against a fixed
  target direction), which a well-calibrated hill climb can solve almost
  exactly. The trained policy's advantage is convergence *speed*: it reaches
  90% of its achievable alignment about one step sooner on average, which is
  the more practically relevant property for an interactive tool where every
  step costs a render.
]

#note(title: "Known limitation", color: rgb("#8a2e2e"))[
  The centroid-alignment reward is an intentional, acknowledged proxy for
  "can you actually see this tissue" — it has *no occlusion model*. A camera
  can achieve high alignment while another structure blocks the actual view.
  Ray-marching-based visibility scoring was considered and explicitly
  deferred as a separate, much larger effort. The action space is also
  angle-only (azimuth/elevation) — zoom is not controlled — by deliberate
  scoping decision.
]

#pagebreak()

= Server, session model, and UI <sec-server>

`server.py` exposes a FastAPI app whose route handlers are thin `async def`
wrappers around a plain-Python `Session` class with no FastAPI or threading
dependency — kept independently unit-testable, and kept `async` specifically
so Starlette runs handlers on the main event-loop thread rather than a
worker pool (VTK's Cocoa render window can only be created on the true
process main thread on macOS).

== Session state

```python
class Session:
    history: list[dict]   # one entry per applied step
    cursor: int            # current position for back/forward navigation
    pending: dict | None   # an in-progress human-judgment pair, if any
```

Each history entry carries its transfer-function params, its camera dict,
computed tissue masses, image features, the command that produced it, and
any thumbs-up/down feedback — camera and TF state travel together but are
never conflated (@sec-camera). A new command truncates any existing forward
history from the current cursor, matching ordinary browser back/forward
semantics. A `pending` human-judgment entry freezes a before/after image
pair and the camera in effect at the time, so the judgment being evaluated
can never be silently corrupted by state that changes while the human is
still deciding.

== Persistence

#table(
  columns: (auto, 1fr),
  stroke: (bottom: 0.5pt + luma(200)),
  inset: (x: 6pt, y: 4pt),
  table.header([*Path*], [*Contents*]),
  [`out/ui_session.json`], [Full session (history + cursor), reloaded across server restarts],
  [`out/ui_images/<session_id>/`], [Every rendered PNG, namespaced per session so ids never collide across dataset switches],
  [`out/log.jsonl`], [Every applied step and hill-climb iteration, with before/after params and verdict],
  [`out/preferences.jsonl`], [Human judgments with before/after image paths and the objective evaluator's agreeing/disagreeing verdict],
  [`out/feedback.jsonl`], [Thumbs up/down ratings on any rendered step],
  [`out/llm_requests.jsonl`], [Every LLM-parser call: raw response, parsed command, whether/why it fell back],
)

== UI

The static frontend (`static/index.html`, `app.js`, `style.css`) is a
chat-style interface: type or speak a command (🎤, via the browser mic and
the same `faster-whisper` wrapper used by the CLI), watch it render, step
back/forward through history, judge pending hill-climb pairs as
Better/Worse, leave thumbs-up/down feedback on any step, switch datasets,
and open a command-reference help dialog generated from the exact same
`COMMAND_REFERENCE` table the parsers use internally.

#pagebreak()

= Known limitations and next steps <sec-limitations>

- *Neither RL agent is live.* Both sub-projects are trained and evaluated
  entirely offline against the same metrics the hand-coded hill-climber
  already uses; neither is currently reachable from the chat/voice loop.
- *Camera-viewpoint reward has no occlusion model* (@sec-rl-camera) — it is
  a directional-alignment proxy, not a rendering-accurate visibility claim.
- *Camera-viewpoint RL controls angle only*, not zoom — an approved scoping
  decision.
- *Everything runs locally.* Ollama, `faster-whisper`, and VTK's offscreen
  renderer all run on-machine; no data or audio leaves the process.

== Roadmap toward voice-driven camera control in browser and VR

The longer-term goal is a camera the user can drive by voice, with the same
backend logic reachable from both a browser client and, eventually, a
native client for a lab VR rig (Vrui/CAVE-based). That was decomposed into
four sub-projects:

+ *Camera as controllable state* — done. Camera state lives in the backend
  session, addressed relatively (rotate/tilt/zoom deltas) exactly like
  transfer-function commands are.
+ *RL for camera-viewpoint optimization* — done (@sec-rl-camera).
+ *Client/backend split* — satisfied by construction: because camera state
  already lives server-side rather than in any one client's view state
  (@sec-camera), a second client does not require re-deriving this logic.
+ *A native Vrui/CAVE client* reusing the same backend `Session`/command
  logic — deliberately deferred until an upcoming lab visit clarifies the
  actual VR hardware available, since Vrui targets specific CAVE/lab rig
  configurations rather than being a general browser-compatible toolkit the
  way WebXR is.
