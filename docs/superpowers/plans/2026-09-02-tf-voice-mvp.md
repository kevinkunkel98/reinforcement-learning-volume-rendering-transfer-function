# Voice-Driven Transfer Function MVP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, offscreen VTK prototype where natural-language or voice commands
(parsed by a rule-based parser or a local Ollama LLM) nudge a 4-Gaussian-peak transfer
function over a synthetic CT phantom, searched by hill-climbing against an exact
Hounsfield-range opacity metric or a human judge, with every judgment logged as a
preference pair for later offline analysis.

**Architecture:** Nine small modules, no framework, no torch. `phantom.py` builds the
volume once; `transfer.py` is pure numpy math (vector ↔ VTK objects, opacity integral);
`render.py` wraps VTK offscreen rendering + pixel features; `commands.py` holds two
interchangeable text→command-dict parsers (regex rule parser, Ollama-backed LLM parser
with automatic fallback); `evaluate.py` holds the two verdict functions plus preference
logging; `asr.py` wraps faster-whisper; `mvp.py` is the CLI/orchestrator with the
hill-climbing loop and a tiny on-disk state file so commands compose across invocations;
`stats.py` and `eval_parsers.py` are standalone analysis scripts.

**Tech Stack:** Python 3, `numpy`, `vtk`, `faster-whisper`, `sounddevice`, stdlib
`urllib`/`wave`/`json` (no `requests`, no `torch`). Local Ollama server for the LLM
parser (`qwen2.5:7b` default), no other network calls. `pytest` for the testable pure
modules.

---

## File Structure

```
phantom.py               # synthetic CT volume in HU
transfer.py               # 24-float vector <-> VTK transfer functions, opacity_mass
render.py                  # offscreen VTK render, pixel grab, image features
commands.py                # rule parser, Ollama LLM parser, apply_command
evaluate.py                 # objective(), human(), preference-pair logging
asr.py                      # faster-whisper wrapper, mic + file transcription
mvp.py                       # CLI, hill-climbing loop, log.jsonl, state persistence
stats.py                      # preferences.jsonl analysis
eval_parsers.py                # rule vs LLM parser accuracy comparison
data/parser_eval_phrases.json   # ~20 hand-labeled free-form phrases
tests/test_phantom.py
tests/test_transfer.py
tests/test_commands.py
tests/test_evaluate.py
tests/test_render.py
README.md
requirements.txt
.gitignore                        # out/, __pycache__/
```

Design reference: `docs/superpowers/specs/2026-09-02-tf-voice-mvp-design.md`.

---

### Task 0: Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `out/.gitkeep`, `out/audio/.gitkeep`
- Create: `data/.gitkeep` (removed once `parser_eval_phrases.json` lands in Task 9)

- [ ] **Step 1: Create `requirements.txt`**

```
numpy
vtk
faster-whisper
sounddevice
pytest
```

- [ ] **Step 2: Create `.gitignore`**

```
out/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 3: Create placeholder dirs**

```bash
mkdir -p out/audio data tests
touch out/.gitkeep out/audio/.gitkeep
```

- [ ] **Step 4: Install deps**

Run: `pip install -r requirements.txt`
Expected: succeeds (VTK wheels are large; allow a few minutes).

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore out/.gitkeep out/audio/.gitkeep
git commit -m "chore: scaffold project layout and dependencies"
```

---

### Task 1: `phantom.py`

**Files:**
- Create: `phantom.py`
- Test: `tests/test_phantom.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_phantom.py
import numpy as np
from phantom import build_phantom, HU

def test_shape_and_dtype():
    vol = build_phantom(size=32)
    assert vol.shape == (32, 32, 32)
    assert vol.dtype == np.float32

def test_deterministic_same_seed():
    a = build_phantom(size=32, seed=42)
    b = build_phantom(size=32, seed=42)
    assert np.array_equal(a, b)

def test_different_seed_differs():
    a = build_phantom(size=32, seed=42)
    b = build_phantom(size=32, seed=7)
    assert not np.array_equal(a, b)

def test_value_range_covers_tissues():
    vol = build_phantom(size=48, seed=42)
    assert vol.min() < HU["fat"]          # background/air present
    assert vol.max() > HU["spongy"]        # dense core present
    assert vol.min() >= -1050.0
    assert vol.max() <= 2000.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phantom.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'phantom'`

- [ ] **Step 3: Implement `phantom.py`**

```python
"""Synthetic CT phantom in Hounsfield units."""
import numpy as np

SIZE_DEFAULT = 96
SEED = 42

HU = {
    "air": -1000.0,
    "fat": -100.0,
    "soft": 40.0,
    "spongy": 300.0,
    "bone": 900.0,
}


def _soft_blob(xx, yy, zz, cx, cy, cz, radius, value):
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
    edge = 1.0 / (1.0 + np.exp((d - radius) / (radius * 0.08 + 1e-6)))
    return edge * value


def build_phantom(size: int = SIZE_DEFAULT, seed: int = SEED) -> np.ndarray:
    """Layered, mutually-occluding tissue blobs in HU, fixed noise."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size].astype(np.float32)
    center = size / 2.0

    vol = np.full((size, size, size), HU["air"], dtype=np.float32)

    torso = _soft_blob(xx, yy, zz, center, center, center,
                        size * 0.34, HU["soft"] - HU["air"])
    vol += torso

    fat_mask = torso > (HU["soft"] - HU["air"]) * 0.3
    fat = _soft_blob(xx, yy, zz, center, center, center,
                      size * 0.30, HU["fat"] - HU["soft"]) * fat_mask
    vol += fat

    bx, by, bz = center * 1.15, center * 0.9, center
    spongy = _soft_blob(xx, yy, zz, bx, by, bz,
                         size * 0.16, HU["spongy"] - HU["soft"])
    vol += spongy

    bone = _soft_blob(xx, yy, zz, bx, by, bz,
                       size * 0.10, HU["bone"] - HU["spongy"])
    vol += bone

    noise = rng.normal(0.0, 15.0, vol.shape).astype(np.float32)
    vol += noise
    return np.clip(vol, -1050.0, 2000.0).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_phantom.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add phantom.py tests/test_phantom.py
git commit -m "feat: synthetic HU phantom with mutually-occluding tissue blobs"
```

---

### Task 2: `transfer.py`

**Files:**
- Create: `transfer.py`
- Test: `tests/test_transfer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfer.py
import numpy as np
import math
from transfer import (
    default_params, opacity_mass, vector_to_vtk,
    TISSUE_BANDS, TISSUE_HU, PARAMS_PER_PEAK, N_PEAKS,
)

def test_default_params_shape():
    p = default_params()
    assert p.shape == (N_PEAKS * PARAMS_PER_PEAK,)
    assert np.all(p >= -1.0) and np.all(p <= 1.0)

def test_opacity_mass_single_gaussian_matches_analytic():
    # peak 0 active (center=40 HU / soft, width=40, height=0.2), rest zero-height
    params = np.full(N_PEAKS * PARAMS_PER_PEAK, -1.0)
    # center external for 40 HU:
    lo, hi = -1050.0, 2000.0
    center_ext = 2.0 * (40.0 - lo) / (hi - lo) - 1.0
    width_ext = 2.0 * (40.0 - 10.0) / (400.0 - 10.0) - 1.0
    height_ext = 2.0 * 0.2 - 1.0
    params[0:6] = [center_ext, width_ext, height_ext, -1.0, -1.0, -1.0]
    mass = opacity_mass(params, -1050.0, 2000.0)
    analytic = 0.2 * 40.0 * math.sqrt(2 * math.pi)
    assert abs(mass - analytic) / analytic < 0.05

def test_opacity_mass_zero_for_all_zero_height():
    params = np.full(N_PEAKS * PARAMS_PER_PEAK, -1.0)  # height=-1 -> internal 0
    mass = opacity_mass(params, *TISSUE_BANDS["bone"])
    assert mass < 1e-6

def test_vector_to_vtk_returns_expected_types():
    import vtk
    ctf, otf = vector_to_vtk(default_params())
    assert isinstance(ctf, vtk.vtkColorTransferFunction)
    assert isinstance(otf, vtk.vtkPiecewiseFunction)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transfer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'transfer'`

- [ ] **Step 3: Implement `transfer.py`**

```python
"""24-float transfer-function vector <-> VTK objects, opacity_mass metric."""
import numpy as np
import vtk

N_PEAKS = 4
PARAMS_PER_PEAK = 6  # center, width, height, r, g, b
TOTAL_PARAMS = N_PEAKS * PARAMS_PER_PEAK

CENTER_RANGE = (-1050.0, 2000.0)
WIDTH_RANGE = (10.0, 400.0)

TISSUE_HU = {"air": -1000.0, "fat": -100.0, "soft": 40.0,
             "spongy": 300.0, "bone": 900.0}

TISSUE_BANDS = {
    "air": (-1050.0, -550.0),
    "fat": (-550.0, -30.0),
    "soft": (-30.0, 170.0),
    "spongy": (170.0, 600.0),
    "bone": (600.0, 2000.0),
}


def _to_range(x, lo, hi):
    return lo + (x + 1.0) / 2.0 * (hi - lo)


def _from_range(v, lo, hi):
    return 2.0 * (v - lo) / (hi - lo) - 1.0


def _unit(x):
    return (x + 1.0) / 2.0


def _from_unit(u):
    return u * 2.0 - 1.0


def peak_internal(params: np.ndarray, i: int) -> dict:
    """One peak's params in real (internal) units."""
    c, w, h, r, g, b = params[i * PARAMS_PER_PEAK:(i + 1) * PARAMS_PER_PEAK]
    return {
        "center": _to_range(c, *CENTER_RANGE),
        "width": _to_range(w, *WIDTH_RANGE),
        "height": _unit(h),
        "rgb": (_unit(r), _unit(g), _unit(b)),
    }


def default_params() -> np.ndarray:
    """One peak seeded near fat/soft/spongy/bone, ascending heights."""
    specs = [
        ("fat", 60.0, 0.05, (0.95, 0.90, 0.60)),
        ("soft", 40.0, 0.15, (0.85, 0.35, 0.35)),
        ("spongy", 80.0, 0.30, (0.90, 0.80, 0.60)),
        ("bone", 150.0, 0.60, (0.95, 0.95, 0.90)),
    ]
    params = np.zeros(TOTAL_PARAMS, dtype=np.float64)
    for i, (tissue, width, height, rgb) in enumerate(specs):
        base = i * PARAMS_PER_PEAK
        params[base + 0] = _from_range(TISSUE_HU[tissue], *CENTER_RANGE)
        params[base + 1] = _from_range(width, *WIDTH_RANGE)
        params[base + 2] = _from_unit(height)
        params[base + 3] = _from_unit(rgb[0])
        params[base + 4] = _from_unit(rgb[1])
        params[base + 5] = _from_unit(rgb[2])
    return params


def _opacity_and_color_at(params: np.ndarray, hu: np.ndarray):
    """Combined opacity (clipped [0,1]) and blended RGB at each HU sample."""
    total_opacity = np.zeros_like(hu, dtype=np.float64)
    weighted_rgb = np.zeros((hu.shape[0], 3), dtype=np.float64)
    weight_sum = np.full_like(hu, 1e-6, dtype=np.float64)
    for i in range(N_PEAKS):
        p = peak_internal(params, i)
        gauss = np.exp(-0.5 * ((hu - p["center"]) / p["width"]) ** 2)
        contrib = p["height"] * gauss
        total_opacity += contrib
        weight_sum += contrib
        for c in range(3):
            weighted_rgb[:, c] += contrib * p["rgb"][c]
    total_opacity = np.clip(total_opacity, 0.0, 1.0)
    rgb = weighted_rgb / weight_sum[:, None]
    rgb = np.clip(rgb, 0.0, 1.0)
    return total_opacity, rgb


def opacity_mass(params: np.ndarray, hu_lo: float, hu_hi: float, n: int = 256) -> float:
    hu = np.linspace(hu_lo, hu_hi, n)
    opacity, _ = _opacity_and_color_at(params, hu)
    return float(np.trapz(opacity, hu))


def vector_to_vtk(params: np.ndarray, n: int = 256):
    hu = np.linspace(*CENTER_RANGE, n)
    opacity, rgb = _opacity_and_color_at(params, hu)

    ctf = vtk.vtkColorTransferFunction()
    otf = vtk.vtkPiecewiseFunction()
    for i in range(n):
        ctf.AddRGBPoint(float(hu[i]), *[float(v) for v in rgb[i]])
        otf.AddPoint(float(hu[i]), float(opacity[i]))
    return ctf, otf
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transfer.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add transfer.py tests/test_transfer.py
git commit -m "feat: transfer-function vector math, opacity_mass, VTK conversion"
```

---

### Task 3: `render.py`

**Files:**
- Create: `render.py`
- Test: `tests/test_render.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_render.py
import numpy as np
from phantom import build_phantom
from transfer import default_params
from render import render, grab, features

def test_render_grab_shape_and_dtype():
    vol = build_phantom(size=48)
    win = render(vol, default_params())
    img = grab(win)
    assert img.shape == (400, 512, 3)
    assert img.dtype == np.uint8

def test_render_deterministic():
    vol = build_phantom(size=48)
    win1 = render(vol, default_params())
    win2 = render(vol, default_params())
    img1, img2 = grab(win1), grab(win2)
    assert np.array_equal(img1, img2)

def test_features_keys_and_ranges():
    vol = build_phantom(size=48)
    img = grab(render(vol, default_params()))
    f = features(img)
    assert set(f.keys()) == {"mean", "std", "coverage", "entropy"}
    assert 0.0 <= f["coverage"] <= 1.0
    assert f["entropy"] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'render'`

- [ ] **Step 3: Implement `render.py`**

```python
"""Offscreen VTK volume rendering, pixel grab, image features."""
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from transfer import vector_to_vtk

WIDTH, HEIGHT = 512, 400
_MAPPER_ANNOUNCED = False


def _make_mapper(vtk_image):
    global _MAPPER_ANNOUNCED
    mapper = vtk.vtkGPUVolumeRayCastMapper()
    mapper.SetInputData(vtk_image)
    try:
        mapper.SetUseJittering(0)
        ok = bool(mapper.IsRenderSupported(vtk.vtkRenderWindow(), None))
    except Exception:
        ok = False
    if not ok:
        mapper = vtk.vtkFixedPointVolumeRayCastMapper()
        mapper.SetInputData(vtk_image)
        name = "vtkFixedPointVolumeRayCastMapper (CPU fallback)"
    else:
        name = "vtkGPUVolumeRayCastMapper"
    if not _MAPPER_ANNOUNCED:
        print(f"[render] using {name}")
        _MAPPER_ANNOUNCED = True
    return mapper


def render(volume: np.ndarray, params: np.ndarray) -> vtk.vtkRenderWindow:
    size = volume.shape[0]
    flat = np.ascontiguousarray(volume.ravel(order="F"))
    vtk_arr = numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)

    image = vtk.vtkImageData()
    image.SetDimensions(size, size, size)
    image.GetPointData().SetScalars(vtk_arr)

    ctf, otf = vector_to_vtk(params)
    prop = vtk.vtkVolumeProperty()
    prop.SetColor(ctf)
    prop.SetScalarOpacity(otf)
    prop.ShadeOn()
    prop.SetInterpolationTypeToLinear()

    mapper = _make_mapper(image)
    volume_actor = vtk.vtkVolume()
    volume_actor.SetMapper(mapper)
    volume_actor.SetProperty(prop)

    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume_actor)
    renderer.SetBackground(0.0, 0.0, 0.0)

    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.AddRenderer(renderer)
    win.SetSize(WIDTH, HEIGHT)

    renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    cam.Azimuth(30)
    cam.Elevation(20)
    renderer.ResetCameraClippingRange()

    win.Render()
    return win


def grab(win: vtk.vtkRenderWindow) -> np.ndarray:
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(win)
    w2i.SetInputBufferTypeToRGB()
    w2i.ReadFrontBufferOff()
    w2i.Update()
    vtk_image = w2i.GetOutput()
    w, h, _ = vtk_image.GetDimensions()
    from vtk.util.numpy_support import vtk_to_numpy
    arr = vtk_to_numpy(vtk_image.GetPointData().GetScalars())
    arr = arr.reshape(h, w, 3).astype(np.uint8)
    return arr[::-1]  # VTK draws bottom-up


def features(rgb: np.ndarray) -> dict:
    gray = rgb.astype(np.float64).mean(axis=2)
    coverage = float((gray > 2.0).mean())
    hist, _ = np.histogram(gray, bins=32, range=(0, 255), density=False)
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())
    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "coverage": coverage,
        "entropy": entropy,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_render.py -v`
Expected: PASS (3 tests). Note the printed mapper line — confirm it says which mapper macOS picked.

- [ ] **Step 5: Commit**

```bash
git add render.py tests/test_render.py
git commit -m "feat: offscreen VTK rendering with GPU->CPU mapper fallback"
```

---

### Task 4: `commands.py` — rule parser + `apply_command`

**Files:**
- Create: `commands.py`
- Test: `tests/test_commands.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_commands.py
import numpy as np
from commands import parse_command_rule, apply_command, TISSUE_SYNONYMS
from transfer import default_params, peak_internal, N_PEAKS

def test_parse_increase_with_strength():
    cmd = parse_command_rule("increase opacity for bone strongly")
    assert cmd == {"target": "bone", "attribute": "opacity",
                    "direction": "increase", "strength": "strongly"}

def test_parse_decrease_default_strength():
    cmd = parse_command_rule("decrease opacity for fat")
    assert cmd["target"] == "fat"
    assert cmd["direction"] == "decrease"
    assert cmd["strength"] == "moderately"

def test_parse_show_only():
    cmd = parse_command_rule("show only spongy bone")
    assert cmd["direction"] == "show_only"
    assert cmd["target"] == "spongy"

def test_parse_reset():
    cmd = parse_command_rule("reset")
    assert cmd == {"target": None, "attribute": None,
                    "direction": "reset", "strength": None}

def test_parse_unrecognized_raises():
    import pytest
    with pytest.raises(ValueError):
        parse_command_rule("make the skeleton pop")

def test_apply_command_increases_target_height():
    params = default_params()
    before_h = peak_internal(params, 3)["height"]  # bone is peak index 3
    cmd = {"target": "bone", "attribute": "opacity",
           "direction": "increase", "strength": "strongly"}
    after = apply_command(cmd, params)
    after_h = peak_internal(after, 3)["height"]
    assert after_h > before_h

def test_apply_command_reset_returns_default():
    params = default_params()
    params[2] = 1.0  # perturb
    cmd = {"target": None, "attribute": None, "direction": "reset", "strength": None}
    after = apply_command(cmd, params)
    np.testing.assert_array_equal(after, default_params())

def test_apply_command_show_only_suppresses_others():
    params = default_params()
    cmd = {"target": "bone", "attribute": "opacity",
           "direction": "show_only", "strength": None}
    after = apply_command(cmd, params)
    heights = [peak_internal(after, i)["height"] for i in range(N_PEAKS)]
    assert heights[3] > 0.5
    assert all(h < 0.05 for i, h in enumerate(heights) if i != 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_commands.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'commands'`

- [ ] **Step 3: Implement `commands.py` (rule parser + apply_command only — LLM parser added in Task 5)**

```python
"""Text -> command dict (two implementations) and command -> params."""
import re
import numpy as np
from transfer import (
    TISSUE_HU, PARAMS_PER_PEAK, N_PEAKS, peak_internal,
    _from_range, _from_unit, _unit, CENTER_RANGE, WIDTH_RANGE, default_params,
)

TISSUE_SYNONYMS = {
    "bone": ["bone", "bones", "cortical bone", "cortical", "skeleton"],
    "spongy": ["spongy bone", "spongy", "cancellous", "trabecular"],
    "soft": ["soft tissue", "soft", "tissue", "muscle"],
    "fat": ["fatty", "fat", "adipose"],
    "air": ["air", "background"],
}
# longest phrase first so "cortical bone" matches before "bone"
_SYNONYM_LOOKUP = sorted(
    ((phrase, tissue) for tissue, phrases in TISSUE_SYNONYMS.items() for phrase in phrases),
    key=lambda t: -len(t[0]),
)

STRENGTH_WORDS = {"slightly": 0.15, "moderately": 0.35, "strongly": 0.6}
NEAR_THRESHOLD_HU = 300.0


def _find_tissue(text: str):
    for phrase, tissue in _SYNONYM_LOOKUP:
        if phrase in text:
            return tissue
    return None


def parse_command_rule(text: str) -> dict:
    t = text.lower().strip()

    if "reset" in t:
        return {"target": None, "attribute": None, "direction": "reset", "strength": None}

    m = re.search(r"show only ([\w ]+)", t)
    if m:
        tissue = _find_tissue(m.group(1))
        if tissue:
            return {"target": tissue, "attribute": "opacity",
                     "direction": "show_only", "strength": None}

    m = re.search(r"(increase|decrease)\s+opacity\s+for\s+([\w ]+)", t)
    if m:
        direction, target_text = m.group(1), m.group(2)
        strength = "moderately"
        for word in STRENGTH_WORDS:
            if word in target_text:
                strength = word
                target_text = target_text.replace(word, "")
        tissue = _find_tissue(target_text)
        if tissue:
            return {"target": tissue, "attribute": "opacity",
                     "direction": direction, "strength": strength}

    raise ValueError(f"rule parser cannot parse: {text!r}")


def parse_command(text: str, parser: str = "rule", model: str = "qwen2.5:7b") -> dict:
    if parser == "llm":
        from commands import parse_command_llm  # defined in Task 5
        return parse_command_llm(text, model=model)
    return parse_command_rule(text)


def _peak_center_hu(params: np.ndarray, i: int) -> float:
    return peak_internal(params, i)["center"]


def _find_or_create_peak(params: np.ndarray, tissue: str):
    """Nearest peak to the tissue's HU; reseed the weakest peak if none is near."""
    target_hu = TISSUE_HU[tissue]
    centers = [_peak_center_hu(params, i) for i in range(N_PEAKS)]
    distances = [abs(c - target_hu) for c in centers]
    idx = int(np.argmin(distances))
    params = params.copy()
    if distances[idx] > NEAR_THRESHOLD_HU:
        heights = [peak_internal(params, i)["height"] for i in range(N_PEAKS)]
        idx = int(np.argmin(heights))
        base = idx * PARAMS_PER_PEAK
        params[base + 0] = _from_range(target_hu, *CENTER_RANGE)
        params[base + 1] = _from_range(80.0, *WIDTH_RANGE)
        params[base + 2] = _from_unit(0.05)
    return params, idx


def apply_command(cmd: dict, params: np.ndarray) -> np.ndarray:
    params = params.copy()

    if cmd["direction"] == "reset":
        return default_params()

    params, idx = _find_or_create_peak(params, cmd["target"])
    base = idx * PARAMS_PER_PEAK

    if cmd["direction"] == "show_only":
        for i in range(N_PEAKS):
            b = i * PARAMS_PER_PEAK
            params[b + 2] = _from_unit(0.7 if i == idx else 0.02)
        return params

    delta = STRENGTH_WORDS[cmd["strength"] or "moderately"]
    current_h = _unit(params[base + 2])
    sign = 1.0 if cmd["direction"] == "increase" else -1.0
    new_h = float(np.clip(current_h + sign * delta, 0.0, 1.0))
    params[base + 2] = _from_unit(new_h)
    return params
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_commands.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add commands.py tests/test_commands.py
git commit -m "feat: rule-based command parser and apply_command peak search"
```

---

### Task 5: `commands.py` — LLM parser via local Ollama

**Files:**
- Modify: `commands.py`
- Test: `tests/test_commands.py` (append)

- [ ] **Step 1: Write the failing test (mocks `urllib.request.urlopen`, no real Ollama needed)**

```python
# append to tests/test_commands.py
import json
from unittest.mock import patch, MagicMock
from commands import parse_command_llm

def _fake_response(payload: dict):
    resp = MagicMock()
    resp.read.return_value = json.dumps({"response": json.dumps(payload)}).encode()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp

def test_parse_command_llm_valid_response():
    payload = {"target": "bone", "attribute": "opacity",
               "direction": "increase", "strength": "strongly"}
    with patch("commands.urlopen", return_value=_fake_response(payload)):
        cmd = parse_command_llm("make the skeleton pop")
    assert cmd == payload

def test_parse_command_llm_falls_back_on_connection_error(capsys):
    with patch("commands.urlopen", side_effect=OSError("connection refused")):
        cmd = parse_command_llm("increase opacity for bone strongly")
    assert cmd["target"] == "bone"
    assert cmd["direction"] == "increase"
    assert "falling back to rule parser" in capsys.readouterr().out

def test_parse_command_llm_falls_back_on_invalid_schema(capsys):
    with patch("commands.urlopen", return_value=_fake_response({"foo": "bar"})):
        cmd = parse_command_llm("increase opacity for bone strongly")
    assert cmd["target"] == "bone"
    assert "falling back to rule parser" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_commands.py -k llm -v`
Expected: FAIL with `ImportError: cannot import name 'parse_command_llm'`

- [ ] **Step 3: Append the LLM parser to `commands.py`**

```python
# append to commands.py
import json as _json
from urllib.request import Request, urlopen
from urllib.error import URLError

OLLAMA_HOST = "http://localhost:11434"
VALID_DIRECTIONS = {"increase", "decrease", "show_only", "reset"}
VALID_STRENGTHS = set(STRENGTH_WORDS) | {None}

_SYSTEM_PROMPT = """You convert one spoken instruction about a volume-rendering \
transfer function into strict JSON, nothing else.

Tissues: air, fat, soft, spongy, bone.
Output schema (all four keys always present):
{"target": "<tissue>|null", "attribute": "opacity"|null,
 "direction": "increase"|"decrease"|"show_only"|"reset",
 "strength": "slightly"|"moderately"|"strongly"|null}

Never output numeric values. "reset" has target=null, attribute=null, strength=null.
"show_only" has strength=null. Respond with JSON only, no prose."""


def _validate_cmd(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    if set(obj.keys()) != {"target", "attribute", "direction", "strength"}:
        return False
    if obj["direction"] not in VALID_DIRECTIONS:
        return False
    if obj["target"] is not None and obj["target"] not in TISSUE_HU:
        return False
    if obj["strength"] not in VALID_STRENGTHS:
        return False
    return True


def parse_command_llm(text: str, model: str = "qwen2.5:7b", host: str = OLLAMA_HOST) -> dict:
    body = _json.dumps({
        "model": model,
        "system": _SYSTEM_PROMPT,
        "prompt": text,
        "format": "json",
        "stream": False,
    }).encode()
    req = Request(f"{host}/api/generate", data=body,
                   headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=10) as resp:
            raw = _json.loads(resp.read().decode())
        cmd = _json.loads(raw["response"])
    except (URLError, OSError, ValueError, KeyError) as exc:
        print(f"[llm-parser] falling back to rule parser: {exc}")
        return parse_command_rule(text)

    if not _validate_cmd(cmd):
        print(f"[llm-parser] falling back to rule parser: invalid schema {cmd!r}")
        return parse_command_rule(text)
    return cmd
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_commands.py -v`
Expected: PASS (11 tests total, no live Ollama needed — all network calls mocked)

- [ ] **Step 5: Commit**

```bash
git add commands.py tests/test_commands.py
git commit -m "feat: Ollama-backed LLM command parser with schema validation and fallback"
```

---

### Task 6: `evaluate.py`

**Files:**
- Create: `evaluate.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
import json
import numpy as np
from transfer import default_params
from commands import apply_command
from evaluate import objective, jsonl_append

def test_objective_positive_for_correct_increase():
    params = default_params()
    cmd = {"target": "bone", "attribute": "opacity",
           "direction": "increase", "strength": "strongly"}
    after = apply_command(cmd, params)
    assert objective(params, after, cmd) == 1

def test_objective_negative_when_no_change():
    params = default_params()
    cmd = {"target": "bone", "attribute": "opacity",
           "direction": "increase", "strength": "strongly"}
    assert objective(params, params, cmd) == -1

def test_objective_negative_for_wrong_direction():
    params = default_params()
    cmd_inc = {"target": "bone", "attribute": "opacity",
               "direction": "increase", "strength": "strongly"}
    after = apply_command(cmd_inc, params)
    cmd_dec = {**cmd_inc, "direction": "decrease"}
    assert objective(params, after, cmd_dec) == -1

def test_jsonl_append_writes_one_line(tmp_path):
    path = tmp_path / "log.jsonl"
    jsonl_append(str(path), {"a": 1})
    jsonl_append(str(path), {"b": 2})
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"b": 2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluate'`

- [ ] **Step 3: Implement `evaluate.py`**

```python
"""+1/-1 verdicts: objective (opacity_mass) and human (console)."""
import json
from transfer import TISSUE_BANDS, opacity_mass

EPS = 1e-4


def _effective_direction(cmd: dict):
    if cmd["direction"] in ("increase", "show_only"):
        return "increase"
    if cmd["direction"] == "decrease":
        return "decrease"
    return None  # reset — nothing to judge


def objective(params_before, params_after, cmd: dict) -> int:
    direction = _effective_direction(cmd)
    if direction is None:
        return 1
    lo, hi = TISSUE_BANDS[cmd["target"]]
    delta_target = opacity_mass(params_after, lo, hi) - opacity_mass(params_before, lo, hi)
    signed = delta_target if direction == "increase" else -delta_target
    if signed <= EPS:
        return -1
    max_other = 0.0
    for tissue, (tlo, thi) in TISSUE_BANDS.items():
        if tissue == cmd["target"]:
            continue
        d = abs(opacity_mass(params_after, tlo, thi) - opacity_mass(params_before, tlo, thi))
        max_other = max(max_other, d)
    return 1 if max_other <= abs(delta_target) else -1


def human(before_png: str, after_png: str) -> int:
    print(f"[human] before: {before_png}")
    print(f"[human] after:  {after_png}")
    while True:
        answer = input("Besser oder schlechter? (b/s): ").strip().lower()
        if answer == "b":
            return 1
        if answer == "s":
            return -1
        print("Bitte 'b' (besser) oder 's' (schlechter) eingeben.")


def jsonl_append(path: str, entry: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluate.py tests/test_evaluate.py
git commit -m "feat: objective opacity_mass evaluator, human console evaluator, jsonl logger"
```

---

### Task 7: `asr.py`

**Files:**
- Create: `asr.py`

No automated tests here — recording and model download require a mic and a
multi-second first-run download, verified manually in Task 10's smoke test via
`--wav`. Keep the module import-safe (no work at import time) so the rest of the
test suite is unaffected.

- [ ] **Step 1: Implement `asr.py`**

```python
"""faster-whisper wrapper: push-to-talk mic capture and file transcription."""
import time
import wave
import datetime
import os

import numpy as np
import sounddevice as sd

_MODEL_CACHE = {}
SAMPLE_RATE = 16000
AUDIO_DIR = "out/audio"


def _load_model(model_size: str = "small"):
    if model_size in _MODEL_CACHE:
        return _MODEL_CACHE[model_size]
    try:
        import mlx_whisper  # noqa: F401
        _MODEL_CACHE[model_size] = ("mlx", model_size)
        print(f"[asr] using mlx-whisper ({model_size})")
    except ImportError:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        _MODEL_CACHE[model_size] = ("faster-whisper", model)
        print(f"[asr] using faster-whisper ({model_size}, cpu, int8)")
    return _MODEL_CACHE[model_size]


def _transcribe_path(path: str, lang: str, model_size: str) -> str:
    kind, model = _load_model(model_size)
    if kind == "mlx":
        import mlx_whisper
        result = mlx_whisper.transcribe(path, language=lang)
        return result["text"].strip()
    segments, _ = model.transcribe(path, language=lang)
    return " ".join(seg.text.strip() for seg in segments).strip()


def _save_wav(path: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    audio_i16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_i16.tobytes())


def transcribe_mic(lang: str = "en", model_size: str = "small") -> str:
    input("Enter druecken zum Start der Aufnahme...")
    print("[asr] recording... Enter zum Stoppen.")
    frames = []
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                             callback=lambda indata, *_: frames.append(indata.copy()))
    with stream:
        input()
    audio = np.concatenate(frames, axis=0).flatten() if frames else np.zeros(0, dtype=np.float32)

    os.makedirs(AUDIO_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    wav_path = os.path.join(AUDIO_DIR, f"{stamp}.wav")
    txt_path = os.path.join(AUDIO_DIR, f"{stamp}.txt")
    _save_wav(wav_path, audio)

    start = time.time()
    text = _transcribe_path(wav_path, lang, model_size)
    duration = time.time() - start
    with open(txt_path, "w") as f:
        f.write(text)
    print(f"[asr] '{text}' ({duration:.2f}s)")
    return text


def transcribe_file(path: str, lang: str = "en", model_size: str = "small") -> str:
    start = time.time()
    text = _transcribe_path(path, lang, model_size)
    duration = time.time() - start
    txt_path = os.path.splitext(path)[0] + ".txt"
    with open(txt_path, "w") as f:
        f.write(text)
    print(f"[asr] '{text}' ({duration:.2f}s)")
    return text
```

- [ ] **Step 2: Verify import safety**

Run: `python -c "import asr; print('ok')"`
Expected: prints `ok`, no side effects, no model download triggered.

- [ ] **Step 3: Commit**

```bash
git add asr.py
git commit -m "feat: faster-whisper push-to-talk mic and file transcription"
```

---

### Task 8: `mvp.py`

**Files:**
- Create: `mvp.py`

State persists in `out/state.json` (a 24-float list) so sequential CLI invocations
compose — required for the multi-step demo in the closing task. `python mvp.py`
with no flags renders whatever state currently exists (default on first run)
without changing it.

- [ ] **Step 1: Implement `mvp.py`**

```python
"""CLI: render, apply a command, hill-climb, or listen on the mic."""
import argparse
import datetime
import json
import os
import uuid

import numpy as np
from PIL import Image

from phantom import build_phantom
from transfer import default_params, opacity_mass, TISSUE_BANDS, N_PEAKS, PARAMS_PER_PEAK, peak_internal
from render import render, grab, features
from commands import parse_command, apply_command, STRENGTH_WORDS, _find_or_create_peak
from evaluate import objective, human, jsonl_append

STATE_PATH = "out/state.json"
LOG_PATH = "out/log.jsonl"
PREF_PATH = "out/preferences.jsonl"

_PHANTOM = None


def get_phantom():
    global _PHANTOM
    if _PHANTOM is None:
        _PHANTOM = build_phantom()
    return _PHANTOM


def load_state() -> np.ndarray:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return np.array(json.load(f), dtype=np.float64)
    return default_params()


def save_state(params: np.ndarray) -> None:
    os.makedirs("out", exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(params.tolist(), f)


def save_png(params: np.ndarray, path: str) -> np.ndarray:
    img = grab(render(get_phantom(), params))
    Image.fromarray(img).save(path)
    return img


def masses(params: np.ndarray) -> dict:
    return {t: opacity_mass(params, lo, hi) for t, (lo, hi) in TISSUE_BANDS.items()}


def print_step(cmd, before, after, feats_before, feats_after, verdict):
    print("command:", cmd)
    print("opacity mass before:", {k: round(v, 4) for k, v in masses(before).items()})
    print("opacity mass after: ", {k: round(v, 4) for k, v in masses(after).items()})
    print("features before:", {k: round(v, 3) for k, v in feats_before.items()})
    print("features after: ", {k: round(v, 3) for k, v in feats_after.items()})
    print("verdict:", verdict)


def log_step(cmd, before, after, feats_before, feats_after, verdict):
    jsonl_append(LOG_PATH, {
        "timestamp": datetime.datetime.now().isoformat(),
        "command": cmd,
        "params_before": before.tolist(),
        "params_after": after.tolist(),
        "features_before": feats_before,
        "features_after": feats_after,
        "verdict": verdict,
    })


def log_preference(session_id, cmd_text, cmd, before, after, feats_before, feats_after,
                    before_png, after_png, human_verdict, objective_verdict):
    jsonl_append(PREF_PATH, {
        "timestamp": datetime.datetime.now().isoformat(),
        "session_id": session_id,
        "cmd_text": cmd_text,
        "cmd_dict": cmd,
        "params_before": before.tolist(),
        "params_after": after.tolist(),
        "features_before": feats_before,
        "features_after": feats_after,
        "before_png": before_png,
        "after_png": after_png,
        "human_verdict": "better" if human_verdict == 1 else "worse",
        "objective_verdict": "better" if objective_verdict == 1 else "worse",
    })


def hill_climb(cmd_text, cmd, params, steps, use_human, session_id, out_dir="out"):
    os.makedirs(out_dir, exist_ok=True)
    _, idx = _find_or_create_peak(params, cmd["target"])
    base = idx * PARAMS_PER_PEAK
    sign = 1.0 if cmd["direction"] == "increase" else -1.0
    step = STRENGTH_WORDS[cmd["strength"] or "moderately"]

    current = params.copy()
    _, current_idx = _find_or_create_peak(current, cmd["target"])
    base = current_idx * PARAMS_PER_PEAK

    for i in range(steps):
        proposed = current.copy()
        internal = (proposed[base + 2] + 1.0) / 2.0
        internal = float(np.clip(internal + sign * step, 0.0, 1.0))
        proposed[base + 2] = internal * 2.0 - 1.0

        before_png = os.path.join(out_dir, f"step_{i:03d}_before.png")
        after_png = os.path.join(out_dir, f"step_{i:03d}_after.png")
        img_before = save_png(current, before_png)
        img_after = save_png(proposed, after_png)
        feats_before, feats_after = features(img_before), features(img_after)

        obj_verdict = objective(current, proposed, cmd)
        if use_human:
            human_verdict = human(before_png, after_png)
            log_preference(session_id, cmd_text, cmd, current, proposed,
                            feats_before, feats_after, before_png, after_png,
                            human_verdict, obj_verdict)
            verdict = human_verdict
        else:
            verdict = obj_verdict

        print_step(cmd, current, proposed, feats_before, feats_after, verdict)
        log_step(cmd, current, proposed, feats_before, feats_after, verdict)

        if verdict == 1:
            current = proposed
            step = min(step * 1.2, 1.0)
        else:
            step *= 0.5
        if step < 0.01:
            print(f"[mvp] step size below threshold after {i + 1} steps, stopping")
            break

    return current


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd")
    ap.add_argument("--learn", action="store_true")
    ap.add_argument("--human", action="store_true")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--listen", action="store_true")
    ap.add_argument("--wav")
    ap.add_argument("--parser", choices=["rule", "llm"], default="rule")
    ap.add_argument("--llm-model", default="qwen2.5:7b")
    ap.add_argument("--model-size", default="small")
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()

    os.makedirs("out", exist_ok=True)
    params = load_state()
    session_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    cmd_text = args.cmd
    if args.listen:
        import asr
        cmd_text = asr.transcribe_mic(lang=args.lang, model_size=args.model_size)
    elif args.wav:
        import asr
        cmd_text = asr.transcribe_file(args.wav, lang=args.lang, model_size=args.model_size)

    if not cmd_text:
        save_png(params, "out/start.png")
        print("[mvp] rendered current state -> out/start.png")
        return

    cmd = parse_command(cmd_text, parser=args.parser, model=args.llm_model)
    print("parsed command:", cmd)

    if args.learn and cmd["direction"] in ("increase", "decrease"):
        new_params = hill_climb(cmd_text, cmd, params, args.steps, args.human, session_id)
    else:
        before = params
        new_params = apply_command(cmd, params)
        img_before = save_png(before, "out/before.png")
        img_after = save_png(new_params, "out/after.png")
        feats_before, feats_after = features(img_before), features(img_after)
        verdict = objective(before, new_params, cmd)
        print_step(cmd, before, new_params, feats_before, feats_after, verdict)
        log_step(cmd, before, new_params, feats_before, feats_after, verdict)

    save_state(new_params)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add `Pillow` to `requirements.txt`** (PNG writing)

```
numpy
vtk
faster-whisper
sounddevice
Pillow
pytest
```

Run: `pip install -r requirements.txt`

- [ ] **Step 3: Smoke test — no-arg render**

Run: `python mvp.py`
Expected: prints `[render] using ...` and `[mvp] rendered current state -> out/start.png`; `out/start.png` and `out/state.json` exist.

- [ ] **Step 4: Smoke test — single command**

Run: `rm -f out/state.json && python mvp.py --cmd "increase opacity for bone strongly"`
Expected: prints parsed command, mass tables, verdict; `out/before.png`, `out/after.png` exist; bone mass after > before.

- [ ] **Step 5: Commit**

```bash
git add mvp.py requirements.txt
git commit -m "feat: mvp.py CLI, hill-climbing loop, state persistence, logging"
```

---

### Task 9: `eval_parsers.py` + `data/parser_eval_phrases.json`

**Files:**
- Create: `data/parser_eval_phrases.json`
- Create: `eval_parsers.py`

- [ ] **Step 1: Write the phrase fixture**

```json
[
  {"text": "I can't see the bones very well", "expected_target": "bone", "expected_direction": "increase"},
  {"text": "the soft tissue is hiding everything", "expected_target": "soft", "expected_direction": "decrease"},
  {"text": "make the skeleton pop", "expected_target": "bone", "expected_direction": "increase"},
  {"text": "the fat layer is way too visible", "expected_target": "fat", "expected_direction": "decrease"},
  {"text": "let me see through the muscle", "expected_target": "soft", "expected_direction": "decrease"},
  {"text": "bring the cortical shell forward", "expected_target": "bone", "expected_direction": "increase"},
  {"text": "the spongy core is barely there", "expected_target": "spongy", "expected_direction": "increase"},
  {"text": "tone down the fatty layer a bit", "expected_target": "fat", "expected_direction": "decrease"},
  {"text": "hide the soft tissue completely", "expected_target": "soft", "expected_direction": "decrease"},
  {"text": "the trabecular bone is invisible", "expected_target": "spongy", "expected_direction": "increase"},
  {"text": "punch up the bone density", "expected_target": "bone", "expected_direction": "increase"},
  {"text": "everything is washed out by the fat", "expected_target": "fat", "expected_direction": "decrease"},
  {"text": "I want the skeleton to stand out more", "expected_target": "bone", "expected_direction": "increase"},
  {"text": "fade the muscle layer", "expected_target": "soft", "expected_direction": "decrease"},
  {"text": "the cancellous bone needs more presence", "expected_target": "spongy", "expected_direction": "increase"},
  {"text": "cut back on the adipose tissue", "expected_target": "fat", "expected_direction": "decrease"},
  {"text": "let the cortical bone dominate the view", "expected_target": "bone", "expected_direction": "increase"},
  {"text": "the tissue layer is drowning out the bone", "expected_target": "soft", "expected_direction": "decrease"},
  {"text": "give the spongy bone a boost", "expected_target": "spongy", "expected_direction": "increase"},
  {"text": "the fat is completely obscuring the structure", "expected_target": "fat", "expected_direction": "decrease"}
]
```

- [ ] **Step 2: Implement `eval_parsers.py`**

```python
"""Compare rule vs LLM parser accuracy on free-form phrases the rule parser
is expected to fail on. Prints a table; this is the point of the change."""
import argparse
import json

from commands import parse_command_rule, parse_command_llm

PHRASES_PATH = "data/parser_eval_phrases.json"


def _check(text, expected_target, expected_direction, fn):
    try:
        cmd = fn(text)
    except Exception:
        return False, None
    ok = cmd.get("target") == expected_target and cmd.get("direction") == expected_direction
    return ok, cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-model", default="qwen2.5:7b")
    args = ap.parse_args()

    with open(PHRASES_PATH) as f:
        phrases = json.load(f)

    rule_correct = llm_correct = 0
    rows = []
    for item in phrases:
        text, exp_t, exp_d = item["text"], item["expected_target"], item["expected_direction"]
        r_ok, r_cmd = _check(text, exp_t, exp_d, parse_command_rule)
        l_ok, l_cmd = _check(text, exp_t, exp_d,
                              lambda t: parse_command_llm(t, model=args.llm_model))
        rule_correct += r_ok
        llm_correct += l_ok
        rows.append((text, exp_t, exp_d, r_ok, l_ok))

    print(f"{'phrase':<45} {'expected':<20} {'rule':<6} {'llm':<6}")
    for text, exp_t, exp_d, r_ok, l_ok in rows:
        print(f"{text:<45} {exp_t}/{exp_d:<10} {'ok' if r_ok else 'X':<6} {'ok' if l_ok else 'X':<6}")

    n = len(phrases)
    print(f"\nrule parser:  {rule_correct}/{n} correct")
    print(f"llm parser:   {llm_correct}/{n} correct")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run**

Run: `python eval_parsers.py`
Expected: table printed, rule parser near 0/20 (these phrases are chosen to defeat
it), LLM parser's score depends on Ollama being up — if it isn't, every row falls
back to the rule parser and both columns match (fallback messages will print above
the table).

- [ ] **Step 4: Commit**

```bash
git add data/parser_eval_phrases.json eval_parsers.py
git commit -m "feat: rule vs LLM parser accuracy comparison over free-form phrases"
```

---

### Task 10: `stats.py`

**Files:**
- Create: `stats.py`

- [ ] **Step 1: Implement `stats.py`**

```python
"""Analyze out/preferences.jsonl: pair counts, agreement rate, disagreements."""
import argparse
import json
from collections import defaultdict

DEFAULT_PATH = "out/preferences.jsonl"


def load(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT_PATH)
    args = ap.parse_args()

    entries = load(args.path)
    if not entries:
        print(f"no preference pairs found in {args.path}")
        return

    by_tissue = defaultdict(list)
    for e in entries:
        tissue = e["cmd_dict"].get("target") or "unknown"
        by_tissue[tissue].append(e)

    print("pairs per target tissue:")
    for tissue, items in sorted(by_tissue.items()):
        print(f"  {tissue}: {len(items)}")

    def agreement(items):
        agree = sum(1 for e in items if e["human_verdict"] == e["objective_verdict"])
        return agree, len(items)

    total_agree, total_n = agreement(entries)
    print(f"\noverall agreement: {total_agree}/{total_n} ({100 * total_agree / total_n:.1f}%)")
    print("agreement per tissue:")
    for tissue, items in sorted(by_tissue.items()):
        a, n = agreement(items)
        print(f"  {tissue}: {a}/{n} ({100 * a / n:.1f}%)")

    disagreements = [e for e in entries if e["human_verdict"] != e["objective_verdict"]]
    print(f"\ndisagreements ({len(disagreements)}):")
    for e in disagreements:
        print(f"  [{e['cmd_dict'].get('target')}] human={e['human_verdict']} "
              f"objective={e['objective_verdict']}  {e['before_png']} -> {e['after_png']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify with a synthetic fixture**

Run:
```bash
python -c "
import json
rows = [
  {'cmd_dict': {'target': 'bone'}, 'human_verdict': 'better', 'objective_verdict': 'better',
   'before_png': 'a.png', 'after_png': 'b.png'},
  {'cmd_dict': {'target': 'bone'}, 'human_verdict': 'worse', 'objective_verdict': 'better',
   'before_png': 'c.png', 'after_png': 'd.png'},
]
with open('/tmp/prefs_test.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
"
python stats.py --path /tmp/prefs_test.jsonl
```
Expected: `bone: 2` pairs, overall agreement `1/2 (50.0%)`, one disagreement line listing `c.png -> d.png`.

- [ ] **Step 3: Commit**

```bash
git add stats.py
git commit -m "feat: preference-pair stats — agreement rate and disagreement listing"
```

---

### Task 11: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# Voice-Driven Transfer Function MVP

Local prototype: synthetic CT phantom (HU) -> 4-Gaussian-peak transfer function ->
offscreen VTK render, driven by rule-based or local-LLM-parsed text/voice commands,
searched by hill-climbing against an exact opacity-mass metric or a human judge.

## Setup

    pip install -r requirements.txt

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
up a view.

## Tests

    pytest -v
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with setup and usage"
```

---

### Task 12: Update `architecture-mvp.drawio`

**Files:**
- Modify: `architecture-mvp.drawio`

- [ ] **Step 1: Use the drawio skill to update the diagram**

Invoke the `drawio` skill (not manual XML editing) to:
- Promote the "LLM Parser" box from dashed/"later" to solid, and label it
  "flag-selectable, default: rule" with an edge from Parser showing the
  `--parser rule|llm` choice (both feeding the same command-dict output).
- Delete the "Reward Model / distilled from MLLM judge" box and its edges entirely.
- Add a `preferences.jsonl` box fed from `eval`'s human-judgment edge, and a
  `stats.py` box consuming it.
- Rewrite the scope note: state explicitly that an MLLM judge was considered and
  rejected in favor of the exact `opacity_mass` metric, and that human/objective
  verdicts are both logged on every human-judged step for later agreement analysis.

- [ ] **Step 2: Commit**

```bash
git add architecture-mvp.drawio
git commit -m "docs: update architecture diagram — LLM parser built, no MLLM judge, preference logging"
```

---

### Task 13: Full test suite + closing demonstrations

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `pytest -v`
Expected: all tests pass (phantom, transfer, render, commands, evaluate).

- [ ] **Step 2: Demonstration 1 — monotonic bone hill-climb**

```bash
rm -f out/state.json out/log.jsonl
python mvp.py --cmd "increase opacity for bone strongly" --learn --steps 15
```
Capture the per-step bone opacity-mass column from stdout into a table; report
whether it rises monotonically (accept/revert steps are expected to interrupt
strict monotonicity — that's the point of showing the table, not just the final
number).

- [ ] **Step 3: Demonstration 2 — sequential emphasis, where it "swims"**

```bash
rm -f out/state.json out/log.jsonl
python mvp.py --cmd "increase opacity for bone strongly" --learn --steps 8
python mvp.py --cmd "increase opacity for soft strongly" --learn --steps 8
python mvp.py --cmd "increase opacity for bone strongly" --learn --steps 8
```
Capture bone and soft opacity-mass after each of the three runs into a table;
report where bone's mass regresses when soft is emphasized, and whether the final
bone run recovers it fully or only partially.

- [ ] **Step 4: Parser comparison**

```bash
python eval_parsers.py
```
Capture the accuracy table (rule vs LLM, both as fraction correct out of 20).

- [ ] **Step 5: Report all three tables to the user in chat.**
```
