# RL v2 Plan 3: Views and Visibility Estimate

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A fast per-tissue visibility and brightness estimate for a transfer function on a CT volume, averaged over 6 fixed views, validated against real VTK renders — the reward signal and observation source for the RL v2 environment.

**Architecture:** `views.py` defines the 6 standard cameras of a volume (renderer-neutral). `render.py` learns to render those cameras. `visibility.py` resamples each volume once per view into a cube whose first axis runs front-to-back, stores a quantized copy on disk, and composites front-to-back with torch to produce per-tissue visibility, per-tissue brightness and coverage in ~15 ms. `tools/validate_visibility.py` compares the estimate against VTK renders and is a gate: the environment (Plan 4) is not built until it passes.

**Tech Stack:** Python 3.14 (`.venv`), torch (CPU), numpy, VTK, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-rl-v2-visibility-rlhf-design.md`, sections "Views" and "Visibility and brightness estimate".

**Measured facts (controller prototype, 2026-09-15, ct_chest + ts_s0454 + ts_s1337):**
- Resampling one volume into 6 view cubes at N=80: ~0.35 s; cached afterwards.
- `features()` at N=80 with a 256-entry lookup table: **15 ms on CPU** (N=96: 30 ms; MPS gives no useful gain). Values are stable across N (bone visibility 0.0017–0.0021 on ct_chest at default TF).
- Occlusion behaves as designed: on ct_chest, clearing the fat and soft peaks raises bone visibility 9.4× (0.0020 → 0.0188).
- Agreement with VTK (12 random TFs per volume; reference = mean rendered luminance with a tissue present minus the same with that tissue's voxels removed from the volume), Pearson on visibility × brightness: **bone 0.90/0.94, fat 0.87/0.82, soft 0.81/0.88** (ct_chest / ts_s1337). Coverage vs. rendered coverage: Pearson 0.92–0.96.
- **Spongy is not validatable this way:** its contribution is at the render's noise floor (Pearson 0.22/0.66) because trabecular bone sits inside cortical bone. The gate therefore skips tissues whose reference contribution range is below a noise floor, and reports them as unvalidated.
- A naive check ("proxy bone visibility vs. total image brightness") anti-correlates and is wrong: raising a peak also makes its HU tails opaque, so total brightness rises while the target tissue's own share can fall. Only the masked-volume reference is meaningful.

**Repo rules:**
- Run from the repository root with `.venv/bin/python`.
- Commit messages: plain `git commit -m "..."`, **no** trailers.
- Stage files explicitly by path; never `git add -A`/`.`. Leave untracked `docs/prompts_report.pdf` and `docs/vr-integration.md` alone.
- Branch `rl-v2`.

---

## File map

| File | Responsibility |
|---|---|
| `views.py` (new) | the 6 standard view directions and renderer-neutral cameras for a volume |
| `tests/test_views.py` (new) | direction/camera geometry |
| `render.py` | render a renderer-neutral camera (position/focal point/view up/parallel scale) |
| `tests/test_render.py` | camera-form tests |
| `visibility.py` (new) | per-volume view cubes, disk cache, `features()`, `solo_max()`, histogram |
| `tests/test_visibility.py` (new) | compositing, occlusion, brightness, coverage, caching |
| `tools/build_visibility_cache.py` (new) | precompute caches for all RL v2 volumes |
| `tools/validate_visibility.py` (new) | VTK agreement gate, writes `out/visibility_validation.json` |
| `tests/test_validate_visibility.py` (new) | correlation/noise-floor helpers |

---

### Task 1: `views.py`

**Files:**
- Create: `views.py`
- Test: `tests/test_views.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_views.py`:

```python
import numpy as np
import pytest

import views


def test_six_directions_are_unit_and_distinct():
    directions = views.view_directions()
    assert len(directions) == views.N_VIEWS == 6
    for d in directions:
        assert d.shape == (3,)
        assert np.isclose(np.linalg.norm(d), 1.0)
        assert np.isclose(d[2], 0.0)                     # rotations about the superior axis
    for i in range(6):
        for j in range(i + 1, 6):
            assert not np.allclose(directions[i], directions[j])


def test_first_direction_looks_from_anterior_towards_posterior():
    assert np.allclose(views.view_directions()[0], [0.0, -1.0, 0.0])


def test_directions_are_evenly_spaced_by_sixty_degrees():
    directions = views.view_directions()
    for i in range(6):
        angle = np.degrees(np.arccos(np.clip(directions[i] @ directions[(i + 1) % 6], -1, 1)))
        assert np.isclose(angle, 60.0)


def test_cameras_frame_the_volume_from_each_direction():
    extent = (200.0, 100.0, 300.0)                        # physical size in mm
    cameras = views.cameras_for_extent(extent)
    assert len(cameras) == 6
    center = np.array(extent) / 2.0
    for camera, direction in zip(cameras, views.view_directions()):
        assert np.allclose(camera["focal_point"], center)
        assert np.allclose(camera["view_up"], [0.0, 0.0, 1.0])
        offset = center - np.array(camera["position"])
        assert np.allclose(offset / np.linalg.norm(offset), direction)
        assert np.linalg.norm(offset) > max(extent)       # outside the volume
        assert camera["parallel_scale"] >= max(extent) / 2.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_views.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'views'`.

- [ ] **Step 3: Implement**

Create `views.py`:

```python
"""The fixed set of views RL v2 looks at a volume from.

Volumes are in canonical RAS axis order (axis 0 -> patient right, 1 ->
anterior, 2 -> superior; see datasets.load_dataset(..., canonical=True)).
View 0 looks from in front of the patient towards the back; views 1-5 are
that direction rotated around the patient's head-to-foot axis in 60 degree
steps. The same views are used by the visibility estimate, by the rendered
views the reward model sees, and as the start camera of the collect page, so
all three describe the same thing.
"""
import numpy as np

N_VIEWS = 6
VIEW_UP = np.array([0.0, 0.0, 1.0])       # superior
CAMERA_DISTANCE_FACTOR = 2.2              # times the largest extent, so the camera clears the volume
PARALLEL_SCALE_FACTOR = 0.55              # half-height of the parallel projection, times the largest extent


def view_directions(n: int = N_VIEWS) -> list:
    """Unit vectors pointing from each camera towards the volume."""
    angles = np.deg2rad(360.0 * np.arange(n) / n)
    return [np.array([np.sin(a), -np.cos(a), 0.0]) for a in angles]


def cameras_for_extent(extent) -> list:
    """Renderer-neutral cameras for a volume of this physical size (mm)."""
    extent = np.asarray(extent, dtype=np.float64)
    center = extent / 2.0
    largest = float(extent.max())
    cameras = []
    for direction in view_directions():
        cameras.append({
            "position": (center - direction * largest * CAMERA_DISTANCE_FACTOR).tolist(),
            "focal_point": center.tolist(),
            "view_up": VIEW_UP.tolist(),
            "parallel_scale": largest * PARALLEL_SCALE_FACTOR,
        })
    return cameras


def cameras_for_volume(volume, spacing) -> list:
    """Renderer-neutral cameras for a loaded volume."""
    extent = np.asarray(volume.shape, dtype=np.float64) * np.asarray(spacing, dtype=np.float64)
    return cameras_for_extent(extent)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest -q tests/test_views.py`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add views.py tests/test_views.py
git commit -m "feat(rl): fixed view set for RL v2 volumes"
```

---

### Task 2: Renderer-neutral cameras in `render.py`

**Files:**
- Modify: `render.py`
- Test: `tests/test_render.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_render.py` (it already imports `numpy as np`, `render`, `phantom`/`transfer` helpers — check the file's existing imports and reuse them; add `import views` and any missing import):

```python
def test_render_accepts_a_renderer_neutral_camera():
    volume = build_phantom()
    params = default_params()
    cameras = views.cameras_for_volume(volume, (1.0, 1.0, 1.0))
    front = render.grab(render.render(volume, params, (1.0, 1.0, 1.0), cameras[0]))
    side = render.grab(render.render(volume, params, (1.0, 1.0, 1.0), cameras[2]))
    assert front.shape == side.shape
    assert front.max() > 0                       # something is visible
    assert not np.array_equal(front, side)       # different views differ


def test_render_still_accepts_the_azimuth_camera():
    volume = build_phantom()
    params = default_params()
    image = render.grab(render.render(volume, params, (1.0, 1.0, 1.0),
                                      {"azimuth": 30.0, "elevation": 20.0, "zoom": 1.0}))
    assert image.shape[2] == 3 and image.max() > 0


def test_same_camera_renders_deterministically():
    volume = build_phantom()
    params = default_params()
    camera = views.cameras_for_volume(volume, (1.0, 1.0, 1.0))[1]
    first = render.grab(render.render(volume, params, (1.0, 1.0, 1.0), camera))
    second = render.grab(render.render(volume, params, (1.0, 1.0, 1.0), camera))
    assert np.array_equal(first, second)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_render.py`
Expected: the first and third new tests FAIL (the camera dict has no `"azimuth"` key → `KeyError`).

- [ ] **Step 3: Implement**

In `render.py`, replace the camera block inside `render()`:

```python
    bounds = _visible_bounds(volume, params, spacing)
    if bounds is not None:
        renderer.ResetCamera(bounds)
    else:
        renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    cam_state = camera or {"azimuth": 30.0, "elevation": 20.0, "zoom": 1.0}
    cam.Azimuth(cam_state["azimuth"])
    cam.Elevation(cam_state["elevation"])
    cam.Zoom(cam_state["zoom"])
    renderer.ResetCameraClippingRange()
```

with:

```python
    cam = renderer.GetActiveCamera()
    cam_state = camera or {"azimuth": 30.0, "elevation": 20.0, "zoom": 1.0}
    if "position" in cam_state:
        # Renderer-neutral camera (views.py): an absolute placement in the
        # volume's own millimetre coordinates, so the same camera means the
        # same picture in VTK, in vtk.js and in the visibility estimate. No
        # ResetCamera here -- that would re-frame and undo the placement.
        cam.SetParallelProjection("parallel_scale" in cam_state)
        cam.SetPosition(*[float(v) for v in cam_state["position"]])
        cam.SetFocalPoint(*[float(v) for v in cam_state["focal_point"]])
        cam.SetViewUp(*[float(v) for v in cam_state["view_up"]])
        if "parallel_scale" in cam_state:
            cam.SetParallelScale(float(cam_state["parallel_scale"]))
    else:
        cam.SetParallelProjection(False)
        bounds = _visible_bounds(volume, params, spacing)
        if bounds is not None:
            renderer.ResetCamera(bounds)
        else:
            renderer.ResetCamera()
        cam.Azimuth(cam_state["azimuth"])
        cam.Elevation(cam_state["elevation"])
        cam.Zoom(cam_state["zoom"])
    renderer.ResetCameraClippingRange()
```

Update `render()`'s docstring/comment if it claims the camera is always azimuth/elevation.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest -q tests/test_render.py tests/test_server.py`
Expected: all pass (the chat UI keeps using the azimuth form).

- [ ] **Step 5: Commit**

```bash
git add render.py tests/test_render.py
git commit -m "feat(render): render renderer-neutral cameras"
```

---

### Task 3: `visibility.py` core

**Files:**
- Create: `visibility.py`
- Test: `tests/test_visibility.py`

Definitions this task implements, all per volume and averaged over the 6 views:

- The volume is resampled into one cube per view (side `GRID_N`, isotropic step `side/GRID_N` mm, covering the volume's largest extent, centred on the volume). Cube axis 0 runs **front to back** away from the camera; axes 1 and 2 are the image plane. Samples outside the volume are air.
- Each cube sample is quantized to an index into a 256-entry lookup table over `transfer.CENTER_RANGE`; that `uint8` array is what gets cached.
- Per sample: opacity `a` from the transfer function's opacity table, corrected for the step length, `a = 1 - (1 - a_table)^(step_mm / 1mm)`; luminance `lum` from the colour table (`0.2126 R + 0.7152 G + 0.0722 B`).
- Front-to-back transmittance `T` is the exclusive running product of `1 - a` along axis 0; a sample's contribution is `w = T * a`.
- `vis[t]` = sum of `w` over samples labelled `t`, divided by the number of rays (`n_views * GRID_N²`).
- `bright[t]` = `sum(w * lum) / sum(w)` over samples labelled `t`, or 0 when that tissue contributes nothing.
- `coverage` = fraction of rays whose accumulated opacity `1 - prod(1 - a)` reaches `COVERAGE_THRESHOLD`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_visibility.py`:

```python
import numpy as np
import pytest

import visibility
from transfer import TISSUE_BANDS, default_params, PARAMS_PER_PEAK


def _slab_volume(front_hu, back_hu, n=24):
    """Two slabs stacked along the anterior axis: front slab is nearer to view 0."""
    volume = np.full((n, n, n), -1000.0, dtype=np.float32)
    volume[:, n // 2:, :] = front_hu      # higher anterior index = nearer the front camera
    volume[:, : n // 2, :] = back_hu
    return volume


def _params(heights):
    """A transfer function with the given per-peak heights (unit scale), default widths."""
    params = default_params().copy()
    for index, height in enumerate(heights):
        params[index * PARAMS_PER_PEAK + 2] = height * 2.0 - 1.0
    return params


def _model(volume, spacing=(2.0, 2.0, 2.0), **kwargs):
    return visibility.VisibilityModel.from_volume(volume, spacing, **kwargs)


def test_transparent_transfer_function_is_invisible():
    model = _model(_slab_volume(700.0, 900.0))
    features = model.features(_params([0.0, 0.0, 0.0, 0.0]))
    assert features["coverage"] == 0.0
    for tissue in visibility.TISSUES:
        assert features["vis"][tissue] == pytest.approx(0.0, abs=1e-6)
        assert features["bright"][tissue] == 0.0


def test_front_tissue_occludes_the_one_behind_it():
    # front slab: soft tissue; back slab: bone
    volume = _slab_volume(front_hu=50.0, back_hu=900.0)
    model = _model(volume)
    occluded = model.features(_params([0.0, 0.9, 0.0, 0.9]))["vis"]["bone"]
    cleared = model.features(_params([0.0, 0.0, 0.0, 0.9]))["vis"]["bone"]
    assert cleared > occluded * 2.0


def test_visibility_rises_with_opacity_until_it_saturates():
    model = _model(_slab_volume(-1000.0, 900.0))       # nothing in front of the bone
    values = [model.features(_params([0, 0, 0, h]))["vis"]["bone"] for h in (0.1, 0.3, 0.6)]
    assert values[0] < values[1] < values[2]


def test_brightness_follows_peak_colour():
    volume = _slab_volume(-1000.0, 900.0)
    model = _model(volume)
    dark = default_params().copy()
    bright = default_params().copy()
    for channel in range(3):
        dark[3 * PARAMS_PER_PEAK + 3 + channel] = -0.6      # dim bone colour
        bright[3 * PARAMS_PER_PEAK + 3 + channel] = 1.0     # white bone colour
    assert model.features(bright)["bright"]["bone"] > model.features(dark)["bright"]["bone"]


def test_coverage_counts_rays_that_accumulate_opacity():
    model = _model(_slab_volume(-1000.0, 900.0))
    thin = model.features(_params([0, 0, 0, 0.02]))["coverage"]
    thick = model.features(_params([0, 0, 0, 1.0]))["coverage"]
    assert 0.0 <= thin < thick <= 1.0


def test_views_see_different_things():
    """Bone behind soft tissue is hidden from the front and open from behind."""
    n = 24
    volume = np.full((n, n, n), -1000.0, dtype=np.float32)
    volume[:, n // 2: 3 * n // 4, :] = 900.0     # bone in the middle
    volume[:, 3 * n // 4:, :] = 50.0             # soft tissue in front of it (anterior)
    model = _model(volume, n_views=2)            # view 0 anterior, view 1 posterior
    per_view = model.per_view_visibility(_params([0.0, 0.9, 0.0, 0.9]), "bone")
    assert len(per_view) == 2
    assert per_view[1] > per_view[0] * 1.5       # seen from behind, bone is not occluded


def test_solo_max_is_the_visibility_of_that_tissue_alone():
    model = _model(_slab_volume(50.0, 900.0))
    solo = model.solo_max("bone")
    assert solo > model.features(_params([0.0, 0.9, 0.0, 0.9]))["vis"]["bone"]
    assert 0.0 < solo <= 1.0


def test_histogram_is_normalised_and_shaped():
    model = _model(_slab_volume(50.0, 900.0))
    histogram = model.histogram
    assert histogram.shape == (visibility.HISTOGRAM_BINS,)
    assert histogram.sum() == pytest.approx(1.0)
    assert (histogram >= 0).all()


def test_labels_follow_tissue_bands():
    lo, hi = TISSUE_BANDS["bone"]
    model = _model(_slab_volume(-1000.0, (lo + hi) / 2.0))
    assert model.features(_params([0, 0, 0, 0.9]))["vis"]["bone"] > 0.0
    assert model.features(_params([0, 0, 0.9, 0]))["vis"]["spongy"] == pytest.approx(0.0, abs=1e-6)


def test_features_are_deterministic():
    model = _model(_slab_volume(50.0, 900.0))
    params = _params([0.2, 0.4, 0.1, 0.7])
    assert model.features(params) == model.features(params)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_visibility.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'visibility'`.

- [ ] **Step 3: Implement**

Create `visibility.py`:

```python
"""How much of each tissue a transfer function actually shows.

Direct volume rendering is expensive to run inside an RL loop, and a rendered
image does not say which tissue produced which pixel. This module estimates
both cheaply: it resamples a volume once per view into a cube whose first axis
runs front to back, then composites that cube front to back for a given
transfer function. The result is, per tissue, how much of the final image that
tissue contributes (`vis`) and how bright it appears (`bright`), plus how much
of the frame is covered at all (`coverage`).

It is an estimate, not a renderer: no shading, no perspective, a coarse grid.
tools/validate_visibility.py measures how well it tracks real VTK renders.
"""
import hashlib
import os

import numpy as np
import torch

from transfer import CENTER_RANGE, TISSUE_BANDS, _opacity_and_color_at
from views import N_VIEWS, view_directions

TISSUES = ("fat", "soft", "spongy", "bone")
GRID_N = 80                     # samples per cube axis; 15 ms/features() on CPU
LUT_SIZE = 256                  # quantization levels over CENTER_RANGE
HISTOGRAM_BINS = 16
COVERAGE_THRESHOLD = 0.3        # accumulated opacity for a ray to count as covered
AIR_HU = CENTER_RANGE[0]
LUMINANCE = np.array([0.2126, 0.7152, 0.0722])
CACHE_DIR = "out/cache/visibility"
CACHE_VERSION = 1


def quantize(values: np.ndarray) -> np.ndarray:
    """HU -> index into a LUT_SIZE-entry table over CENTER_RANGE."""
    lo, hi = CENTER_RANGE
    scaled = (values - lo) / (hi - lo) * (LUT_SIZE - 1)
    return np.clip(np.rint(scaled), 0, LUT_SIZE - 1).astype(np.uint8)


def lut_values() -> np.ndarray:
    """The HU value each LUT index stands for."""
    return np.linspace(*CENTER_RANGE, LUT_SIZE)


def tissue_lut_labels() -> np.ndarray:
    """LUT index -> tissue index, or -1 for none of them."""
    values = lut_values()
    labels = np.full(LUT_SIZE, -1, dtype=np.int8)
    for index, tissue in enumerate(TISSUES):
        lo, hi = TISSUE_BANDS[tissue]
        labels[(values >= lo) & (values < hi)] = index
    return labels


def transfer_tables(params: np.ndarray):
    """(opacity, luminance) per LUT index for one transfer function."""
    opacity, rgb = _opacity_and_color_at(np.asarray(params, dtype=np.float64), lut_values())
    return opacity.astype(np.float32), (rgb @ LUMINANCE).astype(np.float32)


def _sample_cubes(volume, spacing, directions, n):
    """Resample the volume into one cube per view; axis 0 runs front to back."""
    shape = np.asarray(volume.shape, dtype=np.float64)
    extent = shape * np.asarray(spacing, dtype=np.float64)
    side = float(extent.max())
    step = side / n
    center = extent / 2.0
    offsets = (np.arange(n) - (n - 1) / 2.0) * step
    # Sampling outside the volume must read as air, and grid_sample pads with
    # zeros, so shift the values and shift them back.
    source = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32) - AIR_HU)[None, None]
    cubes = []
    for direction in directions:
        right = np.cross(direction, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, direction)
        depth, vertical, horizontal = np.meshgrid(offsets, offsets, offsets, indexing="ij")
        points = (center + depth[..., None] * direction
                  + vertical[..., None] * up + horizontal[..., None] * right)
        normalized = 2.0 * (points / np.asarray(spacing)) / (shape - 1.0) - 1.0
        grid = torch.from_numpy(np.stack(
            [normalized[..., 2], normalized[..., 1], normalized[..., 0]], -1).astype(np.float32))
        sampled = torch.nn.functional.grid_sample(
            source, grid[None], align_corners=True, padding_mode="zeros")[0, 0] + AIR_HU
        cubes.append(sampled.numpy())
    return np.stack(cubes), step


class VisibilityModel:
    """Per-tissue visibility of a transfer function on one volume."""

    def __init__(self, indices: np.ndarray, step_mm: float, histogram: np.ndarray,
                 volume_id: str = "", spacing=None):
        self.indices = torch.from_numpy(np.ascontiguousarray(indices))
        self.step_mm = float(step_mm)
        self.histogram = np.asarray(histogram, dtype=np.float32)
        self.volume_id = volume_id
        self.spacing = tuple(spacing) if spacing is not None else None
        self._lut_labels = torch.from_numpy(tissue_lut_labels())
        self._labels = self._lut_labels[self.indices.long()]
        self._rays = int(self.indices.shape[0] * self.indices.shape[2] * self.indices.shape[3])
        self._solo_max = {}

    @property
    def n_views(self) -> int:
        return int(self.indices.shape[0])

    @classmethod
    def from_volume(cls, volume, spacing, n=GRID_N, n_views=N_VIEWS, volume_id=""):
        directions = view_directions(n_views)
        cubes, step = _sample_cubes(volume, spacing, directions, n)
        counts, _ = np.histogram(volume, bins=HISTOGRAM_BINS, range=CENTER_RANGE)
        histogram = counts / max(counts.sum(), 1)
        return cls(quantize(cubes), step, histogram, volume_id, spacing)

    def _weights(self, params):
        """(contribution per sample, luminance per sample, accumulated opacity per ray)."""
        opacity_table, luminance_table = transfer_tables(params)
        index = self.indices.long()
        alpha = torch.from_numpy(opacity_table)[index]
        alpha = 1.0 - (1.0 - alpha) ** (self.step_mm / 1.0)
        luminance = torch.from_numpy(luminance_table)[index]
        transparency = 1.0 - alpha
        ones = torch.ones_like(transparency[:, :1])
        before = torch.cat([ones, torch.cumprod(transparency, dim=1)[:, :-1]], dim=1)
        return before * alpha, luminance, 1.0 - transparency.prod(dim=1)

    def features(self, params) -> dict:
        """{"vis": {tissue: float}, "bright": {tissue: float}, "coverage": float}."""
        weights, luminance, accumulated = self._weights(params)
        vis, bright = {}, {}
        for index, tissue in enumerate(TISSUES):
            masked = torch.where(self._labels == index, weights, torch.zeros(()))
            total = float(masked.sum())
            vis[tissue] = total / self._rays
            bright[tissue] = float((masked * luminance).sum() / total) if total > 1e-6 else 0.0
        coverage = float((accumulated >= COVERAGE_THRESHOLD).to(torch.float32).mean())
        return {"vis": vis, "bright": bright, "coverage": coverage}

    def per_view_visibility(self, params, tissue: str) -> list:
        """Visibility of one tissue in each view separately (diagnostics, tests)."""
        weights, _, _ = self._weights(params)
        index = TISSUES.index(tissue)
        masked = torch.where(self._labels == index, weights, torch.zeros(()))
        per_ray = self.indices.shape[2] * self.indices.shape[3]
        return [float(masked[view].sum()) / per_ray for view in range(self.n_views)]

    def solo_max(self, tissue: str) -> float:
        """Visibility of `tissue` when only its own peak is shown, at full opacity.

        The reference for absolute levels ("high opacity bone"): what "fully
        shown" means depends on how much of that tissue this volume contains
        and what sits in front of it.
        """
        if tissue not in self._solo_max:
            from commands import _find_or_create_peak
            from transfer import PARAMS_PER_PEAK, default_params
            params, peak = _find_or_create_peak(default_params().copy(), tissue)
            for other in range(len(TISSUES)):
                params[other * PARAMS_PER_PEAK + 2] = -1.0
            params[peak * PARAMS_PER_PEAK + 2] = 1.0
            self._solo_max[tissue] = self.features(params)["vis"][tissue]
        return self._solo_max[tissue]
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest -q tests/test_visibility.py`
Expected: 10 passed.

- [ ] **Step 5: Check speed and the real occlusion effect**

Run:

```bash
.venv/bin/python -c "
import time, numpy as np, datasets, visibility
from transfer import default_params, PARAMS_PER_PEAK
volume, spacing = datasets.load_dataset('ct_chest', canonical=True)
t = time.time(); model = visibility.VisibilityModel.from_volume(volume, spacing); build = time.time() - t
params = default_params()
t = time.time()
for _ in range(10): features = model.features(params)
print(f'build {build:.2f}s  features {(time.time()-t)/10*1000:.1f} ms')
print('default bone', round(features['vis']['bone'], 4), 'coverage', round(features['coverage'], 3))
cleared = params.copy()
for peak in (0, 1): cleared[peak * PARAMS_PER_PEAK + 2] = -1.0
print('cleared bone', round(model.features(cleared)['vis']['bone'], 4))
"
```

Expected: build ~0.3–0.6 s, features ~15 ms (accept ≤ 30 ms), default bone ≈ 0.002, cleared bone ≈ 0.019 (a rise of at least 5×), coverage ≈ 0.73.

- [ ] **Step 6: Commit**

```bash
git add visibility.py tests/test_visibility.py
git commit -m "feat(rl): per-tissue visibility and brightness estimate"
```

---

### Task 4: Disk cache and build tool

**Files:**
- Modify: `visibility.py`
- Create: `tools/build_visibility_cache.py`
- Test: `tests/test_visibility.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_visibility.py`:

```python
def test_cache_round_trip_reproduces_features(tmp_path, monkeypatch):
    monkeypatch.setattr(visibility, "CACHE_DIR", str(tmp_path))
    volume = _slab_volume(50.0, 900.0)
    params = _params([0.2, 0.5, 0.1, 0.8])
    built = visibility.VisibilityModel.from_volume(volume, (2.0, 2.0, 2.0), volume_id="fake")
    path = built.save_cache("version-1")
    assert os.path.exists(path)
    loaded = visibility.VisibilityModel.load_cache("fake", "version-1")
    assert loaded is not None
    assert loaded.features(params) == built.features(params)
    assert loaded.step_mm == built.step_mm
    assert np.array_equal(loaded.histogram, built.histogram)


def test_cache_miss_on_different_version(tmp_path, monkeypatch):
    monkeypatch.setattr(visibility, "CACHE_DIR", str(tmp_path))
    visibility.VisibilityModel.from_volume(_slab_volume(50.0, 900.0), (2.0, 2.0, 2.0),
                                           volume_id="fake").save_cache("version-1")
    assert visibility.VisibilityModel.load_cache("fake", "version-2") is None
    assert visibility.VisibilityModel.load_cache("other", "version-1") is None


def test_for_volume_builds_once_then_loads(tmp_path, monkeypatch):
    monkeypatch.setattr(visibility, "CACHE_DIR", str(tmp_path))
    calls = []
    real_from_volume = visibility.VisibilityModel.from_volume

    def counting(volume, spacing, **kwargs):
        calls.append(1)
        return real_from_volume(volume, spacing, **kwargs)

    monkeypatch.setattr(visibility.VisibilityModel, "from_volume", staticmethod(counting))
    monkeypatch.setattr(visibility, "_load_volume", lambda name: (_slab_volume(50.0, 900.0), (2.0, 2.0, 2.0)))
    monkeypatch.setattr(visibility, "_volume_version", lambda name: "v1")
    first = visibility.for_volume("fake")
    second = visibility.for_volume("fake")
    assert len(calls) == 1
    assert first.features(_params([0, 0, 0, 0.5])) == second.features(_params([0, 0, 0, 0.5]))
```

Add `import os` to the test file's imports if it is not there.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_visibility.py`
Expected: the three new tests FAIL (`save_cache` / `load_cache` / `for_volume` missing).

- [ ] **Step 3: Implement**

In `visibility.py`, add to the `VisibilityModel` class:

```python
    def cache_key(self, volume_version: str) -> str:
        parts = (self.volume_id, volume_version, self.n_views, self.indices.shape[1],
                 LUT_SIZE, CACHE_VERSION)
        return hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()[:16]

    def save_cache(self, volume_version: str, cache_dir: str = None) -> str:
        cache_dir = cache_dir or CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{self.volume_id}-{self.cache_key(volume_version)}.npz")
        temporary = path + ".tmp.npz"     # np.savez_compressed appends .npz unless it is there
        np.savez_compressed(temporary, indices=self.indices.numpy(),
                            step_mm=np.float64(self.step_mm), histogram=self.histogram,
                            spacing=np.asarray(self.spacing if self.spacing else (0, 0, 0), dtype=np.float64))
        os.replace(temporary, path)
        return path

    @classmethod
    def load_cache(cls, volume_id: str, volume_version: str, cache_dir: str = None,
                   n=GRID_N, n_views=N_VIEWS):
        cache_dir = cache_dir or CACHE_DIR
        probe = cls(np.zeros((n_views, n, n, n), dtype=np.uint8), 1.0,
                    np.zeros(HISTOGRAM_BINS), volume_id)
        path = os.path.join(cache_dir, f"{volume_id}-{probe.cache_key(volume_version)}.npz")
        if not os.path.exists(path):
            return None
        with np.load(path) as data:
            spacing = tuple(float(v) for v in data["spacing"])
            return cls(data["indices"], float(data["step_mm"]), data["histogram"],
                       volume_id, spacing if any(spacing) else None)
```

and at module level, after the class:

```python
def _load_volume(name: str):
    from datasets import load_dataset
    return load_dataset(name, canonical=True)


def _volume_version(name: str) -> str:
    from datasets import _dataset_version
    return _dataset_version(name)


def for_volume(name: str, cache_dir: str = None) -> VisibilityModel:
    """The visibility model of a volume, from the cache when possible."""
    version = _volume_version(name)
    cached = VisibilityModel.load_cache(name, version, cache_dir)
    if cached is not None:
        return cached
    volume, spacing = _load_volume(name)
    model = VisibilityModel.from_volume(volume, spacing, volume_id=name)
    model.save_cache(version, cache_dir)
    return model
```

Create `tools/build_visibility_cache.py`:

```python
"""Precompute visibility caches for the RL v2 volumes.

    python -m tools.build_visibility_cache                  # train + val + test + out_of_source
    python -m tools.build_visibility_cache --splits train
"""
import argparse
import time

import datasets
import visibility

SPLITS = ("train", "val", "test", "out_of_source")


def build(splits=SPLITS, cache_dir=None) -> list:
    built = []
    for split in splits:
        for name in datasets.volumes_for_split(split):
            started = time.time()
            model = visibility.for_volume(name, cache_dir)
            built.append((split, name, time.time() - started))
            print(f"[visibility] {split:13s} {name:12s} "
                  f"step={model.step_mm:5.2f}mm {built[-1][2]:5.2f}s")
    return built


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    built = build(tuple(args.splits), args.cache_dir)
    print(f"[visibility] {len(built)} volumes cached in {sum(b[2] for b in built):.1f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest -q tests/test_visibility.py`
Expected: 13 passed.

- [ ] **Step 5: Build the real cache**

Run: `.venv/bin/python -m tools.build_visibility_cache`
Expected: 34 lines (20 train, 4 val, 6 test, 4 out-of-source), each a few seconds, then a total. Then:

```bash
du -sh out/cache/visibility && ls out/cache/visibility | wc -l
.venv/bin/python -c "
import time, visibility
t = time.time(); model = visibility.for_volume('ct_chest'); print(f'cached load {time.time()-t:.2f}s')
"
```

Expected: 34 files, well under 1 GB total, and a cached load in about a second or less.

- [ ] **Step 6: Commit**

```bash
git add visibility.py tools/build_visibility_cache.py tests/test_visibility.py
git commit -m "feat(rl): cache visibility models per volume"
```

---

### Task 5: Validation gate against VTK

**Files:**
- Create: `tools/validate_visibility.py`
- Test: `tests/test_validate_visibility.py`

The reference for one tissue is the mean rendered luminance of the volume minus the mean rendered luminance of the same volume with that tissue's voxels replaced by air, averaged over the 6 views. The estimate's comparable quantity is `vis[t] * bright[t]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_validate_visibility.py`:

```python
import numpy as np

from tools.validate_visibility import CONTRIBUTION_FLOOR, pearson, summarize


def test_pearson_matches_numpy():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 4.1, 5.9, 8.2])
    assert np.isclose(pearson(a, b), np.corrcoef(a, b)[0, 1])


def test_pearson_is_zero_for_a_constant_series():
    assert pearson(np.ones(5), np.arange(5.0)) == 0.0


def test_summarize_marks_tissues_below_the_noise_floor_unvalidated():
    estimate = {"bone": np.array([0.1, 0.2, 0.3]), "spongy": np.array([0.1, 0.2, 0.3])}
    reference = {"bone": np.array([0.01, 0.02, 0.031]),
                 "spongy": np.array([0.001, 0.001 + CONTRIBUTION_FLOOR / 10, 0.001])}
    summary = summarize(estimate, reference, threshold=0.7)
    assert summary["bone"]["validated"] is True
    assert summary["bone"]["passed"] is True
    assert summary["spongy"]["validated"] is False
    assert "passed" not in summary["spongy"] or summary["spongy"]["passed"] is None


def test_summarize_fails_an_uncorrelated_tissue():
    estimate = {"bone": np.array([0.1, 0.2, 0.3, 0.4])}
    reference = {"bone": np.array([0.05, 0.01, 0.06, 0.02])}
    summary = summarize(estimate, reference, threshold=0.7)
    assert summary["bone"]["validated"] is True
    assert summary["bone"]["passed"] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_validate_visibility.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.validate_visibility'`.

- [ ] **Step 3: Implement**

Create `tools/validate_visibility.py`:

```python
"""Does the visibility estimate track real renders?

For each of several random transfer functions, render the volume from the 6
standard views, then render it again with one tissue's voxels replaced by air.
The drop in mean luminance is that tissue's real contribution to the picture.
Compare that against the estimate's vis * bright for the same tissue.

A tissue whose real contribution barely moves across the sampled transfer
functions cannot be validated this way (spongy bone, for instance, sits inside
cortical bone and stays near the renderer's noise floor); such tissues are
reported as unvalidated rather than counted as failures.

    python -m tools.validate_visibility ct_chest ts_s1337 --trials 12
"""
import argparse
import json
import os

import numpy as np

import datasets
import render
import views
import visibility
from transfer import CENTER_RANGE, PARAMS_PER_PEAK, TISSUE_BANDS, default_params

RENDER_SIZE = 192
CONTRIBUTION_FLOOR = 0.002      # luminance range below this is renderer noise
DEFAULT_THRESHOLD = 0.7         # Pearson correlation required per validated tissue
COVERAGE_THRESHOLD = 0.85       # Pearson correlation required for coverage
OUT_PATH = "out/visibility_validation.json"


def pearson(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    return float((a * b).sum() / denominator) if denominator > 0 else 0.0


def summarize(estimate: dict, reference: dict, threshold=DEFAULT_THRESHOLD) -> dict:
    summary = {}
    for tissue, values in reference.items():
        values = np.asarray(values, dtype=np.float64)
        spread = float(values.max() - values.min())
        entry = {"contribution_range": spread, "validated": spread >= CONTRIBUTION_FLOOR}
        if entry["validated"]:
            entry["pearson"] = pearson(estimate[tissue], values)
            entry["passed"] = entry["pearson"] >= threshold
        else:
            entry["pearson"] = pearson(estimate[tissue], values)
            entry["passed"] = None
        summary[tissue] = entry
    return summary


def random_params(rng) -> np.ndarray:
    params = default_params().copy()
    for peak in range(4):
        params[peak * PARAMS_PER_PEAK + 2] = np.clip(
            params[peak * PARAMS_PER_PEAK + 2] + rng.uniform(-1.0, 1.0), -1.0, 1.0)
        params[peak * PARAMS_PER_PEAK + 1] = np.clip(
            params[peak * PARAMS_PER_PEAK + 1] + rng.uniform(-0.5, 0.5), -1.0, 1.0)
    return params


def _render_mean_luminance(volume, spacing, params, cameras) -> float:
    original = (render.WIDTH, render.HEIGHT)
    render.WIDTH, render.HEIGHT = RENDER_SIZE, RENDER_SIZE
    try:
        frames = [render.grab(render.render(volume, params, spacing, camera)) for camera in cameras]
    finally:
        render.WIDTH, render.HEIGHT = original
    return float(np.mean([frame.astype(np.float64).mean() / 255.0 for frame in frames]))


def _rendered_coverage(volume, spacing, params, cameras) -> float:
    original = (render.WIDTH, render.HEIGHT)
    render.WIDTH, render.HEIGHT = RENDER_SIZE, RENDER_SIZE
    try:
        frames = [render.grab(render.render(volume, params, spacing, camera)) for camera in cameras]
    finally:
        render.WIDTH, render.HEIGHT = original
    return float(np.mean([(frame.astype(np.float64).mean(axis=2) > 2).mean() for frame in frames]))


def validate_volume(name: str, trials: int = 12, seed: int = 0, threshold=DEFAULT_THRESHOLD) -> dict:
    volume, spacing = datasets.load_dataset(name, canonical=True)
    cameras = views.cameras_for_volume(volume, spacing)
    model = visibility.for_volume(name)
    rng = np.random.default_rng(seed)
    sampled = [random_params(rng) for _ in range(trials)]
    features = [model.features(params) for params in sampled]

    estimate = {t: np.array([f["vis"][t] * f["bright"][t] for f in features]) for t in visibility.TISSUES}
    full = np.array([_render_mean_luminance(volume, spacing, params, cameras) for params in sampled])
    reference = {}
    for tissue in visibility.TISSUES:
        lo, hi = TISSUE_BANDS[tissue]
        without_tissue = np.where((volume >= lo) & (volume < hi),
                                  np.float32(CENTER_RANGE[0]), volume)
        removed = np.array([_render_mean_luminance(without_tissue, spacing, params, cameras)
                            for params in sampled])
        reference[tissue] = full - removed

    coverage_estimate = np.array([f["coverage"] for f in features])
    coverage_reference = np.array([_rendered_coverage(volume, spacing, params, cameras)
                                   for params in sampled])
    coverage = {"pearson": pearson(coverage_estimate, coverage_reference)}
    coverage["passed"] = coverage["pearson"] >= COVERAGE_THRESHOLD
    return {"volume": name, "trials": trials, "seed": seed,
            "tissues": summarize(estimate, reference, threshold), "coverage": coverage}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    reports, failures = [], []
    for name in args.names:
        report = validate_volume(name, args.trials, args.seed, args.threshold)
        reports.append(report)
        print(f"{name}: coverage pearson={report['coverage']['pearson']:+.3f} "
              f"{'PASS' if report['coverage']['passed'] else 'FAIL'}")
        if not report["coverage"]["passed"]:
            failures.append(f"{name}: coverage")
        for tissue, entry in report["tissues"].items():
            state = "unvalidated" if not entry["validated"] else ("PASS" if entry["passed"] else "FAIL")
            print(f"    {tissue:7s} pearson={entry['pearson']:+.3f} "
                  f"range={entry['contribution_range']:.4f} {state}")
            if entry["validated"] and not entry["passed"]:
                failures.append(f"{name}: {tissue}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as stream:
        json.dump({"threshold": args.threshold, "coverage_threshold": COVERAGE_THRESHOLD,
                   "contribution_floor": CONTRIBUTION_FLOOR, "reports": reports}, stream, indent=2)
    print(f"[validate_visibility] wrote {args.out}")
    if failures:
        raise SystemExit("FAILED: " + ", ".join(failures))
    print("[validate_visibility] gate passed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest -q tests/test_validate_visibility.py`
Expected: 4 passed.

- [ ] **Step 5: Run the gate**

Run: `.venv/bin/python -m tools.validate_visibility ct_chest ts_s1337 ts_s0454 --trials 12`

Expected (based on the controller's prototype): coverage PASS on all three; bone, fat and soft PASS (Pearson roughly 0.8–0.95); spongy reported `unvalidated` on at least ct_chest. Takes several minutes (each trial renders 6 views of 5 volume variants).

**This is a gate.** If a validated tissue FAILS on any volume, STOP: do not weaken the threshold, do not change the estimate to chase the number. Report the JSON and let the controller decide.

- [ ] **Step 6: Commit**

```bash
git add tools/validate_visibility.py tests/test_validate_visibility.py
git commit -m "feat(rl): validate the visibility estimate against VTK renders"
```

---

### Task 6: Document the estimate

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a README subsection**

In `README.md`, directly after the "### RL v2 volumes" subsection (before `## Local 3D Viewer`), insert, filling the four Pearson values and the spongy note from `out/visibility_validation.json`:

````markdown
### Visibility estimate

RL v2 scores a transfer function by how much of each tissue it actually shows.
`visibility.py` resamples every volume once per view (6 views, `views.py`),
caches a quantized copy under `out/cache/visibility/`, and composites it
front to back: per tissue it reports visibility (share of the picture that
tissue contributes) and brightness, plus overall coverage — about 15 ms per
call, against ~50–200 ms for a real render.

```bash
python -m tools.build_visibility_cache          # once per machine, ~34 volumes
python -m tools.validate_visibility ct_chest ts_s1337 ts_s0454
```

The estimate tracks real VTK renders: correlations of <!-- fill in --> for
bone, fat and soft tissue, and <!-- fill in --> for coverage, measured against
renders with one tissue removed from the volume (`out/visibility_validation.json`).
Spongy bone cannot be validated this way — it sits inside cortical bone, so its
contribution to the image stays near the renderer's noise floor — and is
reported as unvalidated.
````

Replace both `<!-- fill in -->` markers with the measured ranges (e.g. "0.81–0.94"). Do not leave the markers in.

- [ ] **Step 2: Full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: describe the RL v2 visibility estimate"
```
