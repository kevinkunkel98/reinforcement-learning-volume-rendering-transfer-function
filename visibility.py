"""How much of each anatomical class a transfer function actually shows.

Direct volume rendering is expensive to run inside an RL loop, and a rendered
image does not say which tissue produced which pixel. This module estimates
both cheaply: it resamples a volume once per view into a cube whose first axis
runs front to back, then composites that cube front to back for a given
transfer function. The result is, per class, how much of the final image that
class contributes (`vis`) and how bright it appears (`bright`), plus how much
of the frame is covered at all (`coverage`).

Classes come from the TotalSegmentator label volume (`totalseg.load_labels`)
when one exists for the volume; `label_source` reports "anatomy" in that
case. The four Slicer CTs and the synthetic phantom carry no such labels, so
their samples fall back to coarse Hounsfield bands mapped onto the same class
names (`label_source` is then "intensity"), which only separate skeleton,
lungs and organs -- muscle and vessels are never populated by the fallback.

It is an estimate, not a renderer: no shading, no perspective, a coarse grid.
tools/validate_visibility.py measures how well it tracks real VTK renders.
"""
import collections
import hashlib
import os

import numpy as np
import torch

from anatomy import CLASS_LAYOUT_VERSION, MEASURED_CLASSES
from transfer import CENTER_RANGE, N_PEAKS, PARAMS_PER_PEAK, anatomical_params, _opacity_and_color_at
from views import N_VIEWS, view_directions

CLASSES = MEASURED_CLASSES[:-1]
GRID_N = 80                     # samples per cube axis; 15 ms/features() on CPU
LUT_SIZE = 256                  # quantization levels over CENTER_RANGE
HISTOGRAM_BINS = 16
COVERAGE_THRESHOLD = 0.3        # accumulated opacity for a ray to count as covered
AIR_HU = CENTER_RANGE[0]
LUMINANCE = np.array([0.2126, 0.7152, 0.0722])
CACHE_DIR = "out/cache/visibility"
CACHE_VERSION = 3


def quantize(values: np.ndarray) -> np.ndarray:
    """HU -> index into a LUT_SIZE-entry table over CENTER_RANGE."""
    lo, hi = CENTER_RANGE
    scaled = (values - lo) / (hi - lo) * (LUT_SIZE - 1)
    return np.clip(np.rint(scaled), 0, LUT_SIZE - 1).astype(np.uint8)


def lut_values() -> np.ndarray:
    """The HU value each LUT index stands for."""
    return np.linspace(*CENTER_RANGE, LUT_SIZE)


def _intensity_class_ids(hu: np.ndarray) -> np.ndarray:
    """Class ids from Hounsfield bands, for volumes with no anatomical label
    volume. Only skeleton/lungs/organs are separable by intensity alone --
    only skeleton, lungs and soft are populated; all promoted organs and
    vessels stay empty because intensity alone cannot identify them."""
    ids = np.zeros(hu.shape, dtype=np.uint8)
    ids[hu <= -500.0] = CLASSES.index("lungs") + 1
    ids[(hu >= -30.0) & (hu < 300.0)] = CLASSES.index("soft") + 1
    ids[hu >= 300.0] = CLASSES.index("skeleton") + 1
    return ids


def transfer_tables(params: np.ndarray):
    """(opacity, luminance) per LUT index for one transfer function."""
    opacity, rgb = _opacity_and_color_at(np.asarray(params, dtype=np.float64), lut_values())
    return opacity.astype(np.float32), (rgb @ LUMINANCE).astype(np.float32)


def _sample_cubes(volume, spacing, directions, n, labels=None):
    """Resample the volume (and, if given, a label volume) into one cube per
    view; axis 0 runs front to back. Returns (hu_cubes, step_mm, label_cubes
    or None)."""
    shape = np.asarray(volume.shape, dtype=np.float64)
    extent = shape * np.asarray(spacing, dtype=np.float64)
    side = float(extent.max())
    step = side / n
    center = extent / 2.0
    offsets = (np.arange(n) - (n - 1) / 2.0) * step
    # Sampling outside the volume must read as air, and grid_sample pads with
    # zeros, so shift the values and shift them back.
    source = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32) - AIR_HU)[None, None]
    label_source = None
    if labels is not None:
        # No shift needed here: grid_sample's zero padding already means "no
        # class" (id 0, "other"), exactly the sentinel the label volume uses.
        label_source = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.float32))[None, None]
    cubes, label_cubes = [], []
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
        # Nearest, not interpolated: at a ~4 mm step, interpolating across a
        # tissue boundary invents values (an HU reading between air and bone,
        # a label between two classes) that fall into a class the volume does
        # not actually contain there.
        sampled = torch.nn.functional.grid_sample(
            source, grid[None], mode="nearest", align_corners=True,
            padding_mode="zeros")[0, 0] + AIR_HU
        cubes.append(sampled.numpy())
        if label_source is not None:
            sampled_labels = torch.nn.functional.grid_sample(
                label_source, grid[None], mode="nearest", align_corners=True,
                padding_mode="zeros")[0, 0]
            label_cubes.append(sampled_labels.round().numpy().astype(np.uint8))
    stacked_labels = np.stack(label_cubes) if label_cubes else None
    return np.stack(cubes), step, stacked_labels


class VisibilityModel:
    """Per-class visibility of a transfer function on one volume."""

    def __init__(self, indices: np.ndarray, step_mm: float, histogram: np.ndarray,
                 class_ids: np.ndarray, label_source: str,
                 volume_id: str = "", spacing=None):
        self.indices = torch.from_numpy(np.ascontiguousarray(indices))
        self.step_mm = float(step_mm)
        self.histogram = np.asarray(histogram, dtype=np.float32)
        self.class_ids = np.ascontiguousarray(class_ids, dtype=np.uint8)
        self.label_source = label_source
        self.volume_id = volume_id
        self.spacing = tuple(spacing) if spacing is not None else None
        # class id 0 ("other") -> -1 (matches no class); id i -> CLASSES[i - 1]
        self._labels = torch.from_numpy(self.class_ids.astype(np.int32) - 1)
        self._rays = int(self.indices.shape[0] * self.indices.shape[2] * self.indices.shape[3])
        self._solo_max = {}

    @property
    def n_views(self) -> int:
        return int(self.indices.shape[0])

    @classmethod
    def from_volume(cls, volume, spacing, labels=None, n=GRID_N, n_views=N_VIEWS, volume_id=""):
        directions = view_directions(n_views)
        cubes, step, label_cubes = _sample_cubes(volume, spacing, directions, n, labels)
        counts, _ = np.histogram(volume, bins=HISTOGRAM_BINS, range=CENTER_RANGE)
        histogram = counts / max(counts.sum(), 1)
        if label_cubes is not None:
            class_ids, label_source = label_cubes, "anatomy"
        else:
            class_ids, label_source = _intensity_class_ids(cubes), "intensity"
        return cls(quantize(cubes), step, histogram, class_ids, label_source, volume_id, spacing)

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
        """{"vis": {class: float, "other": float}, "bright": {class: float}, "coverage": float}.

        "other" is voxels the label volume (or the intensity fallback)
        assigned to none of the five classes -- fat, connective tissue,
        partial-volume edges. A transfer function can still put opacity
        there, so it is measured the same way the five classes are, instead
        of silently vanishing: `goals.distance` charges for it under the same
        keep-tolerance any unmentioned class gets, or a search/policy could
        satisfy "show only X" by rendering an opaque wall of unclassified
        material instead of X (see goals.py's OTHER keep term)."""
        weights, luminance, accumulated = self._weights(params)
        vis, bright = {}, {}
        for index, name in enumerate(CLASSES):
            masked = torch.where(self._labels == index, weights, torch.zeros(()))
            total = float(masked.sum())
            vis[name] = total / self._rays
            bright[name] = (float((masked * luminance).sum() / total)
                            if total / self._rays > 1e-4 else 0.0)
        other_masked = torch.where(self._labels == -1, weights, torch.zeros(()))
        vis["other"] = float(other_masked.sum()) / self._rays
        coverage = float((accumulated >= COVERAGE_THRESHOLD).to(torch.float32).mean())
        return {"vis": vis, "bright": bright, "coverage": coverage}

    def per_view_visibility(self, params, name: str) -> list:
        """Visibility of one class in each view separately (diagnostics, tests)."""
        weights, _, _ = self._weights(params)
        index = CLASSES.index(name)
        masked = torch.where(self._labels == index, weights, torch.zeros(()))
        per_ray = self.indices.shape[2] * self.indices.shape[3]
        return [float(masked[view].sum()) / per_ray for view in range(self.n_views)]

    def solo_max(self, name: str) -> float:
        """Largest vis_c over the four single-peak transfer functions (peak i
        at height 1, the rest at 0) -- the reference for absolute levels
        ("fully shown"): what that means depends on how much of that class
        this volume contains and what sits in front of it.
        """
        if name not in self._solo_max:
            best = 0.0
            for peak in range(N_PEAKS):
                params = anatomical_params().copy()
                for i in range(N_PEAKS):
                    params[i * PARAMS_PER_PEAK + 2] = 1.0 if i == peak else -1.0
                best = max(best, self.features(params)["vis"][name])
            self._solo_max[name] = best
        return self._solo_max[name]

    def cache_key(self, volume_version: str) -> str:
        parts = (self.volume_id, volume_version, self.n_views, self.indices.shape[1],
                 LUT_SIZE, CACHE_VERSION, CLASS_LAYOUT_VERSION, ",".join(CLASSES))
        return hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()[:16]

    def save_cache(self, volume_version: str, cache_dir: str = None) -> str:
        cache_dir = cache_dir or CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{self.volume_id}-{self.cache_key(volume_version)}.npz")
        temporary = path + ".tmp.npz"     # np.savez_compressed appends .npz unless it is there
        np.savez_compressed(temporary, indices=self.indices.numpy(),
                            step_mm=np.float64(self.step_mm), histogram=self.histogram,
                            class_ids=self.class_ids, label_source=np.array(self.label_source),
                            spacing=np.asarray(self.spacing if self.spacing else (0, 0, 0), dtype=np.float64))
        os.replace(temporary, path)
        return path

    @classmethod
    def load_cache(cls, volume_id: str, volume_version: str, cache_dir: str = None,
                   n=GRID_N, n_views=N_VIEWS):
        cache_dir = cache_dir or CACHE_DIR
        probe = cls(np.zeros((n_views, n, n, n), dtype=np.uint8), 1.0,
                    np.zeros(HISTOGRAM_BINS), np.zeros((n_views, n, n, n), dtype=np.uint8),
                    "intensity", volume_id)
        path = os.path.join(cache_dir, f"{volume_id}-{probe.cache_key(volume_version)}.npz")
        if not os.path.exists(path):
            return None
        with np.load(path) as data:
            spacing = tuple(float(v) for v in data["spacing"])
            return cls(data["indices"], float(data["step_mm"]), data["histogram"],
                       data["class_ids"], data["label_source"].item(),
                       volume_id, spacing if any(spacing) else None)


def _load_volume(name: str):
    from datasets import load_dataset
    return load_dataset(name, canonical=True)


def _volume_version(name: str) -> str:
    from datasets import _dataset_version
    return _dataset_version(name)


def _has_labels(name: str) -> bool:
    import totalseg
    return totalseg.is_totalseg(name) and totalseg.has_labels(name)


def _load_labels(name: str):
    import totalseg
    return totalseg.load_labels(name)


# Keep a few built models in memory, most-recently-used last.
#
# `solo_max` memoises its four single-peak probes on the instance (~190 ms the
# first time), but returning a fresh instance per call threw that memo away on
# every call -- and `for_volume` is called on every render for the viewer's
# readout and on every policy answer. Reusing the instance is what makes the
# memo do its job.
#
# Bounded because training sweeps every volume in a split and a model is about
# 6 MB (two 6x80x80x80 arrays); MODEL_CACHE_SIZE keeps the working set for a
# viewer session (one volume, occasionally switched) without pinning a whole
# split. The key carries the volume version and the resolved cache directory,
# so a model never outlives the data it describes.
MODEL_CACHE_SIZE = 4
_MODEL_CACHE = collections.OrderedDict()


def clear_model_cache() -> None:
    """Drop every in-memory model. For tests that patch what a volume is."""
    _MODEL_CACHE.clear()


def for_volume(name: str, cache_dir: str = None) -> VisibilityModel:
    """The visibility model of a volume, from memory or the on-disk cache when
    possible.

    Returns a *shared* instance: callers must treat it as read-only. Nothing
    assigns to a model after `__init__` except `solo_max`'s own memo, which is
    what the sharing exists to preserve."""
    version = _volume_version(name)
    key = (name, version, os.path.abspath(cache_dir or CACHE_DIR))
    if key in _MODEL_CACHE:
        _MODEL_CACHE.move_to_end(key)
        return _MODEL_CACHE[key]

    model = VisibilityModel.load_cache(name, version, cache_dir)
    if model is None:
        volume, spacing = _load_volume(name)
        labels = _load_labels(name) if _has_labels(name) else None
        model = VisibilityModel.from_volume(volume, spacing, labels=labels, volume_id=name)
        model.save_cache(version, cache_dir)

    _MODEL_CACHE[key] = model
    _MODEL_CACHE.move_to_end(key)
    while len(_MODEL_CACHE) > MODEL_CACHE_SIZE:
        _MODEL_CACHE.popitem(last=False)
    return model
