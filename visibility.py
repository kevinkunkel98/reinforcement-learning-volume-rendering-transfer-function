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
        # Nearest, not interpolated: at a ~4 mm step, interpolating across a
        # tissue boundary invents HU values that fall into bands the volume
        # does not contain (a soft-tissue reading between air and bone), which
        # would then be counted as that tissue being visible.
        sampled = torch.nn.functional.grid_sample(
            source, grid[None], mode="nearest", align_corners=True,
            padding_mode="zeros")[0, 0] + AIR_HU
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
