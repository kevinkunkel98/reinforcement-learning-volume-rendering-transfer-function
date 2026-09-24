"""48-float transfer-function vector <-> VTK objects, opacity_mass metric."""
import numpy as np
import vtk
from anatomy import CANONICAL_PEAK_ORDER

N_PEAKS = len(CANONICAL_PEAK_ORDER)
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


# One peak per canonical goal class, in the shared anatomy order. Centres stay
# fixed; the policy changes each peak's height, width and colour.
ANATOMICAL_PEAK_INDEX = {name: i for i, name in enumerate(CANONICAL_PEAK_ORDER)}
ANATOMICAL_CENTRES_HU = {
    "lungs": -800.0,
    "soft": 40.0,
    "liver": 65.0,
    "kidneys": 45.0,
    "spleen": 55.0,
    "heart": 50.0,
    "vessels": 300.0,
    "skeleton": 900.0,
}
ANATOMICAL_WIDTHS_HU = {
    "lungs": 60.0,
    "soft": 80.0,
    "liver": 45.0,
    "kidneys": 45.0,
    "spleen": 45.0,
    "heart": 50.0,
    "vessels": 80.0,
    "skeleton": 280.0,
}

# "Show only X" sets X's height high and everything else's height to zero, but
# a wide peak (skeleton's default is 280 HU) still has real opacity reaching
# into a neighbour's HU range -- boosting it lights up organ/muscle voxels and
# unlabeled ("other") tissue too, even though their own peaks are zeroed. On
# ts_s0454, capping skeleton's width to 150 HU cut soft-tissue haze 124x
# (3.98% -> 0.03%) and unlabeled bleed 28x (22.3% -> 0.8%) while costing only
# 12% of skeleton's own visibility (11.1% -> 9.8%) -- narrower than 150
# started cutting into skeleton itself for diminishing haze reduction. Applied
# as a cap (`min(current, 150)`), never a widening.
SHOW_ONLY_MAX_WIDTH_HU = 150.0
ANATOMICAL_HEIGHTS = {
    "lungs": 0.05,
    "soft": 0.15,
    "liver": 0.22,
    "kidneys": 0.24,
    "spleen": 0.20,
    "heart": 0.25,
    "vessels": 0.30,
    "skeleton": 0.60,
}
ANATOMICAL_COLOURS = {
    "lungs": (0.55, 0.70, 0.95),
    "soft": (0.85, 0.35, 0.35),
    "liver": (0.75, 0.30, 0.20),
    "kidneys": (0.90, 0.50, 0.40),
    "spleen": (0.65, 0.25, 0.30),
    "heart": (0.95, 0.20, 0.20),
    "vessels": (0.90, 0.45, 0.40),
    "skeleton": (0.95, 0.95, 0.90),
}


def anatomical_params() -> np.ndarray:
    """One peak per goal class at its tissue's Hounsfield value."""
    params = np.zeros(TOTAL_PARAMS, dtype=np.float64)
    for goal_class, index in ANATOMICAL_PEAK_INDEX.items():
        base = index * PARAMS_PER_PEAK
        params[base + 0] = _from_range(ANATOMICAL_CENTRES_HU[goal_class], *CENTER_RANGE)
        params[base + 1] = _from_range(ANATOMICAL_WIDTHS_HU[goal_class], *WIDTH_RANGE)
        params[base + 2] = _from_unit(ANATOMICAL_HEIGHTS[goal_class])
        r, g, b = ANATOMICAL_COLOURS[goal_class]
        params[base + 3] = _from_unit(r)
        params[base + 4] = _from_unit(g)
        params[base + 5] = _from_unit(b)
    return params


def default_params() -> np.ndarray:
    """The transfer function everything starts from: the anatomical layout.

    This used to be a separate band layout (peaks at fat/soft/spongy/bone),
    which left the viewer starting from a different geometry than the policy
    was trained in. Since the peak *centres are fixed* -- no command and no
    policy action moves them -- a viewer session seeded with the band layout
    gave peak 0 a centre at fat (-100 HU) while the policy, trained on
    `anatomical_params`, treats peak 0 as lungs (-800 HU). "More lungs" in
    policy mode then adjusted the fat peak and could not touch the lungs at
    all. The parser retired fat/air/spongy as tissue names for the same
    reason, so there is no longer anything the band layout is for.
    """
    return anatomical_params()


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
    return float(np.trapezoid(opacity, hu))


def mass_fraction(params: np.ndarray, tissue: str) -> float:
    """opacity_mass normalized by band width -- an average-opacity-in-band
    fraction in [0, 1], used as the RL observation/reward signal since raw
    opacity_mass scales with band width (bone's band alone is 1400 HU wide)."""
    lo, hi = TISSUE_BANDS[tissue]
    return opacity_mass(params, lo, hi) / (hi - lo)


def vector_to_vtk(params: np.ndarray, n: int = 256):
    hu = np.linspace(*CENTER_RANGE, n)
    opacity, rgb = _opacity_and_color_at(params, hu)

    ctf = vtk.vtkColorTransferFunction()
    otf = vtk.vtkPiecewiseFunction()
    for i in range(n):
        ctf.AddRGBPoint(float(hu[i]), *[float(v) for v in rgb[i]])
        otf.AddPoint(float(hu[i]), float(opacity[i]))
    return ctf, otf
