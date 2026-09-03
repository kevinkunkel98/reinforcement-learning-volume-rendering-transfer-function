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
        ("bone", 280.0, 0.60, (0.95, 0.95, 0.90)),
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
    return float(np.trapezoid(opacity, hu))


def vector_to_vtk(params: np.ndarray, n: int = 256):
    hu = np.linspace(*CENTER_RANGE, n)
    opacity, rgb = _opacity_and_color_at(params, hu)

    ctf = vtk.vtkColorTransferFunction()
    otf = vtk.vtkPiecewiseFunction()
    for i in range(n):
        ctf.AddRGBPoint(float(hu[i]), *[float(v) for v in rgb[i]])
        otf.AddPoint(float(hu[i]), float(opacity[i]))
    return ctf, otf
