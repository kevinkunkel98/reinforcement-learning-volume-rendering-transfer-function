import math
import numpy as np
from transfer import (
    default_params, opacity_mass, mass_fraction, vector_to_vtk,
    TISSUE_BANDS, PARAMS_PER_PEAK, N_PEAKS,
)


def test_default_params_shape():
    p = default_params()
    assert p.shape == (N_PEAKS * PARAMS_PER_PEAK,)
    assert np.all(p >= -1.0) and np.all(p <= 1.0)


def test_opacity_mass_single_gaussian_matches_analytic():
    params = np.full(N_PEAKS * PARAMS_PER_PEAK, -1.0)
    lo, hi = -1050.0, 2000.0
    center_ext = 2.0 * (40.0 - lo) / (hi - lo) - 1.0
    width_ext = 2.0 * (40.0 - 10.0) / (400.0 - 10.0) - 1.0
    height_ext = 2.0 * 0.2 - 1.0
    params[0:6] = [center_ext, width_ext, height_ext, -1.0, -1.0, -1.0]
    mass = opacity_mass(params, -1050.0, 2000.0)
    analytic = 0.2 * 40.0 * math.sqrt(2 * math.pi)
    assert abs(mass - analytic) / analytic < 0.05


def test_opacity_mass_zero_for_all_zero_height():
    params = np.full(N_PEAKS * PARAMS_PER_PEAK, -1.0)
    mass = opacity_mass(params, *TISSUE_BANDS["bone"])
    assert mass < 1e-6


def test_vector_to_vtk_returns_expected_types():
    import vtk
    ctf, otf = vector_to_vtk(default_params())
    assert isinstance(ctf, vtk.vtkColorTransferFunction)
    assert isinstance(otf, vtk.vtkPiecewiseFunction)


def test_mass_fraction_is_opacity_mass_normalized_by_band_width():
    params = np.full(N_PEAKS * PARAMS_PER_PEAK, -1.0)
    lo, hi = TISSUE_BANDS["bone"]
    params[N_PEAKS * PARAMS_PER_PEAK - PARAMS_PER_PEAK + 2] = 2.0 * 0.5 - 1.0  # bone peak height = 0.5
    frac = mass_fraction(params, "bone")
    assert 0.0 <= frac <= 1.0
    assert abs(frac - opacity_mass(params, lo, hi) / (hi - lo)) < 1e-9


def test_mass_fraction_zero_for_all_zero_height():
    params = np.full(N_PEAKS * PARAMS_PER_PEAK, -1.0)
    assert mass_fraction(params, "spongy") < 1e-6
