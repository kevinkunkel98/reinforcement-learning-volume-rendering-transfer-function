import math
import numpy as np
from transfer import (
    default_params, opacity_mass, vector_to_vtk,
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
