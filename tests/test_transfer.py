import math
import numpy as np
import pytest
from transfer import (
    default_params, opacity_mass, mass_fraction, vector_to_vtk,
    anatomical_params, peak_internal, TISSUE_BANDS, PARAMS_PER_PEAK, N_PEAKS,
    ANATOMICAL_PEAK_INDEX, ANATOMICAL_CENTRES_HU, ANATOMICAL_WIDTHS_HU,
    ANATOMICAL_HEIGHTS, ANATOMICAL_COLOURS,
)
from anatomy import CANONICAL_PEAK_ORDER


def test_anatomical_layout_uses_canonical_eight_peak_order_and_calibration():
    assert N_PEAKS == 8
    assert PARAMS_PER_PEAK == 6
    assert tuple(ANATOMICAL_PEAK_INDEX) == CANONICAL_PEAK_ORDER
    assert tuple(ANATOMICAL_PEAK_INDEX.values()) == tuple(range(N_PEAKS))
    assert set(ANATOMICAL_CENTRES_HU) == set(CANONICAL_PEAK_ORDER)
    assert set(ANATOMICAL_WIDTHS_HU) == set(CANONICAL_PEAK_ORDER)
    assert set(ANATOMICAL_HEIGHTS) == set(CANONICAL_PEAK_ORDER)
    assert set(ANATOMICAL_COLOURS) == set(CANONICAL_PEAK_ORDER)

    params = anatomical_params()
    assert params.shape == (48,)
    for name in CANONICAL_PEAK_ORDER:
        peak = peak_internal(params, ANATOMICAL_PEAK_INDEX[name])
        assert peak["center"] == pytest.approx(ANATOMICAL_CENTRES_HU[name])
        assert peak["width"] == pytest.approx(ANATOMICAL_WIDTHS_HU[name])
        assert peak["height"] == pytest.approx(ANATOMICAL_HEIGHTS[name])
        assert peak["rgb"] == pytest.approx(ANATOMICAL_COLOURS[name])


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
