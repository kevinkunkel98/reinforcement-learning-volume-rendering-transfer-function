import numpy as np
import pytest

import hue
import goals
import transfer


def test_luminance_weights_are_the_visibility_models():
    import visibility
    np.testing.assert_array_equal(hue.LUMINANCE, visibility.LUMINANCE)


def test_darkening_floor_is_a_fraction_of_the_reference_luminance():
    liver = np.array(transfer.ANATOMICAL_COLOURS["liver"])
    assert hue.darkening_floor(liver, liver) == pytest.approx(
        hue.MIN_LUMINANCE_FRACTION * hue.luminance(liver))


def test_darkening_floor_never_brightens_a_colour_already_below_it():
    liver = np.array(transfer.ANATOMICAL_COLOURS["liver"])
    dark = hue.scale_hue(liver, 0.02)
    assert hue.darkening_floor(liver, dark) == pytest.approx(0.02)


def test_hue_source_is_the_colour_unless_it_is_near_black():
    liver = np.array(transfer.ANATOMICAL_COLOURS["liver"])
    grey = np.array([0.4, 0.4, 0.4])
    np.testing.assert_array_equal(hue.hue_source(grey, liver), grey)
    np.testing.assert_array_equal(hue.hue_source(np.zeros(3), liver), liver)
