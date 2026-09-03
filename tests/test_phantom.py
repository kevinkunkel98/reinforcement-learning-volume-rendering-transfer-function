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
    assert vol.min() < HU["fat"]
    assert vol.max() > HU["spongy"]
    assert vol.min() >= -1050.0
    assert vol.max() <= 2000.0
