"""Synthetic CT phantom in Hounsfield units."""
import numpy as np

SIZE_DEFAULT = 96
SEED = 42

HU = {
    "air": -1000.0,
    "fat": -100.0,
    "soft": 40.0,
    "spongy": 300.0,
    "bone": 900.0,
}


def _soft_blob(xx, yy, zz, cx, cy, cz, radius, value):
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
    exponent = np.clip((d - radius) / (radius * 0.08 + 1e-6), -50.0, 50.0)
    edge = 1.0 / (1.0 + np.exp(exponent))
    return edge * value


def build_phantom(size: int = SIZE_DEFAULT, seed: int = SEED) -> np.ndarray:
    """Layered, mutually-occluding tissue blobs in HU, fixed noise."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size].astype(np.float32)
    center = size / 2.0

    vol = np.full((size, size, size), HU["air"], dtype=np.float32)

    torso = _soft_blob(xx, yy, zz, center, center, center,
                        size * 0.34, HU["soft"] - HU["air"])
    vol += torso

    fat_mask = torso > (HU["soft"] - HU["air"]) * 0.3
    fat = _soft_blob(xx, yy, zz, center, center, center,
                      size * 0.30, HU["fat"] - HU["soft"]) * fat_mask
    vol += fat

    bx, by, bz = center * 1.15, center * 0.9, center
    spongy = _soft_blob(xx, yy, zz, bx, by, bz,
                         size * 0.16, HU["spongy"] - HU["soft"])
    vol += spongy

    bone = _soft_blob(xx, yy, zz, bx, by, bz,
                       size * 0.10, HU["bone"] - HU["spongy"])
    vol += bone

    noise = rng.normal(0.0, 15.0, vol.shape).astype(np.float32)
    vol += noise
    return np.clip(vol, -1050.0, 2000.0).astype(np.float32)
