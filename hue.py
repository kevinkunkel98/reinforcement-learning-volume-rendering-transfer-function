"""Colour helpers shared by the exact command path (`commands.apply_command`)
and the policy path's colour guard (`rl.colour_guard`): luminance, a
hue-preserving rescale, and the floor on a requested darkening.

Numpy only, so `commands` stays free of the visibility model's torch import;
`LUMINANCE` is `visibility.LUMINANCE` (Rec. 709), checked by the tests.
"""
import numpy as np

LUMINANCE = np.array([0.2126, 0.7152, 0.0722])
# Same hue: the angle between the colour and the reference colour in RGB is
# at most ~18 deg. Red (0.95, 0.2, 0.2) against white is 38 deg, against its
# start-noise shifts (rgb +-0.15, rl.oneshot_env.START_NOISE) <= 10 deg; liver
# (0.75, 0.3, 0.2) fails it about half-way to white, at (1, 0.75, 0.65).
HUE_MIN_COSINE = 0.95
# Below this luminance a colour renders black and has no hue to compare.
DARK_LUMINANCE = 0.02
# A requested darkening stops at this fraction of the class's reference
# luminance (its colour in the task's start TF): "darken the liver strongly"
# took it to black on 2026-11-03, which reads as a hole in the body, not a
# darker organ. The goal asks for an absolute change (goals.BRIGHTNESS_STRENGTH,
# strongly -0.4), more than the reference luminance of the liver (0.39),
# heart, spleen and soft tissue, so the floor costs attainment on exactly
# those; the cost on the thesis eval's brightness instructions is in the
# commit that introduced it.
MIN_LUMINANCE_FRACTION = 0.25


def luminance(rgb) -> float:
    return float(np.asarray(rgb, dtype=np.float64) @ LUMINANCE)


def scale_hue(rgb_ref, level: float) -> np.ndarray:
    """`rgb_ref`'s hue at luminance `level` (in [0, 1]), exactly.

    Scaled (`rgb_ref * level / lum(rgb_ref)`) while no channel exceeds 1.
    Beyond that the hue at full saturation (largest channel at 1) is mixed
    toward white just enough to reach `level`: white has luminance 1 and
    luminance is linear, so the mix is exact. A brightened red therefore
    stays red until `level` is nearly 1 and is pure white only at 1.
    """
    rgb_ref = np.asarray(rgb_ref, dtype=np.float64)
    level = float(np.clip(level, 0.0, 1.0))
    ref_level = luminance(rgb_ref)
    if ref_level <= 0.0:
        return np.full(3, level)
    scaled = rgb_ref * (level / ref_level)
    if scaled.max() <= 1.0:
        return scaled
    saturated = rgb_ref / rgb_ref.max()
    saturated_level = luminance(saturated)
    mix = (level - saturated_level) / (1.0 - saturated_level)
    return np.clip((1.0 - mix) * saturated + mix, 0.0, 1.0)


def hue_matches(rgb, rgb_ref, min_cosine=HUE_MIN_COSINE, dark=DARK_LUMINANCE) -> bool:
    """`rgb` has `rgb_ref`'s hue: not near-black, and within `min_cosine`
    of it as a direction in RGB."""
    rgb = np.asarray(rgb, dtype=np.float64)
    rgb_ref = np.asarray(rgb_ref, dtype=np.float64)
    if luminance(rgb) < dark:
        return False
    cosine = float(rgb @ rgb_ref / (np.linalg.norm(rgb) * np.linalg.norm(rgb_ref)))
    return cosine >= min_cosine


def hue_source(rgb, rgb_ref, dark=DARK_LUMINANCE) -> np.ndarray:
    """The colour whose hue a brightness change keeps: `rgb` itself, or
    `rgb_ref` when `rgb` is too dark to have one."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.asarray(rgb_ref, dtype=np.float64) if luminance(rgb) < dark else rgb


def darkening_floor(rgb_ref, rgb_before) -> float:
    """The lowest luminance a requested darkening may reach:
    MIN_LUMINANCE_FRACTION of the reference's -- or the colour's own
    luminance before the command when that is already lower (a session from
    before the floor), so a darkening never brightens."""
    return min(MIN_LUMINANCE_FRACTION * luminance(rgb_ref), luminance(rgb_before))
