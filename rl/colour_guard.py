"""Inference-time colour guard: a class's colour changes only when the
instruction asks for its brightness, and never loses its hue.

The policy's action moves each peak's (r, g, b) as one group
(`rl.baselines.apply_controllable`: one value, the channels' offsets kept,
clipped to the bounds), and planning and the warm search do the same. Pushed
far, every channel clips at 1 or 0 and the hue is gone: red -> white, anything
-> black. `goals.distance` charges colour only through `bright` -- of the
classes with a brightness target, and of visible unmentioned classes beyond a
tolerance -- so a hidden class or a weight-0 occluder recolours for free. In
the viewer that showed up chained (2026-10-25, out/ui_session.json): "show
only skeleton" turned the hidden heart white, so "add heart" rendered it grey
like bone, and "add lungs" turned the skeleton black.

Brightness here is what the visibility model measures: Rec. 709 luminance
(`visibility.LUMINANCE`) of the colour lookup, averaged over a class's
contributing samples. In label-scoped rendering a class's samples are shaded
by its own peak alone, so a class's `bright` is the luminance of its peak's
rgb (up to the lookup's 1e-6 blend floor), and it is linear in rgb. Visibility
(`vis`, `coverage`) depends on opacity only, never on rgb. So any rgb with
the same luminance gives the same `bright` and the same `vis`: the guard below
keeps luminance wherever a brightness target or the keep term could see it,
and changes nothing the goal metric measures except where it repairs a
colour. (The unlabelled row blends the skeleton, lungs and soft peaks; their
luminance is kept too, so its brightness only moves where a colour is
repaired.)

Per class (one peak per goal class, `transfer.ANATOMICAL_PEAK_INDEX`):

- brightness target in the goal (mask n != 0): the reference hue at the
  answer's luminance -- the requested brightening/darkening stays, the hue
  comes back.
- no brightness target, colour before the command (`start_params`) has the
  reference hue (`hue_matches`): that colour, exactly. The answer's rgb is
  discarded.
- no brightness target, colour before the command already garbled (white,
  grey or black heart from an earlier unguarded answer): the reference
  colour, at the reference's luminance. Its luminance carried no request
  either -- it is what the earlier recolouring left behind -- so it is not
  kept.

Centre, width and height (opacity) are never touched. The reference palette
is the task's start TF (`goals.starting_params`, the viewer's
`server.initial_params`).
"""
import numpy as np

import goals
import transfer
import visibility

P = transfer.PARAMS_PER_PEAK
LUMINANCE = visibility.LUMINANCE
# Same hue: the angle between the colour and the reference colour in RGB is
# at most ~18 deg. Red (0.95, 0.2, 0.2) against white is 38 deg, against its
# start-noise shifts (rgb +-0.15, rl.oneshot_env.START_NOISE) <= 10 deg; liver
# (0.75, 0.3, 0.2) fails it about half-way to white, at (1, 0.75, 0.65).
HUE_MIN_COSINE = 0.95
# Below this luminance a colour renders black and has no hue to compare.
DARK_LUMINANCE = 0.02


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


def brightness_classes(goal) -> set:
    """Goal classes the goal sets a brightness target for (mask n != 0)."""
    goal = np.asarray(goal, dtype=np.float64)
    n = len(goals.GOAL_CLASSES)
    return {c for i, c in enumerate(goals.GOAL_CLASSES) if goal[3 * n + i]}


def _rgb(params, index) -> np.ndarray:
    return np.asarray(transfer.peak_internal(params, index)["rgb"], dtype=np.float64)


def colour_guard(goal, start_params, params, reference_params, weights=None) -> np.ndarray:
    """`params` with every peak's rgb set by the rules in the module
    docstring; heights, widths and centres are `params`' own. `weights` is
    accepted for symmetry with `keep_repair` and unused: colour rules do not
    depend on how the instruction weighs a class."""
    del weights
    start_params = np.asarray(start_params, dtype=np.float64)
    reference_params = np.asarray(reference_params, dtype=np.float64)
    guarded = np.asarray(params, dtype=np.float64).copy()
    asked = brightness_classes(goal)
    for goal_class, index in transfer.ANATOMICAL_PEAK_INDEX.items():
        reference = _rgb(reference_params, index)
        if goal_class in asked:
            rgb = scale_hue(reference, luminance(_rgb(guarded, index)))
        else:
            before = _rgb(start_params, index)
            rgb = before if hue_matches(before, reference) else reference
        guarded[index * P + 3:index * P + 6] = transfer._from_unit(rgb)
    return guarded
