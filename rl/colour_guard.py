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
  comes back -- but never darker than `hue.darkening_floor`
  (MIN_LUMINANCE_FRACTION of the reference's luminance): "darken the liver
  strongly" took it to black on 2026-11-03, a hole in the body.
- no brightness target, colour before the command (`start_params`) has the
  reference hue (`hue_matches`): that colour, exactly. The answer's rgb is
  discarded.
- no brightness target, colour before the command near-black (luminance
  below DARK_LUMINANCE, no measurable hue): that colour, exactly. The guard
  keeps an unasked class's colour, so the policy can no longer leave one
  black behind; a black before the command is a user's darkening ("darken
  the liver strongly", 2026-11-03, which the next command "darken skeleton
  strongly" used to undo by resetting the liver to orange) or a pre-guard
  session, and only the first is worth guessing about.
- no brightness target, colour before the command garbled (`is_garbled`: a
  measurable luminance with a clearly different hue -- a white, grey or
  desaturated heart from an earlier unguarded answer): the reference colour,
  at the reference's luminance. Its luminance carried no request either --
  it is what the earlier recolouring left behind -- so it is not kept.

Centre, width and height (opacity) are never touched. The reference palette
is the task's start TF (`goals.starting_params`, the viewer's
`server.initial_params`).
"""
import numpy as np

import goals
import transfer
from hue import (  # noqa: F401  (re-exported: the guard's vocabulary)
    DARK_LUMINANCE, HUE_MIN_COSINE, LUMINANCE, darkening_floor, hue_matches, luminance, scale_hue,
)

P = transfer.PARAMS_PER_PEAK


def is_garbled(rgb, rgb_ref, min_cosine=HUE_MIN_COSINE, dark=DARK_LUMINANCE) -> bool:
    """`rgb` is bright enough to show a hue (luminance >= `dark`) and that
    hue is not `rgb_ref`'s. A near-black colour is never garbled: it has no
    hue to compare, and it is how a darkening looks."""
    return luminance(rgb) >= dark and not hue_matches(rgb, rgb_ref, min_cosine, dark)


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
        before = _rgb(start_params, index)
        if goal_class in asked:
            level = max(luminance(_rgb(guarded, index)), darkening_floor(reference, before))
            rgb = scale_hue(reference, level)
        else:
            rgb = reference if is_garbled(before, reference) else before
        guarded[index * P + 3:index * P + 6] = transfer._from_unit(rgb)
    return guarded
