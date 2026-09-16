# Supported Commands

> **Auto-generated -- do not edit by hand.** Source of truth is `commands.COMMAND_REFERENCE`; run `python -m tools.gen_commands_doc` after changing it and commit the result.

## Opacity (relative)

Nudge a class's visibility up or down by a relative amount.

- `increase opacity for skeleton strongly`
- `decrease opacity for lungs slightly`
- `more bone`
- `a bit less soft tissue`

## Opacity (absolute)

Set a class's visibility to a fixed low/medium/high level.

- `high opacity vessels`
- `low opacity for skeleton`

## Show only

Isolate one or more classes, crushing every other peak to zero.

- `show only skeleton`
- `show only skeleton and lungs`
- `show me the lungs`

## Compound

Set several classes' levels, or nudge several classes' opacity, in one command.

- `high opacity vessels, low opacity skeleton`
- `more bone, a bit less soft tissue`

## Width / sharpness

Adjust how spread out (blended) or narrow (selective) a class's peak is.

- `sharpen the skeleton peak`
- `soften soft tissue`
- `increase width for lungs`

## Brightness

Adjust a class's color brightness.

- `brighten skeleton`
- `darken lungs`
- `low brightness for vessels`

## Center position

Nudge where in Hounsfield space a class's peak sits, clamped to that class's own band.

- `shift skeleton's center up`
- `move lungs down`
- `shift skeleton's position down`

## Camera

Adjust the viewing angle or zoom level. Doesn't change the transfer function.

- `rotate left`
- `tilt down`
- `zoom in`

## Reset

Return the transfer function to its default state.

- `reset`
