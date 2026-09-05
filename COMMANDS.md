# Supported Commands

> **Auto-generated -- do not edit by hand.** Source of truth is `commands.COMMAND_REFERENCE`; run `python -m tools.gen_commands_doc` after changing it and commit the result.

## Opacity (relative)

Nudge a tissue's visibility up or down by a relative amount.

- `increase opacity for bone strongly`
- `decrease opacity for fat slightly`

## Opacity (absolute)

Set a tissue's visibility to a fixed low/medium/high level.

- `high opacity spongy`
- `low opacity for bone`

## Show only

Isolate one or more tissues, crushing every other peak to zero.

- `show only bone`
- `show only bone and spongy`

## Compound

Set several tissues' absolute levels in one command.

- `high opacity spongy, low opacity bones`

## Width / sharpness

Adjust how spread out (blended) or narrow (selective) a tissue's peak is.

- `sharpen the bone peak`
- `soften soft tissue`
- `increase width for fat`

## Brightness

Adjust a tissue's color brightness.

- `brighten bone`
- `darken fat`
- `low brightness for spongy`

## Center position

Nudge where in Hounsfield space a tissue's peak sits, clamped to that tissue's own band.

- `shift bone's center up`
- `move fat down`
- `shift fat's position down`

## Reset

Return the transfer function to its default state.

- `reset`
