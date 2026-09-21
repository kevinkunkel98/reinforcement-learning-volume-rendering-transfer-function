#set page(numbering: "1", margin: (x: 2.5cm, y: 2.5cm))
#set text(size: 10.5pt, lang: "en")
#set heading(numbering: "1.")
#set par(justify: true, leading: 0.65em)
#show raw: set text(size: 9pt)
#show raw.where(block: true): block.with(
  fill: rgb("#f4f6f9"), inset: 8pt, radius: 3pt, width: 100%,
)
#show link: underline

#let navy = rgb("#1c3a5e")
#let blue = rgb("#1a4f8a")
#let sage = rgb("#1e6b3c")
#let amber = rgb("#b7770d")
#let rust = rgb("#a33a2a")

#let note(body, title: "Note", color: amber) = block(
  width: 100%, inset: (x: 0.9em, y: 0.6em), radius: 3pt, fill: rgb("#fef9ec"),
  stroke: (left: 3pt + color),
)[#text(weight: "bold", fill: color, size: 0.9em)[#title:] #h(0.4em) #body]

#let caveat(body) = note(body, title: "Caveat", color: rust)
#let finding(body) = note(body, title: "Design choice", color: sage)

#align(center)[
  #text(size: 18pt, weight: "bold", fill: navy)[The Transfer Function in TF/RL] \
  #v(5pt)
  #text(size: 11.5pt)[From CT intensity to rendered anatomy, goals, and policy actions] \
  #v(9pt)
  #text(size: 9.5pt, fill: luma(90))[
    Kevin Kunkel #sym.dot.c Master's thesis #sym.dot.c Technical note
  ]
]

#v(12pt)
#note[This note describes the transfer-function representation currently used by
the viewer and reinforcement-learning pipeline. It is a constrained semantic
parameterization, not an arbitrary point-by-point curve editor.]

= Purpose

A direct volume renderer maps a three-dimensional scalar field to an image. In
the CT setting, each voxel carries an intensity measured in Hounsfield Units
(HU). A transfer function (TF) maps that intensity to colour and opacity:

$ T: H U -> (r, g, b, alpha) $

Opacity controls how strongly a sample contributes to the image. Colour controls
its appearance. The same opacity curve can produce very different images on two
patients: tissue in front of a target can hide it. Therefore, this project does
not optimise opacity in isolation. It measures what the completed rendering
actually shows.

= Current Representation

The current TF uses four Gaussian peaks over the CT intensity range. Each peak
represents one semantic material group:

#table(
  columns: (2.1cm, 2.2cm, 1fr),
  align: (left, center, left),
  stroke: (x, y) => if y == 0 { (bottom: 0.8pt + navy) } else { (bottom: 0.3pt + luma(200)) },
  table.header[*Class*][*Centre (HU)*][*Role*],
  [lungs], [-800], [lung parenchyma],
  [soft], [40], [organs and muscle together],
  [vessels], [300], [contrast-enhanced vessels],
  [skeleton], [900], [bone],
)

For peak $i$, opacity contribution around intensity $x$ is modelled as:

$ alpha_i(x) = h_i exp(-((x - mu_i)^2)/(2 sigma_i^2)) $

where $mu_i$ is peak centre, $sigma_i$ is width, and $h_i$ is height. Colour
comes from the peak RGB values. The renderer combines contributions from all
peaks into lookup tables before compositing samples through the volume.

The full internal parameter vector contains six values per peak:

#block(inset: (left: 1em))[
  *centre* $mu_i$ #sym.dot.c *width* $sigma_i$ #sym.dot.c *height* $h_i$ #sym.dot.c
  *red, green, blue* $(r_i, g_i, b_i)$
]

Thus, four peaks give 24 internal values. Peak centres remain fixed at the
anatomical positions above. The policy controls the other 12 parameter groups:

#table(
  columns: (2.7cm, 1fr),
  align: (left, left),
  stroke: (x, y) => if y == 0 { (bottom: 0.8pt + navy) } else { (bottom: 0.3pt + luma(200)) },
  table.header[*Controllable group*][*Meaning*],
  [width], [how wide an intensity neighbourhood contributes],
  [height], [opacity strength of a peak],
  [RGB colour], [colour intensity; one action controls the group mean],
)

Parameters are represented internally in a normalized range. `transfer.py`
converts them to renderable transfer-function values and keeps peak semantics
stable across commands and policy outputs.

#finding[Fixed centres reduce action-space size and preserve semantic meaning.
The tradeoff is expressiveness: current system cannot move peak centres or draw
arbitrary piecewise-linear control points.]

= Colour Preservation

RGB is represented as one controllable group per peak, but the implementation
does not force all three channels to become identical. `apply_controllable`
sets the group mean while preserving each channel's offset from that mean:

$ r'_i = v_i + (r_i - bar(r_i, g_i, b_i)) $

with equivalent equations for green and blue. Values are clipped to the valid
normalized range after reconstruction.

This preserves peak hue when a policy changes colour intensity. Without this
offset-preserving decode, policy outputs collapse to grey and can become
visually identifiable in a supposedly blind preference comparison.

= From Transfer Function to Image

Rendering samples the volume along viewing rays. For each sample, the TF gives
opacity and colour. Samples are composited front-to-back. A simplified update is:

$ C_("out") = C_("in") + (1 - A_("in")) alpha C $ \\
$ A_("out") = A_("in") + (1 - A_("in")) alpha $

where $C$ is sample colour, $alpha$ sample opacity, and $A$ accumulated opacity.
The order matters. A high-opacity skin or muscle layer can reduce later
contributions from bone, even when the bone peak itself is high.

#caveat[Changing one peak has side effects. Increasing bone opacity can alter
occlusion, brightness, and coverage of other classes. This is why a TF-curve
area metric alone is insufficient.]

= Visibility Measurement

`visibility.py` provides a fast approximation of what the rendered image shows.
For each volume, it resamples views into a coarse front-to-back grid. Each sample
has an anatomical class label where TotalSegmentator masks are available. The
same transfer-function lookup and compositing logic then estimates class
contribution.

Raw measured classes are:

#block(inset: (left: 1em))[
  *skeleton* #sym.dot.c *lungs* #sym.dot.c *organs* #sym.dot.c *muscle* #sym.dot.c
  *vessels* #sym.dot.c *coverage*
]

The user-facing goal classes are `skeleton`, `lungs`, `soft`, and `vessels`.
`soft` combines organs and muscle because their intensity ranges overlap and a
one-dimensional TF cannot reliably separate them.

For every class, measurement produces visibility and brightness. Coverage tracks
whether the rendered frame contains meaningful accumulated opacity. The
approximation is fast enough for reinforcement-learning loops and was validated
against label-masked VTK renders. Colour-blackening preserves opacity and
occlusion, isolating a class's image contribution more faithfully than deleting
its voxels.

= Natural Language to TF Goal

An instruction is parsed into a command, then compiled into a goal vector. For
example:

#block(inset: (left: 1em), fill: rgb("#f4f6f9"))[
  `show more bone` #sym.arrow increase skeleton visibility \
  `brighten the skeleton` #sym.arrow increase skeleton brightness \
  `more bone, less soft tissue` #sym.arrow compound visibility goal
]

The goal stores, per user-facing class:

- target change in logarithmic visibility;
- mask indicating which classes are mentioned;
- target brightness change and mask;
- tolerance for unmentioned-class drift.

Logarithmic visibility makes verbal changes approximately multiplicative. A
positive target asks for more visible anatomy; a negative target asks for less.
The keep term prevents unnecessary changes to unmentioned classes but allows
small side effects caused by occlusion.

= Attainment and RL Interface

Let $D_("start")$ be distance from the initial rendering to the instruction goal,
and $D_("final")$ distance after applying a candidate TF. Reported attainment is:

$ A = 1 - D_("final") / D_("start") $

Interpretation:

#table(
  columns: (2cm, 1fr),
  align: (left, left),
  stroke: (x, y) => if y == 0 { (bottom: 0.8pt + navy) } else { (bottom: 0.3pt + luma(200)) },
  table.header[*Attainment*][*Meaning*],
  [1], [goal reached],
  [0], [no better than leaving TF unchanged],
  [< 0], [candidate made goal worse],
)

The one-shot policy observes 57 values:

#block(inset: (left: 1em))[
  goal vector (16) #sym.plus histogram (16) #sym.plus current visibility (4) #sym.plus
  current brightness (4) #sym.plus reachable ceilings (4) #sym.plus current controllable
  values (12) #sym.plus coverage (1)
]

It outputs 12 values, one per controllable width, height, or RGB group. In
absolute mode, outputs are new group values. Experimental residual mode treats
outputs as bounded adjustments from the current TF. Both modes remain one-step:
the policy proposes one complete TF, without iterative policy actions.

The same attainment objective evaluates:

- exact command application;
- coordinate hill-climb search;
- the learned policy;
- policy plus measured refinement.

This shared objective prevents the application comparison panel from silently
using a different ruler than the thesis evaluation.

= What Current TF Does Not Yet Do

#caveat[
  Current representation does not support arbitrary control-point insertion,
  direct line-segment editing, or moving Gaussian centres. It also does not learn
  from human preferences yet. The preference collector is implemented, but the
  reported policy is trained against the validated analytical objective.
]

The design prioritizes semantic, scan-aware instruction following over maximum TF
expressiveness. Future work can add a richer piecewise-linear parameterization,
DINOv2-based visual reranking, or a human-preference reward model after the
measurement and one-shot policy interfaces are established.

= Representation Tradeoff and Future Hybrid Design

The original task description assumes a conventional piecewise-linear VTK
transfer function: consecutive control points define opacity and colour ramps
over the intensity histogram. That representation and the current Gaussian
representation serve different purposes.

#table(
  columns: (3.2cm, 1fr, 1fr),
  align: (left, left, left),
  stroke: (x, y) => if y == 0 { (bottom: 0.8pt + navy) } else { (bottom: 0.3pt + luma(200)) },
  table.header[*Property*][*Piecewise-linear TF*][*Semantic Gaussian TF*],
  [Expressiveness], [arbitrary ramps, points, and sharp transitions], [four fixed semantic peaks],
  [User control], [precise expert curve editing], [compact tissue-level commands],
  [RL interface], [variable or large action space], [fixed 12-value action],
  [Language mapping], [ambiguous point selection], [bone/lung/vessel/soft targets],
  [Generalisation], [can represent unusual materials], [depends on fixed peak centres],
  [Evaluation], [point or curve distance is insufficient], [scan-aware visibility attainment],
)

Gaussian parameterization is therefore a research instrument, not a claim that
piecewise-linear transfer functions are obsolete. It reduces search space,
preserves semantic tissue names, and makes fixed-dimensional policy possible.
Cost: reduced curve expressiveness and dependence on anatomical peak assumptions.

#finding[Most current contribution lies at intersection of two designs:
semantic transfer-function parameterization and RL policy learning. Policy
question is tractable because TF representation is constrained; TF question is
meaningful because visibility is measured in rendered image.]

Future system can combine both representations:

#block(inset: (left: 1em))[
  1. Policy proposes semantic Gaussian TF from spoken or typed instruction. \
  2. Gaussian proposal converts to standard VTK piecewise-linear points. \
  3. Expert edits arbitrary points when more control is needed. \
  4. Visibility measurement evaluates final rendered result. \
  5. Preference or RL training learns from accepted refinements.
]

This hybrid keeps RL action space compact while preserving expert-level curve
editing. It also gives future human-feedback training a correction signal:
accepted point edits become demonstrations of how semantic policy should refine
its proposal.

= Source Files

#table(
  columns: (3.4cm, 1fr),
  align: (left, left),
  stroke: (x, y) => if y == 0 { (bottom: 0.8pt + navy) } else { (bottom: 0.3pt + luma(200)) },
  table.header[*File*][*Role*],
  [`transfer.py`], [TF parameterization and lookup-table conversion],
  [`visibility.py`], [scan-aware class visibility and brightness measurement],
  [`goals.py`], [instruction goals, distances, and attainment],
  [`rl/oneshot_env.py`], [one-shot observation, action, and reward interface],
  [`rl/baselines.py`], [action encoding and search baselines],
  [`server.py`], [interactive command execution and comparison arms],
)
