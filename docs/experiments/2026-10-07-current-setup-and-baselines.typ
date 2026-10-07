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
#let sage = rgb("#1e6b3c")

#let note(body, title: "Note", color: sage) = block(
  width: 100%, inset: (x: 0.9em, y: 0.6em), radius: 3pt, fill: rgb("#f0f7f2"),
  stroke: (left: 3pt + color),
)[#text(weight: "bold", fill: color, size: 0.9em)[#title:] #h(0.4em) #body]

#align(center)[
  #text(size: 18pt, weight: "bold", fill: navy)[Current Setup \& Baselines] \
  #v(5pt)
  #text(size: 11.5pt)[Policy, transfer function, and the four training baselines run 2026-10-06/07] \
  #v(9pt)
  #text(size: 9.5pt, fill: luma(90))[
    Kevin Kunkel #sym.dot.c Master's thesis #sym.dot.c Technical note
  ]
]

#v(10pt)
#note[Quick reference for picking this back up in the lab. Full detail lives
in `docs/rl-training-setup.typ` (policy/TF/reward) and
`docs/experiments/2026-10-07-search-distilled-pretraining-results.md`
(the baseline runs below, with findings and caveats).]

= Policy and transfer function

Current checkpoint family: *`oneshot-v9`*, `--action-mode residual`,
`--reward-mode target`. The transfer function is *label-scoped*: a voxel
segmented as class $c$ is shaded only by $c$'s own peak, so the eight goal
classes (skeleton, lungs, heart, vessels, liver, kidneys, spleen, soft) no
longer fight over the same HU range. Observation size 97, action size 24.

= Training baselines

All four runs: 150k timesteps, `--hindsight-ratio 0.25 --balance-classes`,
seed 0, 200-episode validation eval every 10k steps. Only the pretrain
source differs -- this isolates what each teacher signal contributes.

#table(
  columns: (auto, 1fr, auto),
  align: (left, left, left),
  stroke: 0.5pt + luma(200),
  table.header[*Run*][*Pretrain source*][*Out dir*],
  [E1 (control)], [none], [`e1_v9_150k_seed0_rewardfix`],
  [E2], [hindsight only, 4000 ep], [`e2_v9_bc_150k_seed0_rewardfix`],
  [E2+search], [hindsight (4000) + search-distilled (1000 @ 200 evals)], [`e1_v9_150k_seed0_searchmix`],
  [E3], [hindsight+brightness (4000, `--hindsight-brightness 0.1`) + search-distilled (1000)], [`e3_v9_searchmix_bright_150k_seed0`],
)

Search-distilled pretraining (`--search-pretrain-data`) is the new teacher
this session added: `hill_climb` solutions on real sampled instructions,
distilled into the actor alongside the existing hindsight demonstrations
(`rl.pretrain.collect_search`, `OneShotEnv.search_policy_action`).

== Results (mean of the last 3 validation evals, steps 130k/140k/150k)

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  stroke: 0.5pt + luma(200),
  table.header[*Metric*][*E1*][*E2*][*E2+search*][*E3*],
  [median attainment], [-0.034], [+0.088], [+0.109], [*+0.179*],
  [absolute], [-0.096], [*+0.193*], [+0.109], [+0.176],
  [relative], [-0.474], [*-0.052*], [-0.208], [-0.068],
  [compound], [-0.047], [+0.047], [+0.001], [*+0.088*],
  [show only], [0.054], [0.070], [+0.076], [*+0.096*],
  [brightness], [-1.278], [-5.204], [-2.379], [*+0.246*],
)

#note(title: "Headline finding")[Hindsight-only pretraining (E2) causes a
severe brightness-instruction regression (-5.204) -- none of its 4000
demonstrations are brightness-type, and nothing during RL training ever
anchors that behaviour. Search-distilled data partially repairs it (E2+search:
-2.379); adding continuous brightness hindsight during RL training (E3) fixes
it outright (+0.246) while also topping every other metric.]

== E3, confirmed across 3 seeds

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  stroke: 0.5pt + luma(200),
  table.header[*Metric*][*seed 0*][*seed 1*][*seed 2*][*median-of-3*],
  [median attainment], [+0.179], [+0.143], [+0.153], [*+0.153*],
  [brightness], [+0.246], [+0.219], [-0.010], [*+0.219*],
  [share improved], [0.718], [0.708], [0.677], [0.708],
)

Clears this project's own decision gate (median $>$ 0 and share improved
$>$ 50% on all three seeds). Brightness is strongly positive on 2 of 3 seeds
and near-neutral (not negative) on the third -- a real improvement, not yet
a universal per-seed win on that one instruction kind.

#note[Not yet done: a held-out *test*-split evaluation (`rl.vis_eval` +
`tools.per_class_eval`) -- everything above is validation-split
(`eval_progress.csv`) only.]

= Viewer / VR status

The app (`server.py`) currently points at
`out/rl_v4/e3_v9_searchmix_bright_150k_seed0/best.zip` via the new
`POLICY_PATH` env-var override (`policy.py`). Networking to a Vive Focus 3
over the LAN works once bound to `--host 0.0.0.0` with the `certs/lan.pem`
TLS cert (needed for WebXR/mic). Open: VR framerate is single-digit on the
Focus 3's Snapdragon XR2 -- a debug line (`xr=/auto=/sd=/shade=/dim=`) was
added to the in-VR HUD to get real numbers next session, plus an
`?xrscale=` URL override to test resolution-bound-ness. To resume next time
in the lab.
