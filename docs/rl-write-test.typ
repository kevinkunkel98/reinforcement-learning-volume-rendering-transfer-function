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
#let sky = rgb("#edf2f8")
#let amber = rgb("#b7770d")

#let note(body, title: "Note", color: amber) = block(
  width: 100%, inset: (x: 0.9em, y: 0.6em), radius: 3pt, fill: rgb("#fef9ec"),
  stroke: (left: 3pt + color),
)[#text(weight: "bold", fill: color, size: 0.9em)[#title:] #h(0.4em) #body]

// ══════════════════════════════════════════════════════════════════════════
// Title block
// ══════════════════════════════════════════════════════════════════════════
#align(center)[
  #text(size: 18pt, weight: "bold", fill: navy)[
    Reinforcement Learning for Transfer-Function \& Camera Control
  ] \
  #v(0.3em)
  #text(size: 12pt, fill: luma(80))[Math and Implementation Reference]
  #v(0.6em)
  #line(length: 40%, stroke: 1pt + navy)
  #v(0.6em)
  #text(size: 11pt)[Kevin Kunkel] \
  #text(size: 9pt, fill: luma(110))[
    Universität Leipzig · Masterseminar: Optimierung von Transferfunktionen \
    im Volume Rendering mit Reinforcement Learning
  ] \
  #v(0.3em)
  #text(size: 8.5pt, fill: luma(130))[September 2026]
]

#v(1em)

This is a short, math-focused companion to `architecture.typ`: it derives
every formula the two RL sub-projects are built on, states each environment
as a Markov decision process, and summarizes the learning algorithm and the
hand-coded baseline both agents are compared against. Both base agents train
*offline*, directly against a scalar geometric/radiometric metric computed
from raw data — neither ever calls the VTK renderer. Two extensions build on
the opacity agent specifically: @sec-online trains the same MDP continually
instead of in one batch, and @sec-rlhf replaces the automatic metric with a
reward model learned from human judgments of rendered images — the first
point in this document where rendering enters the RL loop at all.

#outline(title: "Contents", indent: 1.5em)

= Shared building block: the opacity metric

Both sub-projects reuse the same transfer-function representation and the
same integral metric, so it is derived once here and referenced from both.

== Transfer function as a Gaussian mixture

The transfer function is a flat vector $p in [-1,1]^24$: four peaks, six
parameters each — center $c_i$, width $w_i$, height $h_i$, and an RGB triple —
always stored normalized and mapped to real units by an affine rescale
$"to_range"(x, "lo", "hi") = "lo" + (x+1)/2 dot ("hi" - "lo")$. At any Hounsfield-unit
value $x$, peak $i$ contributes a Gaussian bump

$ g_i (x) = exp(-1/2 ((x - c_i)/w_i)^2), quad "contrib"_i (x) = h_i dot g_i (x), $

and the four peaks are composited into one clipped total opacity curve

$ alpha (x; p) = op("clip") (sum_(i=1)^(4) "contrib"_i (x),thin 0, 1) . $

(Color is a contribution-weighted blend of the four peaks' RGB values,
irrelevant to any RL signal and omitted here.)

== `opacity_mass` and `mass_fraction`

`opacity_mass` is the exact trapezoidal integral of $alpha$ over an
HU sub-range $["lo","hi"]$, sampled at $n=256$ points:

$ M(p; "lo", "hi") = integral_"lo"^"hi" alpha (x; p) thin dif x approx "trapz"(alpha (x_1;p), dots, alpha (x_n;p)) . $

This is the ground-truth signal everything downstream is built on — no
rendering needed. Five fixed HU bands (`TISSUE_BANDS`) partition the domain
into tissues (air, fat, soft, spongy, bone). Because raw $M$ scales with a
band's width (bone's band alone is 1400 HU wide, vs. 520 HU for fat), the RL
observation and reward instead use the width-normalized *mass fraction*

$ phi (p, t) = M(p; "lo"_t, "hi"_t) \/ ("hi"_t - "lo"_t) in [0,1], $

an average-opacity-in-band figure that does not reward a policy simply for
picking a wide tissue band.

= RL sub-project 1 — transfer-function opacity <sec-tf>

Can a learned continuous-control policy match the hand-coded hill-climber at
nudging one tissue's opacity toward a goal? Formalized as an episodic MDP
$(cal(S), cal(A), R)$ with deterministic transitions (`rl/env.py`):

== Episode setup

On `reset()`, all four peaks' heights are perturbed by uniform noise
$plus.minus 0.3$ from `default_params()`, then a target tissue
$t in {"air","fat","soft","spongy","bone"}$ and a goal direction
$d in {+1,-1}$ (increase/decrease) are sampled uniformly. The target peak
index is resolved once via the same `_find_or_create_peak` the command layer
uses, so the agent moves the vector through the identical mechanism a real
user command would.

== State, action, reward

$ s = ( p, thin op("onehot") (t), thin d, thin phi (p,t) ) in RR^(24+5+1+1) = RR^31 $

$ a in [-1,1], quad Delta = a dot 0.2 \ $

The action scales a single scalar delta ($"MAX_DELTA"=0.2$) that is added
directly to the target peak's *unit* height (the height slot converted from
$[-1,1]$ to $[0,1]$, clipped back to $[0,1]$) — no direction sign is baked
into the transition; the agent must learn to pick the sign of $a$ itself
from $d$ in the observation:

$ h^"unit"_"new" = op("clip") (h^"unit"_"old" + Delta,thin 0, 1). $

The reward is the signed per-step change in mass fraction toward the goal:

$ r = "sign"(d) dot (phi (p', t) - phi (p, t)), quad "sign"(d) = cases(+1 & d = "increase", -1 & d = "decrease") . $

Episodes truncate (not terminate) after 20 steps — there is no terminal
state, only a fixed interaction budget, matching how a real editing session
has no natural "done."

== Result (200k timesteps, SAC, 20 held-out episodes)

#table(
  columns: (auto, auto, auto),
  stroke: (bottom: 0.5pt + luma(200)),
  inset: (x: 6pt, y: 5pt),
  table.header([], [*Mean final* $phi$], [*Mean steps to 90% of best*]),
  [Trained SAC policy], [0.344], [3.3],
  [Hill-climb baseline], [0.347], [2.7],
)

Near-parity: on this low-dimensional, single-scalar-action problem, a
well-tuned hand rule is a strong baseline. Not wired into the live chat/voice
loop.

#pagebreak()

= RL sub-project 2 — camera viewpoint <sec-camera>

Orient the camera toward a target tissue's centroid direction, against a
cheap exact geometric proxy computed directly from the volume's voxel data —
no rendering, no learned reward model (`rl/camera_env.py`).

== View direction and target direction

The camera is parameterized by azimuth/elevation angles; its unit view
direction in the volume's coordinate frame is

$ v("az","el") = mat(cos("el") sin("az"); sin("el"); cos("el") cos("az")) . $

A target tissue's direction is computed once per episode directly from the
HU volume: mask the voxels in the tissue's band, take their physical-space
centroid, and normalize its offset from the volume's geometric center:

$ macron(c)_t = "voxel-mean"({x : "lo"_t <= "HU"(x) < "hi"_t}) dot "spacing", quad hat(d)_t = (macron(c)_t - c_"vol") / (||macron(c)_t - c_"vol"||) . $

A candidate direction is rejected (falls back to the next tissue, ultimately
`bone`) if its offset is below $2%$ of the volume's diagonal extent — too
close to the center to have a meaningful direction, distinguishing real
anatomical asymmetry from grid-discretization noise.

== State, action, reward

$ s = ( sin("az"), cos("az"), "el"\/85, thin op("onehot") (t), thin A ) in RR^9, quad a in [-1,1]^2 $

$ "az"' = ("az" + a_1 dot 30 degree) mod 360 degree, quad
  "el"' = op("clip") ("el" + a_2 dot 30 degree,thin -85 degree, 85 degree) $

Alignment is the cosine similarity between the current view direction and
the fixed target direction (both unit vectors, so a plain dot product):

$ A("az","el") = v("az","el") dot hat(d)_t in [-1, 1], quad quad r = A' - A . $

Azimuth *wraps* (no natural bound); elevation *clamps* (avoids flipping
through the poles) — the same asymmetry the manual camera commands use.
20-step truncated episodes, as in @sec-tf.

#note(title: "A real bug caught during this sub-project", color: rgb("#8a2e2e"))[
  The hill-climb baseline seeds its step at $30 degree$, but the shared
  `resize_step` growth cap defaulted to $1.0$ — correct for opacity values in
  $[0,1]$, wrong by 30$times$ here. The very first accepted move collapsed the
  step from $30 degree$ to $1 degree$ with no way to grow back: mean final
  alignment *0.658*. Adding an explicit `max_step` parameter (default $1.0$,
  so the three opacity-scale callers are unaffected) and passing
  `max_step=30` at this call site fixed it: *0.99998*.
]

== Result (200k timesteps, SAC, `ct_skull`, 20 held-out episodes)

#table(
  columns: (auto, auto, auto),
  stroke: (bottom: 0.5pt + luma(200)),
  inset: (x: 6pt, y: 5pt),
  table.header([], [*Mean final alignment*], [*Mean steps to 90% of best*]),
  [Trained SAC policy], [0.997], [3.05],
  [Hill-climb baseline], [1.000], [4.05],
)

Both reach near-perfect alignment — the alignment landscape is a smooth,
unimodal dot product against a fixed direction, easy for either method. The
trained policy's edge is convergence speed (fewer renders needed in an
interactive tool), not final quality. The reward has *no occlusion model* —
it is a directional proxy for "facing the tissue," not "can see it
unobstructed."

#pagebreak()

= Learning algorithm: Soft Actor-Critic

Both agents are trained with `stable-baselines3`'s SAC
(`MlpPolicy`, default two-hidden-layer-of-256 network, 4 parallel envs via
`make_vec_env`, `gamma = 0.99` default). SAC is an off-policy, maximum-entropy
actor-critic method: alongside expected return, the policy $pi$ is trained to
maximize its own entropy $cal(H)$, trading a bit of reward for exploration
robustness —

$ J(pi) = EE_pi [ sum_t gamma^t (r_t + alpha_"ent" cal(H)(pi(dot|s_t))) ], $

with twin clipped Q-networks (reducing value overestimation) and an
automatically-tuned entropy coefficient $alpha_"ent"$ (both SB3 defaults, not
overridden here). Both environments are pure Python/NumPy with no rendering
in the loop, so training throughput is bounded by CPU, not GPU or I/O.

= Baseline: hill climbing (`search.py`)

The comparison point for both sub-projects, and the system's actual
production optimizer:

$ "propose_step"(p, i, s, delta): quad h^"unit"_i <- op("clip") (h^"unit"_i + s dot delta,thin 0,1) $

$ "resize_step"(delta, "accepted", delta_"max") = cases(
  min(1.2 delta,thin delta_"max") & "accepted",
  0.5 delta & "rejected"
) $

— grow the step $1.2 times$ on acceptance (capped at a domain-appropriate
$delta_"max"$), halve it on rejection. Acceptance for a relative
increase/decrease command is decided by `evaluate.objective`: let
$Delta_t = plus.minus (M(p', "target") - M(p, "target"))$ (sign matching the
requested direction). The step is accepted iff it actually moved the target
band the right way *and* no other tissue's mass moved by more —
a collateral-damage check with no RL analogue:

$ "accept" arrow.l.r.double Delta_t > epsilon
  and max_(t' eq.not t) |M(p',t') - M(p,t')| <= Delta_t, quad epsilon = 10^(-4). $

Both `rl/eval.py` and `rl/camera_eval.py` measure *steps to 90% of best*
generically: given the trajectory's start value $f_0$ and the best value
either method reached across the pair, the step index at which $f$ first
crosses $f_0 + 0.9(f_"best" - f_0)$ (or the symmetric case for a decreasing
goal).

#note(title: "Scope")[
  Both base agents are trained and evaluated entirely offline against the
  metrics above. The camera policy is not reachable from the live chat/voice
  loop; the opacity policy now is, as frozen inference (`server.py`'s
  `policy` search toggle, via `rl/serve.py`) — see @sec-online and @sec-rlhf
  below for the two ways training itself is being extended. See
  `architecture.typ` for the full system this feeds into.
]

#pagebreak()

= Extension: continual online learning <sec-online>

`rl/train.py` trains once, offline, in a single 200k-step batch across 4
parallel environments, evaluated once at the end. `rl/online_train.py`
trains the *identical* MDP from @sec-tf — same state, action, reward — but
as one continuous stream on a single environment, checkpointing held-out
performance periodically instead of only at the end, so the metric's
improvement over training is directly observable.

== Mechanics

Training proceeds in chunks of `eval_interval` steps. Each chunk is a plain
SB3 call — no custom replay-buffer or gradient-step code — so the online
variant's *learning* mechanics are identical to the offline agent's; only
the scheduling around it differs:

$ "for" k = 1, dots, N\/E: quad "model.learn"("total_timesteps": E,
  thin "reset_num_timesteps": "False") $

where $N$ is the total step budget and $E$ the eval interval. After each
chunk, the current policy is replayed (deterministically) against the same
20 held-out episodes @sec-tf's offline evaluation uses, and the mean final
$phi$ and mean steps-to-90% are logged — the exact same two statistics the
offline result table reports, making the two directly comparable.

== Result ($N=50"k"$ steps, $E=5"k"$, single env)

#table(
  columns: (auto, auto, auto),
  stroke: (bottom: 0.5pt + luma(200)),
  inset: (x: 6pt, y: 5pt),
  table.header([*Steps*], [*Mean final* $phi$], [*Mean steps to 90%*]),
  [5,000], [0.199], [17.35],
  [10,000], [0.281], [16.00],
  [15,000], [0.347], [9.35],
  [20,000], [0.347], [5.15],
  [25,000], [0.346], [3.80],
  [30,000], [0.346], [3.30],
  [35,000], [0.344], [3.30],
  [40,000], [0.345], [3.25],
  [45,000], [0.346], [3.20],
  [50,000], [0.345], [3.20],
)

#note[
  $50"k"$ single-env steps are *25%* of the offline agent's $200"k"$-step
  budget (SB3's `total_timesteps` already counts across all parallel envs,
  so this is a direct comparison, not one that needs a $times 4$ correction).
  The online run lands at $0.345$ — almost exactly between the offline
  policy ($0.344$) and hill-climb baseline ($0.347$) — while
  steps-to-90%-of-best falls steadily from $17.4$ to $3.2$: a clean,
  converging curve on a quarter of the budget, not a coincidental match.
]

#pagebreak()

= Extension: RLHF reward model <sec-rlhf>

$phi$ (@sec-tf) measures only whether the TF curve moved the way a command
literally asked — never whether the resulting *image* looks good. This
extension trains a small reward model from human judgments of rendered
before/after pairs, then trains RL against that learned reward instead of
$phi$'s exact delta — the RLHF pattern, and the first point in this
document where the VTK renderer enters the RL loop at all.

== Preference collection and featurization

`render.features(rgb)` (already used elsewhere for logging) reduces a
rendered frame to four cheap scalars:

$ f(x) = ("mean"(x), thin "std"(x), thin "coverage"(x), thin "entropy"(x)) in RR^4 . $

`rl/collect_preferences.py` samples cases the same way @sec-tf's episodes
are sampled, applies one random action, renders before/after, and asks a
human for a label $y in {0,1}$ (better/worse). Each labeled example is
reduced to a 10-dimensional feature vector — the same information a policy
would have available when it takes that step:

$ z = ( f("after") - f("before"), thin op("onehot")(t), thin d ) in RR^(4+5+1) = RR^10 . $

== Reward model

A small MLP $R_theta : RR^10 -> RR$ ($10 arrow.r 16 arrow.r 1$, ReLU) is
trained on labeled pairs $(z_i, y_i)$ via binary cross-entropy on the
sigmoid of its output:

$ cal(L)(theta) = -1/N sum_(i=1)^N [ y_i log sigma(R_theta (z_i)) +
  (1-y_i) log(1 - sigma(R_theta (z_i))) ] . $

At inference, the reward handed to RL is the sigmoid output remapped to a
signed range matching $phi$'s reward shape:

$ tilde(r)(z) = 2 sigma(R_theta (z)) - 1 in [-1, 1] . $

== Environment: one render per step, not two

`RewardModelTFEnv` subclasses `TFEnv` from @sec-tf unchanged — same
sampling, same action, same *observation* (which still includes the free
$phi$ value) — and overrides only the reward: `step()` renders the new
state, computes $tilde(r)$ from $Delta f$ against the *previous* step's
rendered features, then caches this step's features as next step's
"before" — so a rollout of $k$ steps costs $k+1$ renders, not $2k$.
Measured directly on the synthetic phantom (96³, GPU raycast): *109 ms per
render*, which is why this extension's step budget (~2,000) is two orders
of magnitude smaller than the automatic-metric agents' — every step now
does real work a $phi$-based step never had to.

#note(title: "Scope", color: rgb("#8a2e2e"))[
  Deliberately framed as a single-rater pilot ($n approx 50$ labels), not a
  generalizable result — reward-model train/val accuracy is reported
  honestly regardless of value, per the design's explicit methodological
  stance. All four modules (`rl/collect_preferences.py`,
  `rl/reward_model.py`, `rl/reward_model_env.py`,
  `rl/reward_model_train.py`) are implemented and tested; the pipeline has
  not yet been run against real collected labels, since that step needs a
  live human and cannot run unattended. Camera-viewpoint RLHF and live
  chat-UI wiring for either extension in this section are both out of
  scope for now.
]
