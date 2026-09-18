#import "../helpers.typ": *

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — RLHF: Lernen aus menschlichem Feedback
// ══════════════════════════════════════════════════════════════════════════════
== RLHF: Lernen aus menschlichem Feedback

#v(0.05em)
Bradley-Terry-Modell auf den erhobenen Paarvergleichen; der Reward ersetzt die
Metrik, nicht das Training @scurto2021designing.

#v(0.2em)
#align(center)[
  $ cal(L)(theta) = -1/N sum_(i=1)^N [y_i log sigma(R_theta (z_i)) + (1-y_i) log(1 - sigma(R_theta (z_i)))] $
]

#v(0.15em)
#table(
  columns: (auto, 1fr),
  stroke: none,
  row-gutter: 0.45em,
  column-gutter: 0.8em,
  inset: (x: 0pt, y: 2pt),
  [*Stufe 1*], [Policy gegen `attainment` — ohne Menschen, beliebig viele Episoden],
  [*Stufe 2*], [Feintuning gegen das Reward-Modell, verankert an Stufe 1 (KL-Strafe)],
  [*Merkmale*], [rund 25 gemessene Werte je Kandidat statt Bild-Embeddings],
  [*Messgröße*], [Blind-A/B zwischen Stufe 1 und Stufe 2, Gewinnrate mit KI],
)

#v(0.25em)
#thm-box([Warum die Verankerung nötig ist], [
  Ein gelerntes Reward-Modell gilt nur nahe seiner Daten. Hart optimiert
  gewinnt das Bild, das *das Modell* hoch bewertet — nicht das, was ein Mensch
  gut findet.
], fill: mint, stroke-color: sage)
