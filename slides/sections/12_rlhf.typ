#import "../helpers.typ": *

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — RLHF: Lernen aus menschlichem Feedback
// ══════════════════════════════════════════════════════════════════════════════
== RLHF: Lernen aus menschlichem Feedback

#v(0.05em)
`mass_fraction` misst nur Kurvenbewegung, nie Bildqualität — Reward-Modell aus
menschlichen Urteilen schließt die Lücke @scurto2021designing.

#v(0.25em)
#align(center)[
  $ cal(L)(theta) = -1/N sum_(i=1)^N [y_i log sigma(R_theta (z_i)) + (1-y_i) log(1 - sigma(R_theta (z_i)))],
    quad tilde(r)(z) = 2 sigma(R_theta (z)) - 1 $
]

#v(0.15em)
#table(
  columns: (auto, 1fr),
  stroke: none,
  row-gutter: 0.45em,
  column-gutter: 0.8em,
  inset: (x: 0pt, y: 2pt),
  [*Label-Signal*], [50 gerenderte Paare gezielt (`collect_preferences.py`) *oder* passiv aus
    echter Chat-Nutzung (`ingest_feedback.py`, aus `out/feedback.jsonl`)],
  [*Reward-Modell*], [10 $arrow.r$ 16 $arrow.r$ 1 MLP auf Bildmerkmalen (Mean/Std/Coverage/Entropy)],
  [*Echter Lauf*], [15 nutzbare Labels aus echter Nutzung $arrow.r$ train\_acc *0.833*,
    val\_acc *1.000* (n=12/3 — zu klein für eine echte Aussage)],
  [*RL-Umgebung*], [`RewardModelTFEnv`: ein Render pro Schritt, 109 ms gemessen; RL-Training
    gegen das Reward-Modell selbst noch nicht gelaufen],
)

#v(0.3em)
#remark[
  - Passive Sammlung (`ingest_feedback.py`) statt dedizierter Session war die bessere Idee
  - Nutzt Features, die `server.py` ohnehin schon loggt — skaliert mit echter Nutzung
  - n=15 ist ein Pilot, kein belastbares Ergebnis — bewusst so kommuniziert
]
