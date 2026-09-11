#import "../helpers.typ": *

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — RLHF: Lernen aus menschlichem Feedback
// ══════════════════════════════════════════════════════════════════════════════
== RLHF: Lernen aus menschlichem Feedback

#v(0.05em)
`mass_fraction` misst nur, ob sich die TF-Kurve wie gewünscht bewegt hat — nie,
ob das gerenderte Bild tatsächlich gut aussieht. Vier neue Module schließen
diese Lücke @scurto2021designing: ein kleines Reward-Modell, gelernt aus
menschlichen Besser/Schlechter-Urteilen über gerenderte Bildpaare.

#v(0.3em)
#table(
  columns: (auto, 1fr),
  stroke: none,
  row-gutter: 0.5em,
  column-gutter: 0.8em,
  inset: (x: 0pt, y: 2pt),
  [*Label-Signal*], [Mensch bewertet 50 gerenderte Vorher/Nachher-Paare (`collect_preferences.py`)],
  [*Reward-Modell*], [10 $arrow.r$ 16 $arrow.r$ 1 MLP auf Bildmerkmalen (Mean/Std/Coverage/Entropy), BCE-Loss],
  [*RL-Umgebung*], [`RewardModelTFEnv`: ein Render pro Schritt (Cache statt Doppel-Render), 109 ms gemessen],
  [*Budget*], [~2000 Schritte statt 50k — jeder Schritt rendert jetzt wirklich],
)

#v(0.35em)
#remark[
  Pipeline implementiert und getestet (4 Module, alle Reviews bestanden); der
  echte Lauf mit menschlichen Labels steht noch aus. Bewusst als
  Einzelrater-Pilot gerahmt (n≈50) — Verallgemeinerbarkeit ist zukünftige
  Arbeit, nicht Anspruch dieser Iteration.
]
