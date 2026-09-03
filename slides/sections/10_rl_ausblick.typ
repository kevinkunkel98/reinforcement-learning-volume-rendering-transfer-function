#import "../helpers.typ": *

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — Von der Baseline zum RL-Agenten
// ══════════════════════════════════════════════════════════════════════════════
== Von der Baseline zum RL-Agenten

#v(0.05em)
Das Hill-Climbing ist die *Baseline*, die ein gelernter Agent schlagen muss — die Logs sind bereits die Trainingsdaten dafür @scurto2021designing.

#v(0.35em)
#table(
  columns: (auto, 1fr),
  stroke: none,
  row-gutter: 0.5em,
  column-gutter: 0.8em,
  inset: (x: 0pt, y: 2pt),
  [*Zustand*], [TF-Vektor (24 Werte) + Bildmerkmale (mean, std, coverage, Entropie)],
  [*Aktion*], [kontinuierliche Deltas pro Peak-Dimension (statt diskretem Regelparser)],
  [*Reward*], [`opacity_mass` im Ziel-HU-Bereich — exakt, kostenlos, kein MLLM nötig],
  [*Daten*], [`log.jsonl` · `preferences.jsonl` (paarweise) · `feedback.jsonl` (Daumen)],
)

#v(0.35em)
#remark[
  Analoge RL-Formulierung bereits für sprachgesteuerte Viewpoint-Wahl gezeigt
  @zhao2025natural — hier auf die Transferfunktion angewendet, mit einer harten
  Metrik statt eines MLLM-Urteils @ai2025evaluation.
]
