#import "../helpers.typ": *

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — Kontinuierliches Online-Lernen
// ══════════════════════════════════════════════════════════════════════════════
== Kontinuierliches Online-Lernen

#v(0.05em)
Statt eines einzigen 200k-Batch-Trainings: ein frischer Agent lernt fortlaufend
auf einer einzelnen Umgebung, alle 5000 Schritte gegen dieselben 20 Test-Episoden
neu bewertet (`rl/online_train.py`).

#v(0.3em)
#table(
  columns: (auto, 1fr),
  stroke: none,
  row-gutter: 0.5em,
  column-gutter: 0.8em,
  inset: (x: 0pt, y: 2pt),
  [*Budget*], [50 000 Schritte, eine Umgebung statt vier parallel],
  [*Verlauf*], [0.199 (5k) $arrow.r$ 0.347 (15k) $arrow.r$ 0.346 (50k) `mass_fraction`],
  [*Konvergenz*], [Schritte-bis-90% fällt von 17.4 auf 3.2],
  [*Budget-Anteil*], [nur 25 % der Umgebungsschritte des Offline-Laufs (200k)],
)

#v(0.35em)
#remark[
  Landet mit nur einem Viertel des Offline-Budgets fast exakt zwischen Policy-
  (0.344) und Hill-Climb-Baseline (0.347) — die Lernkurve konvergiert klar,
  nicht nur zufällig. Zeigt: dasselbe SAC-Setup lernt auch als fortlaufender
  Prozess, nicht nur als einmaliger Batch-Lauf — Voraussetzung für Nachlernen
  aus echtem Feedback (nächste Folie).
]
