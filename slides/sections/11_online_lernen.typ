#import "../helpers.typ": *

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — Kontinuierliches Online-Lernen
// ══════════════════════════════════════════════════════════════════════════════
== Kontinuierliches Online-Lernen

#v(0.05em)
Statt 200k-Batch: fortlaufendes Training auf einer Umgebung, alle 5000
Schritte gegen dieselben 20 Test-Episoden neu bewertet (`rl/online_train.py`).

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

#v(0.3em)
#remark[
  - Mit ¼ des Offline-Budgets fast exakt zwischen Policy (0.344) und Hill-Climb (0.347)
  - Klare Konvergenz, kein Zufall — dasselbe SAC lernt auch fortlaufend, nicht nur im Batch
  - Voraussetzung für Nachlernen aus echtem Feedback (nächste Folie)
]
