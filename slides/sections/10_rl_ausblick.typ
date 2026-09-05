#import "../helpers.typ": *

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — Vom Hill-Climbing zum SAC-Agenten
// ══════════════════════════════════════════════════════════════════════════════
== Vom Hill-Climbing zum SAC-Agenten

#v(0.05em)
Der SAC-Agent (`stable-baselines3`) ist implementiert, trainiert und gegen die Baseline evaluiert — online gegen `opacity_mass`, ganz ohne menschliche Trainingsdaten.

#v(0.3em)
#table(
  columns: (auto, 1fr),
  stroke: none,
  row-gutter: 0.5em,
  column-gutter: 0.8em,
  inset: (x: 0pt, y: 2pt),
  [*Zustand*], [TF-Vektor (24) + Ziel-Gewebe + Richtung + `mass_fraction` (31-dim)],
  [*Aktion*], [kontinuierliches Delta auf die Höhe des aufgelösten Ziel-Peaks],
  [*Reward*], [Δ`mass_fraction` im Ziel-Band, pro Schritt — exakt, kostenlos, kein MLLM],
  [*Ergebnis*], [*0.344* vs. *0.347* (Baseline) `mass_fraction`, 20 Test-Episoden],
)

#v(0.35em)
#remark[
  Nahezu gleichauf mit dem Hill-Climber, aber langsamer konvergent (3.3 vs. 2.7
  Schritte bis 90 %) — ein optimierter Hand-Regler ist auf diesem niedrigdimensionalen
  Problem ein starker Gegner @zhao2025natural. Noch nicht in die Live-Loop verdrahtet;
  nächste Schritte: volle Peak-Kontrolle, Vergleich mit `preferences.jsonl` @scurto2021designing.
]
