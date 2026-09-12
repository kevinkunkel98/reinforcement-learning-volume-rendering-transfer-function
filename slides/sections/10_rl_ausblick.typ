#import "../helpers.typ": *

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — Vom Hill-Climbing zum SAC-Agenten
// ══════════════════════════════════════════════════════════════════════════════
== Vom Hill-Climbing zum SAC-Agenten

#v(0.05em)
SAC (`stable-baselines3`), trainiert & evaluiert gegen `opacity_mass` — ganz
ohne menschliche Trainingsdaten.

#v(0.2em)
#table(
  columns: (auto, 1fr),
  stroke: none,
  row-gutter: 0.4em,
  column-gutter: 0.8em,
  inset: (x: 0pt, y: 2pt),
  [*Zustand*], [TF-Vektor (24) + Ziel-Gewebe + Richtung + `mass_fraction` (31-dim)],
  [*Aktion*], [kontinuierliches Delta auf die Höhe des aufgelösten Ziel-Peaks],
  [*Reward*], [$r = op("sign")(d) dot (phi(p',t) - phi(p,t))$ — Δ`mass_fraction` im Ziel-Band],
  [*Ergebnis*], [*0.344* vs. *0.347* (Baseline) `mass_fraction`, 20 Test-Episoden],
)

#v(0.3em)
#remark[
  - Nahezu gleichauf mit Hill-Climber, aber langsamer (3.3 vs. 2.7 Schritte bis 90 %)
  - Guter Hand-Regler ist auf niedrigdimensionalem Problem starker Gegner @zhao2025natural
  - Live im Chat-UI (`server.py`, `policy`-Toggle) — eingefroren, noch kein Nachlernen
]
