#import "../helpers.typ": *

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — RL v2: eine ziel-konditionierte Policy
// ══════════════════════════════════════════════════════════════════════════════
== Von der Suche zur gelernten Policy

#v(0.1em)
SAC, trainiert gegen *Sichtbarkeit pro Gewebeklasse* — ohne menschliche Daten.

#v(0.25em)
#table(
  columns: (auto, 1fr),
  stroke: none,
  row-gutter: 0.4em,
  column-gutter: 0.8em,
  inset: (x: 0pt, y: 2pt),
  [*Ziel*], [16 Werte: gewünschte Änderung von Sichtbarkeit und Helligkeit je Klasse],
  [*Zustand*], [Ziel + HU-Histogramm + aktuelle Sichtbarkeit + erreichbare Obergrenze (57-dim)],
  [*Aktion*], [Höhe, Breite und Farbe aller vier Peaks in *einem* Schritt (12-dim)],
  [*Reward*], [`attainment`: welcher Anteil der geforderten Änderung erreicht wurde],
)

#v(0.35em)
#remark[
  Vier anatomische Klassen statt HU-Bänder. Organe und Muskeln teilen sich eine
  Klasse — keine Transferfunktion kann sie trennen.
]

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — Ergebnisse auf ungesehenen Patienten
// ══════════════════════════════════════════════════════════════════════════════
== Ergebnisse: 200 Instruktionen, sechs ungesehene Patienten

#v(0.1em)
#table(
  columns: (1fr, auto, auto, auto),
  align: (left, center, center, center),
  stroke: (x, y) => if y == 0 { (bottom: 0.8pt + navy) } else { (bottom: 0.3pt + luma(200)) },
  inset: (x: 4pt, y: 4pt),
  table.header[*Verfahren*][*Evaluationen*][*Median*][*verbessert*],
  [Hill-Climbing (gründlich)], [200], [+0.730], [100 %],
  [*Policy + 3 Verfeinerungen*], [*4*], [*+0.316*], [76 %],
  [*Policy allein*], [*0*], [*+0.275*], [73 %],
  [Hill-Climbing (günstig)], [10], [+0.263], [93 %],
  [Nichts tun], [0], [0.000], [—],
  [Regelbasierter Executor], [0], [#sym.minus 0.022], [37 %],
  [Zufall], [0], [#sym.minus 0.040], [39 %],
)

#v(0.25em)
#thm-box([Kernaussage], [
  Die Policy erreicht *ohne eine einzige Evaluation* das Niveau eines
  Hill-Climbers mit zehn — und schlägt jede nicht-suchende Baseline
  ($p < 0.001$, drei Seeds).
], fill: mint, stroke-color: sage)

#v(0.2em)
#remark[
  Ehrlich bleiben: die Suche ist weiterhin *zuverlässiger* (93 % vs. 73 % der
  Instruktionen verbessert). Das Argument für die Policy sind die Kosten pro
  Instruktion, nicht die Spitzenqualität.
]
