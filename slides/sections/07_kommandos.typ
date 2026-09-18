#import "../helpers.typ": *

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — Kommandogrammatik & Parser-Auswertung
// ══════════════════════════════════════════════════════════════════════════════
== Kommandogrammatik & Parser-Auswertung

#v(0.1em)
#block(
  width: 100%, inset: (x: 0.9em, y: 0.6em), radius: 3pt,
  fill: sand, stroke: 0.5pt + luma(200),
)[
  ```
  increase|decrease opacity for <klasse> [slightly|moderately|strongly]
  show only <klasse> [and <klasse> ...]
  <low|medium|high> opacity for <klasse> [, ... weitere Klassen]
  brighten|darken <klasse>   ·   reset
  ```
]

#v(0.5em)
#grid(
  columns: (1fr, auto),
  gutter: 1.5em,
  align: (left + horizon, right + horizon),
  [
    *Testset*
    - 21 freie Formulierungen — Regelparser versteht sie *per Design* nicht
    - z. B. _"make the skeleton pop"_, _"dim the soft tissue a little"_
    - LLM bekommt dieselbe Synonymtabelle wie der Regelparser
    - Vier Zielklassen: Skelett, Lunge, Weichgewebe, Gefäße
  ],
  table(
    columns: 2,
    stroke: 0.5pt + luma(210),
    inset: 8pt,
    [*Parser*], [*Korrekt*],
    [Regelbasiert], [0 / 21],
    [LLM (qwen2.5:7b)], [*20 / 21*],
  ),
)
