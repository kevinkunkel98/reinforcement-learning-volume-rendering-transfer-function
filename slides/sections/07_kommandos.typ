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
  increase|decrease opacity for <gewebe> [slightly|moderately|strongly]
  show only <gewebe> [and <gewebe> ...]
  <low|medium|high> opacity for <gewebe> [, ... weitere Gewebe]
  reset
  ```
]

#v(0.5em)
#grid(
  columns: (1fr, auto),
  gutter: 1.5em,
  align: (left + horizon, right + horizon),
  [
    *Testset:* 20 freie Formulierungen, die der Regelparser *per Design* nicht
    versteht (z. B. _"make the skeleton pop"_, _"the spongy core is barely there"_).
    #v(0.3em)
    Der LLM-Parser bekommt dieselbe Synonymtabelle wie der Regelparser im
    Systemprompt — das behebt einen systematischen Fehler (Spongiosa $arrow.r$
    generisches "Bone").
  ],
  table(
    columns: 2,
    stroke: 0.5pt + luma(210),
    inset: 8pt,
    [*Parser*], [*Korrekt*],
    [Regelbasiert], [0 / 20],
    [LLM (qwen2.5:7b)], [*18 / 20*],
  ),
)
