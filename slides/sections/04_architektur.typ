#import "../helpers.typ": *
#import "../diagrams.typ": pipeline-diagram

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — Architektur
// ══════════════════════════════════════════════════════════════════════════════
== Architektur

#v(0.3em)
#align(center)[
  #box(width: 100%, scale(x: 92%, y: 92%, reflow: true)[#pipeline-diagram])
]

#v(0.4em)
Jeder Schritt ist unabhängig austauschbar — der Regelparser wird später durch
einen RL-Agenten ersetzt, ohne dass Rendering, Bewertung oder Logging sich
ändern.
