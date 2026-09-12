#import "@preview/touying:0.6.1": *
#import themes.metropolis: *
#import "helpers.typ": *

// ── Slide setup ───────────────────────────────────────────────────────────────
#show: metropolis-theme.with(
  footer-right: context {
    if state("show-slide-number", true).get() { utils.slide-counter.display() }
  },
  aspect-ratio: "16-9",
  config-colors(
    primary: navy,
    primary-light: rgb("#1a4f8a"),
    secondary: rgb("#1c3a5e"),
    neutral-lightest: white,
    neutral-light: rgb("#edf2f8"),
  ),
  config-page(margin: (x: 2.8em, y: 2.2em)),
  config-info(
    title: [Optimierung von Transferfunktionen im Volume Rendering mit Reinforcement Learning],
    subtitle: [Masterseminar: Pitchvortrag],
    author: [Kevin Kunkel],
    date: [2026],
    institution: [Universität Leipzig — \ Betreuer: Prof. Gerik Scheuermann, Dr. Baldwin Nsonga, Dr. Thomas Wischgoll],
  ),
  footer: [Leipzig, September 2026],
)

#set text(size: 19pt)
#set block(breakable: false)
#set bibliography(title: none, style: "apa")

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 1 — Titelfolie
// ══════════════════════════════════════════════════════════════════════════════
#slide(
  config: utils.merge-dicts(
    config-methods(header: _ => none, footer: _ => none),
    config-common(freeze-slide-counter: true),
  ),
  align: horizon,
)[
  #set align(left)
  #v(1fr)
  #text(size: 10pt, fill: luma(140), tracking: 2pt)[MASTERSEMINAR · PITCH · SS 2026]
  #v(0.5em)
  #set par(leading: 0.75em)
  #text(
    size: 28pt,
    weight: "bold",
    fill: navy,
  )[Optimierung von Transferfunktionen im Volume Rendering mit Reinforcement Learning]
  #v(0.85em)
  #line(length: 100%, stroke: 1.5pt + navy)
  #v(0.65em)
  #grid(
    columns: (1fr, auto),
    align: (left + horizon, right + horizon),
    [
      #text(size: 13pt, weight: "bold")[Kevin Kunkel] \
      #v(0.15em)
      #text(size: 11pt, fill: luma(110))[Universität Leipzig · Betreuer: Prof. Gerik Scheuermann, Dr. Thomas Wischgoll, Dr. Baldwin Nsonga]
    ],
    [#image("leipziglogo.png", width: 12em)],
  )
  #v(1fr)
]

// ══════════════════════════════════════════════════════════════════════════════
// Sections
// ══════════════════════════════════════════════════════════════════════════════
#include "sections/01_gliederung.typ"
#include "sections/02_motivation.typ"
#include "sections/03_verwandte_arbeiten.typ"
#include "sections/04_architektur.typ"
#include "sections/05_transferfunktion.typ"
#include "sections/06_mvp.typ"
#include "sections/07_kommandos.typ"
#include "sections/08_ergebnisse.typ"
#include "sections/09_erkenntnisse.typ"
#include "sections/10_rl_ausblick.typ"
#include "sections/11_online_lernen.typ"
#include "sections/12_rlhf.typ"
#include "sections/13_systemueberblick.typ"

#slide(config: config-methods(header: _ => none, footer: _ => none))[
  #align(center + horizon)[
    #v(1fr)
    #text(size: 36pt, weight: "bold", fill: navy)[Vielen Dank für eure Aufmerksamkeit!]
    #v(0.6em)
    #line(length: 40%, stroke: 1.5pt + navy)
    #v(0.6em)
    #text(size: 24pt, fill: luma(80))[Fragen?]
    #v(1fr)
  ]
]

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — Literatur (backup)
// ══════════════════════════════════════════════════════════════════════════════
#slide(title: [Literatur])[
  #set text(size: 14pt)
  #bibliography("refs.bib")
]
