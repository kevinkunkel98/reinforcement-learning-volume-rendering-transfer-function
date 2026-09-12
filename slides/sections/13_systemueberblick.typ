#import "../helpers.typ": *

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — Systemüberblick (inkl. RL)
// ══════════════════════════════════════════════════════════════════════════════
== Systemüberblick

#v(0.1em)
#align(center)[
  #image("../architecture-highlevel.png", width: 88%)
]

#v(0.1em)
#remark[
  Sieben Komponenten statt einzelner Python-Klassen: Reinforcement Learning
  (offline, online, RLHF) steht neben Search \& Evaluation als zweite Quelle
  für Akzeptanzentscheidungen — austauschbar, ohne dass Parser, Rendering
  oder Logging sich ändern.
]
