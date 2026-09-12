#import "../helpers.typ": *

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — Motivation
// ══════════════════════════════════════════════════════════════════════════════
== Motivation

#v(0.05em)
Direct Volume Rendering: volumetrische Daten (CT, MRT, Simulationen) sichtbar
über eine #text(fill: navy, weight: "bold")[Transferfunktion].

#v(0.35em)
#definition([Transferfunktion], [
  $T: [min, max] -> [0,1]^4$ — jeder Skalarwert bekommt Farbe (RGB) und Opazität
  ($alpha$). Die Wahl von $T$ entscheidet, welche Strukturen sichtbar werden.
])

#v(0.4em)
#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  [
    *Das Problem:*
    - Parameterraum groß, unintuitiv
    - Erfordert Domänenwissen (z. B. Hounsfield-Einheiten)
    - Manuelles Nachjustieren kostet Zeit
  ],
  [
    *Der Wunsch:*
    - "Zeig mir nur den Knochen"
    - "Weichgewebe ist zu dominant"
    - Sprache/Text statt Schieberegler
  ],
)
