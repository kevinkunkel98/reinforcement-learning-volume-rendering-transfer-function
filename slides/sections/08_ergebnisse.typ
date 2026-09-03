#import "../helpers.typ": *

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — Ergebnisse an echten CT-Daten
// ══════════════════════════════════════════════════════════════════════════════
== Ergebnisse an echten CT-Daten

#v(0.1em)
Reale, de-identifizierte CT-Datensätze (3D Slicer Testdaten) — kein zusätzlicher
Code für echte Daten nötig, da bereits in HU gearbeitet wird.

#v(0.4em)
#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  align: center,
  [
    #image("../skull_before.png", width: 92%)
    #v(0.2em)
    #text(size: 0.85em, fill: luma(100))["Startzustand" — Standard-TF]
  ],
  [
    #image("../skull_after.png", width: 92%)
    #v(0.2em)
    #text(size: 0.85em, fill: luma(100))["show only bone" — ein Kommando]
  ],
)
