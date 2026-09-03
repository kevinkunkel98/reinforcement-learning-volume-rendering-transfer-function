#import "../helpers.typ": *

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — Erkenntnisse aus dem Testen
// ══════════════════════════════════════════════════════════════════════════════
== Erkenntnisse aus dem Testen

#v(0.15em)
Drei konkrete Bugs, gefunden durch tatsächliches Testen an echten Daten und
echter Nutzung — nicht nur durch Unit-Tests:

#v(0.35em)
#thm-box([Render-Crash], [
  Rendering nahm implizit einen würfelförmigen Datensatz an — stürzte sofort ab
  auf echten $512 times 512 times 139$-CT-Daten. Fix: Dimensionen pro Achse statt
  einer einzigen Kantenlänge.
], fill: rgb("#fdeaea"), stroke-color: rgb("#a83232"))

#v(0.3em)
#thm-box([Sättigungs-Bug], [
  "increase opacity for bone strongly" erreichte in *einem* Schritt das Maximum —
  jedes weitere Kommando war ein stiller No-Op. Fix: Schrittweite proportional
  zur verbleibenden Distanz zum Rand statt fixer Betrag.
], fill: rgb("#fff3d6"), stroke-color: amber)

#v(0.3em)
#thm-box([Spracherkennung], [
  Whisper (small) verhört sich bei "spongy" reproduzierbar — "sponges", "spudgy",
  "spore G". In 24 echten Sprachkommandos scheiterten 19 (79 %) am Parser, nicht
  am System.
], fill: mint, stroke-color: sage)
