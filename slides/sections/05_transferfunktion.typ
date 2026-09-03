#import "../helpers.typ": *
#import "../diagrams.typ": tf-curve-diagram

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — Die Transferfunktion
// ══════════════════════════════════════════════════════════════════════════════
== Die Transferfunktion: 24 Werte

#v(0.2em)
#align(center)[
  #box(width: 100%, scale(x: 88%, y: 88%, reflow: true)[#tf-curve-diagram])
]

#v(0.3em)
#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  [
    *Design-Entscheidung:* Zentren und Breiten liegen in echten
    #text(fill: navy, weight: "bold")[Hounsfield-Einheiten] (HU) — nicht in
    $[0, 255]$ wie bei den meisten TF-Editoren.
  ],
  [
    *Konsequenz:* Dieselbe Gewebetabelle (Luft, Fett, Weichgewebe, Spongiosa,
    Kortikalis) funktioniert unverändert auf synthetischem *und* echtem
    CT-Material.
  ],
)
