#import "../helpers.typ": *
#import "../diagrams.typ": tf-curve-diagram

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — Die Transferfunktion
// ══════════════════════════════════════════════════════════════════════════════
== Die Transferfunktion: 24 Werte

#v(0.15em)
#align(center)[
  #box(width: 100%, scale(x: 78%, y: 78%, reflow: true)[#tf-curve-diagram])
]

#v(0.1em)
#align(center)[
  $ alpha(x) = op("clip")(sum_(i=1)^4 h_i dot exp(-1/2 ((x-c_i)/w_i)^2), thin 0, 1) $
]

#v(0.2em)
#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  [
    *Design-Entscheidung:* Zentren $c_i$ und Breiten $w_i$ liegen in echten
    #text(fill: navy, weight: "bold")[Hounsfield-Einheiten] (HU) — nicht in
    $[0, 255]$ wie bei den meisten TF-Editoren.
  ],
  [
    *Konsequenz:* Dieselbe Gewebetabelle (Luft, Fett, Weichgewebe, Spongiosa,
    Kortikalis) funktioniert unverändert auf synthetischem, CT- *und*
    MRT-Material (linear auf denselben HU-Bereich reskaliert).
  ],
)
