#import "../helpers.typ": *

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE — Der Prototyp
// ══════════════════════════════════════════════════════════════════════════════
== Der Prototyp

#grid(
  columns: (1.35fr, 1fr),
  gutter: 1.4em,
  align: (left + horizon, left + horizon),
  [
    #image("../screenshot_ui.png", width: 100%)
  ],
  [
    *Lokal, offline-first:*
    - Chat-Oberfläche (FastAPI + Vanilla JS)
    - Text *und* Sprache (faster-whisper, lokal)
    - Regel- *und* LLM-Parser (Ollama, lokal)
    - Sofortiges Rendering nach jedem Kommando

    #v(0.4em)
    *Für RL-Daten von Anfang an gebaut:*
    - Jeder Schritt, jedes Bild, jede LLM-Anfrage wird geloggt
    - 👍/👎-Feedback pro Antwort
    - Verzweigbare Historie (zurück & neu verzweigen)
  ],
)
