"""The voice-command vocabulary (data/lexicon.json): load, validate, and
derive what each stage needs -- the Whisper prompt, normalization, the rule
parser's word lists and the coverage check. One file, so speech recognition,
the rule parser and the LLM can never drift onto different vocabularies."""
import difflib
import functools
import json
import os
import re

import anatomy

LEXICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "lexicon.json")
COMMAND_FIELDS = {
    "attribute": {"opacity", "width", "brightness", "center"},
    "direction": {"increase", "decrease", "show_only", "reset"},
    "strength": {"slightly", "moderately", "strongly"},
}


class LexiconError(ValueError):
    pass


def _check_phrase(phrase: str, where: str) -> None:
    if not isinstance(phrase, str) or phrase != phrase.strip().lower() or not phrase:
        raise LexiconError(f"{where}: phrase {phrase!r} must be lowercase and trimmed")


def validate(lex: dict) -> dict:
    """Raise LexiconError on anything the stages cannot use; return `lex`."""
    if list(lex.get("classes", {})) != list(anatomy.CANONICAL_CLASSES):
        raise LexiconError(f"classes must be exactly {list(anatomy.CANONICAL_CLASSES)} in that order")
    owner = {}

    def claim(phrase, where):
        _check_phrase(phrase, where)
        if phrase in owner:
            raise LexiconError(f"{phrase!r} is in both {owner[phrase]} and {where}")
        owner[phrase] = where

    for cls, terms in lex["classes"].items():
        for term in terms:
            claim(term, f"classes.{cls}")
    for name, terms in lex["retired"].items():
        for term in terms:
            claim(term, f"retired.{name}")
    for i, entry in enumerate(lex["phrases"]):
        for key, value in entry["sets"].items():
            if key not in COMMAND_FIELDS or value not in COMMAND_FIELDS[key]:
                raise LexiconError(f"phrases[{i}]: {key}={value!r} is not a command field value")
        for phrase in entry["say"]:
            claim(phrase, f"phrases[{i}]")
    for strength, words in lex["strengths"].items():
        if strength not in COMMAND_FIELDS["strength"]:
            raise LexiconError(f"strengths: unknown strength {strength!r}")
        for word in words:
            claim(word, f"strengths.{strength}")
    for wrong in lex["asr_fixes"]:
        claim(wrong, "asr_fixes")
    for key in ("filler", "grammar"):
        for word in lex[key]:
            _check_phrase(word, key)
    if not isinstance(lex.get("asr_prompt"), str) or not lex["asr_prompt"]:
        raise LexiconError("asr_prompt must be a non-empty string")
    return lex


def _cached(lex: dict, name: str, build):
    """Derived tables, built once per lexicon dict and stored on it (under
    "_cache", which validate() ignores)."""
    cache = lex.setdefault("_cache", {})
    if name not in cache:
        cache[name] = build(lex)
    return cache[name]


def load(path: str = LEXICON_PATH) -> dict:
    with open(path) as f:
        return validate(json.load(f))


def _cue_patterns(lex: dict):
    pairs = [(phrase, entry["sets"]) for entry in lex["phrases"] for phrase in entry["say"]]
    pairs.sort(key=lambda p: -len(p[0]))
    return [(re.compile(rf"\b{re.escape(p)}(?:s|es|ed|ing)?\b"), sets, p) for p, sets in pairs]


def find_cue_match(text: str, lex: dict):
    """(sets, start, phrase) for the longest lexicon phrase in `text`
    (lowercase): its command fields, where it starts and which phrase it was;
    or None."""
    for pattern, sets, phrase in _cached(lex, "cues", _cue_patterns):
        m = pattern.search(text)
        if m:
            return dict(sets), m.start(), phrase
    return None


def find_cue(text: str, lex: dict):
    """The command fields of the longest lexicon phrase in `text` (lowercase),
    or None."""
    match = find_cue_match(text, lex)
    return match[0] if match else None


@functools.cache
def get() -> dict:
    """The shipped lexicon, loaded once per process."""
    return load()


def strength_pattern(lex: dict, strength: str) -> re.Pattern:
    """Whole-word regex over a strength's words, longest first. "very" is
    kept out of "very well" (as the hand-written regex did)."""
    words = sorted(lex["strengths"][strength], key=len, reverse=True)
    alternatives = [r"very(?! well)" if w == "very" else re.escape(w) for w in words]
    return re.compile(rf"\b({'|'.join(alternatives)})\b")


_PHONETIC = str.maketrans({"k": "g", "c": "g", "q": "g", "t": "d", "p": "b", "f": "v", "w": "v",
                           "z": "s", "x": "s"})


def phonetic_key(word: str) -> str:
    """Crude sound key: voiced/voiceless pairs merged, vowels after the first
    letter dropped, doubled letters collapsed ("sgeleton" == "skeleton")."""
    w = word.lower().translate(_PHONETIC)
    key = w[:1] + re.sub(r"[aeiouyh']", "", w[1:])
    return re.sub(r"(.)\1+", r"\1", key)


def _words(phrases):
    return {w for p in phrases for w in re.findall(r"[a-z]+(?:'[a-z]+)?", p)}


def _vocab(lex: dict):
    content = _words([t for terms in lex["classes"].values() for t in terms]
                     + [p for e in lex["phrases"] for p in e["say"]]
                     + [w for ws in lex["strengths"].values() for w in ws])
    known = content | _words([t for ts in lex["retired"].values() for t in ts]) \
        | _words(lex["filler"]) | _words(lex["grammar"]) | _words(lex["asr_fixes"].values())
    return sorted(w for w in content if len(w) >= 4), known


def known_words(lex: dict) -> set:
    return _cached(lex, "vocab", _vocab)[1]


def _fix_word(word: str, lex: dict):
    """The lexicon word `word` was most likely meant as, or None to keep it."""
    w = word.lower().strip("'")
    if w in lex["asr_fixes"]:
        return lex["asr_fixes"][w]
    targets, known = _cached(lex, "vocab", _vocab)
    if w in known or len(w) < 4:
        return None
    best = difflib.get_close_matches(w, targets, n=1, cutoff=0.7)
    if not best:
        return None
    ratio = difflib.SequenceMatcher(None, w, best[0]).ratio()
    if ratio >= 0.88 or phonetic_key(w) == phonetic_key(best[0]):
        return best[0]
    return None


def normalize(text: str, lex: dict) -> str:
    """Rewrite known transcription errors and near-misses of lexicon words;
    everything else, including punctuation, stays as it was."""
    def repl(m):
        fixed = _fix_word(m.group(0), lex)
        return fixed if fixed is not None else m.group(0)

    return re.sub(r"[A-Za-z]+(?:'[A-Za-z]+)?'?", repl, text)


def uncovered_words(text: str, lex: dict) -> list:
    """Words of `text` the lexicon does not account for, in order. A word
    counts as covered with a plural/verb suffix too ("lungs", "hiding")."""
    known = known_words(lex)
    left = []
    for w in re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower()):
        stems = {w} | {w[: -len(s)] for s in ("s", "es", "ed", "ing", "ly", "er") if w.endswith(s)}
        if not stems & known:
            left.append(w)
    return left
