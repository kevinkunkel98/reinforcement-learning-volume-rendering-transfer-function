"""The voice-command vocabulary (data/lexicon.json): load, validate, and
derive what each stage needs -- the Whisper prompt, normalization, the rule
parser's word lists and the coverage check. One file, so speech recognition,
the rule parser and the LLM can never drift onto different vocabularies."""
import functools
import json

import anatomy

LEXICON_PATH = "data/lexicon.json"
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


@functools.cache
def get() -> dict:
    """The shipped lexicon, loaded once per process."""
    return load()
