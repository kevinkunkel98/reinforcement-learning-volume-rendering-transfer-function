"""Instructions: parsing (rule, LLM, hybrid), the lexicon, and applying a command exactly."""
import io
import json

import numpy as np
import pytest

import commands
import goals
import lexicon
import transfer
from eval_parsers import matches

# Realistic spoken phrasings the rule parser alone does not cover (the hybrid
# parser sends them to the LLM).
NEEDS_LLM = {"Leitenlangs", "Revered skeleton.", "the muscles are hiding everything, get rid of them",
             "everything except the skeleton should go away"}


def _eval_set(name):
    with open(f"data/{name}", encoding="utf-8") as stream:
        return json.load(stream)


def _rule(text):
    return commands.parse_command_with_meta(text, parser="rule")[0]


@pytest.mark.parametrize("name", ["parser_eval_set.json", "parser_eval_realistic.json"])
def test_rule_parser_on_the_eval_sets(name):
    wrong = []
    for row in _eval_set(name):
        try:
            cmd = _rule(row["text"])
            ok = row["expected"] is not None and matches(row["expected"], cmd)
        except ValueError:
            ok = row["expected"] is None
        if not ok:
            wrong.append(row["text"])
    assert set(wrong) <= NEEDS_LLM, wrong


@pytest.mark.parametrize("text, expected", [
    ("a bit more liver", {"target": "liver", "attribute": "opacity", "direction": "increase", "strength": "slightly"}),
    ("less lungs", {"target": "lungs", "attribute": "opacity", "direction": "decrease", "strength": "moderately"}),
    ("show only the skeleton", {"target": "skeleton", "direction": "show_only"}),
    ("make the heart darker", {"target": "heart", "attribute": "brightness", "direction": "decrease"}),
    ("the spleen is way too dark", {"target": "spleen", "attribute": "brightness", "direction": "increase"}),
])
def test_rule_parser_phrases(text, expected):
    cmd = _rule(text)
    assert {k: cmd.get(k) for k in expected} == expected


def test_compound_and_reset():
    cmd = _rule("more liver and less lungs")
    assert [(c["target"], c["direction"]) for c in cmd["compound"]] == [("liver", "increase"), ("lungs", "decrease")]
    assert _rule("reset")["direction"] == "reset"
    assert _rule("reset the heart") == {"target": "heart", "attribute": None, "direction": "reset", "strength": None}


def test_unparseable_and_retired_words_raise():
    with pytest.raises(ValueError):
        _rule("what a lovely day")
    with pytest.raises(commands.RetiredClassError):
        _rule("more fat")


def test_lexicon_normalizes_known_mishearings_but_not_english():
    lex = lexicon.get()
    assert lexicon.normalize("Darkin' Skeleton", lex).lower() == "darken skeleton"
    assert lexicon.normalize("the heat is on", lex) == "the heat is on"
    assert lexicon.uncovered_words("more liver please", lex) == []


def _fake_ollama(response):
    def fake_urlopen(request, timeout):
        return io.BytesIO(json.dumps({"response": json.dumps(response)}).encode())
    return fake_urlopen


def test_hybrid_uses_rules_for_covered_phrases_and_the_llm_for_the_rest(monkeypatch):
    llm_cmd = {"target": "heart", "attribute": "opacity", "direction": "increase", "strength": "moderately"}
    monkeypatch.setattr(commands, "urlopen", _fake_ollama(llm_cmd))
    _cmd, meta = commands.parse_command_with_meta("more liver", parser="hybrid")
    assert meta["parser_used"] == "rule"
    cmd, meta = commands.parse_command_with_meta("the heart is hiding, bring it forward", parser="hybrid")
    assert meta["parser_used"] == "llm" and cmd["target"] == "heart"


def test_hybrid_falls_back_to_rules_when_the_llm_is_offline(monkeypatch):
    def offline(request, timeout):
        raise OSError("connection refused")
    monkeypatch.setattr(commands, "urlopen", offline)
    cmd, meta = commands.parse_command_with_meta("more liver mate", parser="hybrid")
    assert cmd["target"] == "liver" and meta["parser_fallback"]


def _height(params, name):
    return params[transfer.ANATOMICAL_PEAK_INDEX[name] * transfer.PARAMS_PER_PEAK + 2]


def test_apply_command_moves_only_the_named_peak():
    start = goals.starting_params()
    after = commands.apply_command(_rule("more liver"), start)
    changed = np.flatnonzero(after != start)
    base = transfer.ANATOMICAL_PEAK_INDEX["liver"] * transfer.PARAMS_PER_PEAK
    assert set(changed) == {base + 2} and _height(after, "liver") > _height(start, "liver")


def test_show_only_and_class_reset():
    start = goals.starting_params()
    shown = commands.apply_command(_rule("show only the lungs"), start)
    assert _height(shown, "lungs") > _height(shown, "liver") == -1.0
    restored = commands.apply_command(_rule("reset the liver"), shown, reset_params=start)
    base = transfer.ANATOMICAL_PEAK_INDEX["liver"] * transfer.PARAMS_PER_PEAK
    assert np.array_equal(restored[base:base + 6], start[base:base + 6])


def test_brighten_keeps_the_hue():
    start = goals.starting_params()
    for name in ("heart", "liver", "lungs"):
        after = commands.apply_command(_rule(f"brighten the {name}"), start)
        base = transfer.ANATOMICAL_PEAK_INDEX[name] * transfer.PARAMS_PER_PEAK
        before_rgb, after_rgb = (transfer._unit(p[base + 3:base + 6]) for p in (start, after))
        assert after_rgb.sum() > before_rgb.sum()
        assert np.argmax(after_rgb) == np.argmax(before_rgb)   # red stays red, even where it saturates


def test_describe_command():
    assert commands.describe_command(_rule("a bit more liver")) == "a bit more liver"


def test_whisper_hallucinations_are_dropped():
    from asr import is_hallucination
    assert is_hallucination("Thanks for watching!") and is_hallucination("you")
    assert not is_hallucination("show only the skeleton")
