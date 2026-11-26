"""Instructions: parsing (rule, LLM, hybrid), the lexicon, and applying a command exactly."""
import io
import json
import os

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
             # 2026-11-15 viewer session
             "can I see the organs without the skeleton",
             "put the heart in focus and push everything else into the background",
             "make everything except the heart semi transparent"}


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
    assert _rule("restore heart") == {"target": "heart", "attribute": None, "direction": "reset", "strength": None}


@pytest.mark.parametrize("text", ["darken the skeleton by a lot", "I wanna see a little more of the bones",
                                  "cool I want to see only lungs"])
def test_casual_fillers_are_covered_by_the_rule_parser(text):
    # each went to the LLM (1-9 s) over one uncovered word (2026-11-12 session)
    assert lexicon.uncovered_words(text, lexicon.get()) == []


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


def test_llm_one_element_target_list_is_unwrapped(monkeypatch):
    # qwen2.5:7b answered "can I see the organs without the skeleton" with
    # target ["skeleton"] on a decrease; the validator looked the list up as a
    # dict key and the server returned 500 (2026-11-15 session).
    llm_cmd = {"compound": [{"target": ["skeleton"], "attribute": "opacity", "direction": "decrease",
                             "strength": "strongly"}]}
    monkeypatch.setattr(commands, "urlopen", _fake_ollama(llm_cmd))
    cmd, meta = commands.parse_command_with_meta("can I see the organs without the skeleton", parser="hybrid")
    assert meta["parser_used"] == "llm"
    assert cmd == {"target": "skeleton", "attribute": "opacity", "direction": "decrease", "strength": "strongly"}


def test_a_list_target_outside_show_only_is_invalid_not_a_crash():
    cmd = {"target": ["skeleton", "lungs"], "attribute": "opacity", "direction": "decrease", "strength": "strongly"}
    assert commands._validate_cmd(cmd) is False


def test_llm_show_only_plus_reset_everything_else_is_show_only(monkeypatch):
    # "focus on X, hide everything else": the model appends a target-less
    # reset to the show_only, which failed validation (2026-11-15 session).
    llm_cmd = {"compound": [{"target": "heart", "attribute": None, "direction": "show_only"},
                            {"target": None, "attribute": None, "direction": "reset"}]}
    monkeypatch.setattr(commands, "urlopen", _fake_ollama(llm_cmd))
    for text in ("put the heart in focus and push everything else into the background",
                 "I want to focus on the heart, hide everything else"):
        cmd, meta = commands.parse_command_with_meta(text, parser="hybrid")
        assert meta["parser_used"] == "llm" and not meta["parser_fallback"], text
        assert cmd == {"target": "heart", "attribute": None, "direction": "show_only", "strength": None}


def test_into_the_background_is_not_the_retired_air_class():
    commands._check_retired("push the rest into the background")
    commands._check_retired("keep the skeleton in the background")
    with pytest.raises(commands.RetiredClassError):
        commands._check_retired("hide the background")


def test_everything_except_x_changes_every_other_class(monkeypatch):
    # The model listed the other classes but left out soft tissue; the answer
    # was rejected as ungrounded and the rule parser turned it into "less
    # heart" -- the opposite (2026-11-15 session).
    others = ["skeleton", "lungs", "vessels", "liver", "kidneys", "spleen"]
    llm_cmd = {"compound": [{"target": c, "attribute": "opacity", "direction": "decrease", "strength": "moderately"}
                            for c in others]}
    monkeypatch.setattr(commands, "urlopen", _fake_ollama(llm_cmd))
    cmd, meta = commands.parse_command_with_meta("make everything except the heart semi transparent", parser="hybrid")
    assert meta["parser_used"] == "llm" and not meta["parser_fallback"]
    targets = [sub["target"] for sub in cmd["compound"]]
    assert sorted(targets) == sorted(c for c in commands.CLASS_SYNONYMS if c != "heart")
    assert all(sub["direction"] == "decrease" for sub in cmd["compound"])


def test_a_repeated_instruction_keeps_the_explicit_strength():
    # "the bones are blinding, dim them heavily" says one thing twice; the
    # merge kept the first clause's default strength (2026-11-15 session).
    assert _rule("the bones are blinding, dim them heavily")["strength"] == "strongly"
    assert _rule("the liver is too faint, bring it up a lot")["strength"] == "strongly"
    assert _rule("dim the bones a bit, the bones are blinding")["strength"] == "slightly"


def _every_class(attribute, direction, strength):
    return {"compound": [{"target": c, "attribute": attribute, "direction": direction, "strength": strength}
                         for c in commands.CLASS_SYNONYMS]}


def test_global_brightness_changes_every_class():
    # "the picture is too dark overall" was refused (2026-11-15 session).
    assert _rule("the picture is too dark overall") == _every_class("brightness", "increase", "moderately")
    assert _rule("make everything a bit darker") == _every_class("brightness", "decrease", "slightly")
    assert _rule("brighten the whole image a lot") == _every_class("brightness", "increase", "strongly")
    assert _rule("darken the liver")["target"] == "liver"   # a named class stays a single command
    assert commands.describe_command(_rule("make everything a bit darker")) == "darken everything slightly"


def test_llm_global_brightness_is_every_class(monkeypatch):
    llm_cmd = {"target": None, "attribute": "brightness", "direction": "increase", "strength": "moderately"}
    monkeypatch.setattr(commands, "urlopen", _fake_ollama(llm_cmd))
    cmd = commands.parse_command_llm("the scene looks gloomy, lift it")
    assert cmd == _every_class("brightness", "increase", "moderately")


def test_hide_everything_but_x_is_show_only():
    show = lambda *t: {"target": t[0] if len(t) == 1 else list(t), "attribute": "opacity",
                       "direction": "show_only", "strength": None}
    assert _rule("hide everything but the lungs") == show("lungs")
    assert _rule("everything except the skeleton should go away") == show("skeleton")
    assert _rule("remove everything except the liver and the spleen") == show("liver", "spleen")


def test_rule_parser_refuses_an_except_it_cannot_model():
    with pytest.raises(ValueError):
        _rule("make everything except the heart semi transparent")
    assert _rule("show only the heart")["direction"] == "show_only"


def test_hybrid_falls_back_to_rules_when_the_llm_is_offline(monkeypatch):
    def offline(request, timeout):
        raise OSError("connection refused")
    monkeypatch.setattr(commands, "urlopen", offline)
    cmd, meta = commands.parse_command_with_meta("more liver mate", parser="hybrid")
    assert cmd["target"] == "liver" and meta["parser_fallback"]


def test_llm_uses_llama_server_and_a_refusal_does_not_mark_it_down(monkeypatch):
    """The speculative llama-server answers when it serves the model; a phrase
    the model calls "not an instruction" is a refusal, not a server failure."""
    calls = []

    def fake_urlopen(request, timeout):
        url = request.full_url
        calls.append(url)
        if url.endswith("/v1/models"):
            return io.BytesIO(json.dumps({"data": [{"id": commands.DEFAULT_LLM_MODEL}]}).encode())
        if url.endswith("/v1/chat/completions"):
            text = json.loads(request.data)["messages"][-1]["content"]
            answer = ({"none": True} if text == "thank you" else
                      {"target": "heart", "attribute": "opacity", "direction": "increase", "strength": "moderately"})
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps(answer)}}]}).encode())
        raise AssertionError(f"Ollama was asked: {url}")

    monkeypatch.setattr(commands, "urlopen", fake_urlopen)
    monkeypatch.setattr(commands, "_LLAMA_PROBE", {"model": None, "checked": None})
    monkeypatch.setattr(commands, "LLM_LOG_PATH", os.devnull)
    assert commands.parse_command_llm("the heart is hiding, bring it forward")["target"] == "heart"
    with pytest.raises(ValueError):
        commands.parse_command_llm("thank you")
    assert commands.llama_server_model() == commands.DEFAULT_LLM_MODEL   # still trusted
    assert sum(url.endswith("/v1/models") for url in calls) == 1


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
