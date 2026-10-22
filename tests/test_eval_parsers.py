import json

import eval_parsers


def _rel(target, direction, strength="moderately", attribute="opacity"):
    return {"target": target, "attribute": attribute, "direction": direction, "strength": strength}


def test_matches_ignores_compound_order_and_show_only_attribute():
    a, b = _rel("liver", "increase"), _rel("skeleton", "decrease")
    assert eval_parsers.matches({"compound": [a, b]}, {"compound": [b, a]})
    assert eval_parsers.matches({"target": ["heart", "lungs"], "attribute": "opacity", "direction": "show_only",
                                 "strength": None},
                                {"target": ["lungs", "heart"], "attribute": None, "direction": "show_only",
                                 "strength": None})


def test_matches_accepts_any_listed_strength_and_intent_ignores_strength():
    expected = _rel("lungs", "increase", ["moderately", "strongly"])
    assert eval_parsers.matches(expected, _rel("lungs", "increase", "strongly"))
    assert not eval_parsers.matches(expected, _rel("lungs", "increase", "slightly"))
    assert eval_parsers.matches(expected, _rel("lungs", "increase", "slightly"), with_strength=False)
    assert not eval_parsers.matches(expected, _rel("lungs", "decrease"), with_strength=False)


def test_a_refusal_is_correct_only_when_none_is_expected():
    assert eval_parsers.matches(None, None)
    assert not eval_parsers.matches(_rel("lungs", "increase"), None)
    assert not eval_parsers.matches(None, _rel("lungs", "increase"))


def test_eval_set_is_well_formed():
    with open(eval_parsers.EVAL_SET_PATH) as f:
        rows = json.load(f)
    assert {r["split"] for r in rows} == {"dev", "heldout"}
    assert len({r["text"] for r in rows}) == len(rows)


def test_realistic_eval_set_is_well_formed():
    with open(eval_parsers.EVAL_SETS["realistic"]) as f:
        rows = json.load(f)
    assert {r["split"] for r in rows} == {"dev", "heldout"}
    assert len({r["text"] for r in rows}) == len(rows)
    assert all(set(r) == {"text", "expected", "split", "source"} for r in rows)
    assert {r["source"] for r in rows} <= {"log-2026-10-21", "log-2026-10-22", "claude", "author", "colleague"}
    # both outcomes are represented: commands and refusals
    assert any(r["expected"] is None for r in rows if r["split"] == "heldout")
    assert any(r["expected"] is not None for r in rows if r["split"] == "heldout")
