import copy

import pytest

import anatomy
import lexicon


def test_shipped_lexicon_loads_and_names_every_class():
    lex = lexicon.load()
    assert list(lex["classes"]) == list(anatomy.CANONICAL_CLASSES)
    for word in ["skeleton", "lungs", "heart", "vessels", "liver", "kidneys", "spleen", "soft tissue"]:
        assert word in lex["asr_prompt"]


def test_validate_rejects_a_phrase_in_two_entries():
    lex = copy.deepcopy(lexicon.load())
    lex["phrases"][0]["say"].append("liver")
    with pytest.raises(lexicon.LexiconError, match="'liver'"):
        lexicon.validate(lex)


def test_validate_rejects_unknown_command_fields():
    lex = copy.deepcopy(lexicon.load())
    lex["phrases"][0]["sets"]["direction"] = "explode"
    with pytest.raises(lexicon.LexiconError, match="explode"):
        lexicon.validate(lex)


def test_validate_rejects_missing_or_extra_classes():
    lex = copy.deepcopy(lexicon.load())
    del lex["classes"]["spleen"]
    with pytest.raises(lexicon.LexiconError, match="classes"):
        lexicon.validate(lex)


def test_validate_rejects_phrases_that_are_not_lowercase_and_trimmed():
    lex = copy.deepcopy(lexicon.load())
    lex["strengths"]["slightly"].append(" A Bit More")
    with pytest.raises(lexicon.LexiconError, match="A Bit More"):
        lexicon.validate(lex)


def test_strength_pattern_matches_whole_words_and_keeps_very_well_out():
    lex = lexicon.load()
    strong = lexicon.strength_pattern(lex, "strongly")
    assert strong.search("hide it completely")
    assert strong.search("very bright")
    assert not strong.search("works very well")
    assert not strong.search("farther")
    slight = lexicon.strength_pattern(lex, "slightly")
    assert slight.search("a smidge more")
