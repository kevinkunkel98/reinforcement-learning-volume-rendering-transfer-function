import copy
import json

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


def test_find_cue_prefers_the_longest_phrase():
    lex = lexicon.load()
    assert lexicon.find_cue("make the bones more opaque", lex) == {"attribute": "opacity", "direction": "increase"}
    assert lexicon.find_cue("softer edges on the bones", lex) == {"attribute": "width", "direction": "increase"}
    assert lexicon.find_cue("make the lungs see-through", lex)["direction"] == "decrease"
    assert lexicon.find_cue("add kidneys", lex) is None


@pytest.mark.parametrize("raw, fixed", [
    ("Darkin' Skeleton", "darken skeleton"),
    ("Leitenlangs", "lighten lungs"),
    ("Highlight the art.", "highlight the heart."),
    ("Dial the arc down the touch.", "dial the heart down the touch."),
    ("Make the ripsage pop.", "make the ribcage pop."),
    ("leiten lungs", "lighten lungs"),
    ("darken the sgeleton", "darken the skeleton"),
    ("show only the kitneys", "show only the kidneys"),
    ("make the lever more visible", "make the liver more visible"),
])
def test_normalize_fixes_transcription_errors(raw, fixed):
    assert lexicon.normalize(raw, lexicon.load()).lower() == fixed


@pytest.mark.parametrize("text", [
    "make it look cool",
    "thank you for watching",
    "can you show me just the heart please",
    "the muscles are hiding everything, get rid of them",
    "more aorta, less muscle",
    "how long is it",
    "spin left",
    "check mate",
    "the heat is too strong",
    "I heard you",
    "a long way",
])
def test_normalize_leaves_ordinary_sentences_alone(text):
    assert lexicon.normalize(text, lexicon.load()) == text


def test_phonetic_key_merges_voiced_and_voiceless_consonants():
    assert lexicon.phonetic_key("sgeleton") == lexicon.phonetic_key("skeleton")
    assert lexicon.phonetic_key("kitneys") == lexicon.phonetic_key("kidneys")
    assert lexicon.phonetic_key("lever") == lexicon.phonetic_key("liver")


def test_uncovered_words_lists_content_words_outside_the_lexicon():
    lex = lexicon.load()
    assert lexicon.uncovered_words("show only the bones please", lex) == []
    assert lexicon.uncovered_words("make the lungs see-through", lex) == []
    assert lexicon.uncovered_words("everything except the skeleton should go away", lex) == ["except", "go", "away"]


def test_find_cue_match_reports_where_and_which_phrase_matched():
    lex = lexicon.load()
    sets, start, phrase = lexicon.find_cue_match("make the lungs less transparent", lex)
    assert sets["direction"] == "decrease" and phrase == "transparent"
    assert "make the lungs less transparent"[start:].startswith("transparent")
    assert lexicon.find_cue_match("add kidneys", lex) is None


def test_keep_words_are_never_rewritten_and_stay_uncovered():
    lex = lexicon.load()
    for word in ["heard", "long", "spin", "border", "lover"]:
        assert word in lex["keep"]
        assert lexicon.normalize(word, lex) == word
        assert lexicon.uncovered_words(word, lex) == [word]


def test_validate_rejects_keep_words_that_are_not_lowercase_and_trimmed():
    lex = copy.deepcopy(lexicon.load())
    lex["keep"].append("Heard ")
    with pytest.raises(lexicon.LexiconError, match="Heard"):
        lexicon.validate(lex)


def test_derived_tables_stay_out_of_the_lexicon_dict():
    lex = lexicon.get()
    lexicon.find_cue("make the bones solid", lex)
    lexicon.known_words(lex)
    dumped = json.dumps(lex)
    assert "_cache" not in json.loads(dumped)
    assert isinstance(lexicon.known_words(lex), frozenset)


def test_a_modified_copy_gets_its_own_derived_tables():
    lex = lexicon.get()
    lexicon.find_cue("anything", lex)
    other = copy.deepcopy(lex)
    other["phrases"][0]["say"].append("vaporize")
    assert lexicon.find_cue("vaporize the bones", other)["direction"] == "decrease"
    assert lexicon.find_cue("vaporize the bones", lex) is None


@pytest.mark.parametrize("word, covered", [
    ("ones", False), ("used", False), ("shower", False),
    ("lungs", True), ("hiding", True), ("bones", True),
])
def test_suffix_stripping_needs_a_real_stem(word, covered):
    assert (lexicon.uncovered_words(word, lexicon.load()) == []) == covered
