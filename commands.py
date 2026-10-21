"""Text -> command dict (two implementations) and command -> params."""
import datetime as _dt
import difflib
import json as _json
import os
import re
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np

import anatomy
import lexicon
from anatomy_layers import default_layers, normalize_layers
from transfer import (
    ANATOMICAL_CENTRES_HU, ANATOMICAL_WIDTHS_HU, CENTER_RANGE, N_PEAKS,
    PARAMS_PER_PEAK, SHOW_ONLY_MAX_WIDTH_HU, TISSUE_BANDS, TISSUE_HU,
    WIDTH_RANGE, _from_range, _from_unit, default_params, peak_internal,
)

# Vocabulary lives in data/lexicon.json (lexicon.py), shared with speech
# recognition and the LLM prompt.
_LEX = lexicon.get()
CLASS_SYNONYMS = {name: list(_LEX["classes"][name]) for name in anatomy.CANONICAL_CLASSES}
_CLASS_LOOKUP = sorted(
    ((phrase, cls) for cls, phrases in CLASS_SYNONYMS.items() for phrase in phrases),
    key=lambda t: -len(t[0]),
)

# fat/air/spongy are retired -- no goal class isolates them any more, and
# silently remapping them to something else would corrupt collected
# preference data. The rule parser rejects them outright.
RETIRED_CLASS_WORDS = {name: list(words) for name, words in _LEX["retired"].items()}
_RETIRED_LOOKUP = sorted(
    ((phrase, name) for name, phrases in RETIRED_CLASS_WORDS.items() for phrase in phrases),
    key=lambda t: -len(t[0]),
)


class RetiredClassError(ValueError):
    """A retired class (fat/air/spongy) was named: refused, never remapped."""


def _check_retired(t: str) -> None:
    for phrase, name in _RETIRED_LOOKUP:
        if re.search(rf"\b{re.escape(phrase)}\b", t):
            raise RetiredClassError(
                f"{name!r} is no longer a supported class -- the supported "
                f"classes are: {', '.join(CLASS_SYNONYMS)}"
            )


def _find_class(text: str):
    for phrase, cls in _CLASS_LOOKUP:
        if phrase in text:
            return cls
    return None


def _split_class_list(text: str) -> list:
    """'bone and lungs' / 'bone, lungs' / 'bone + lungs' -> ['skeleton', 'lungs']."""
    parts = re.split(r"\s*(?:,|\+|\band\b)\s*", text.strip())
    classes = []
    for part in parts:
        cls = _find_class(part)
        if cls and cls not in classes:
            classes.append(cls)
    return classes


_STRENGTH_MODIFIER_WORDS = {"a bit": "slightly", "a lot": "strongly", "much": "strongly"}


def _parse_relative_clause(clause: str):
    """One relative clause -> a single relative command dict, or None if the
    clause isn't one of the recognized relative phrasings."""
    m = re.search(r"\b(increase|decrease)\s+(opacity|width|sharpness|brightness)\s+(?:for|of)\s+([\w ]+)", clause)
    if m:
        direction, attr_word, target_text = m.group(1), m.group(2), m.group(3)
        attribute = ATTRIBUTE_WORD_ALIASES.get(attr_word, attr_word)
        strength = _strength_in(target_text)
        cls = _find_class(target_text)
        if cls:
            return {"target": cls, "attribute": attribute,
                    "direction": direction, "strength": strength}
        return None

    m = re.search(r"\b(?:(a bit|a lot|much)\s+)?(more|less)\s+([\w ]+)", clause)
    if m:
        modifier, verb, target_text = m.group(1), m.group(2), m.group(3)
        cls = _find_class(target_text)
        if cls:
            strength = _STRENGTH_MODIFIER_WORDS.get(modifier) or _strength_in(clause)
            direction = "increase" if verb == "more" else "decrease"
            return {"target": cls, "attribute": "opacity",
                    "direction": direction, "strength": strength}
    return None

COMMAND_REFERENCE = [
    {
        "category": "Opacity (relative)",
        "examples": ["increase opacity for skeleton strongly", "decrease opacity for lungs slightly",
                     "more bone", "a bit less soft tissue", "more heart", "more liver",
                     "more kidneys", "more spleen", "more aorta", "hide the lungs",
                     "remove the skeleton"],
        "description": "Nudge a class's visibility up or down by a relative amount. "
                        "\"hide\"/\"remove\" are a decrease at strength \"strongly\".",
    },
    {
        "category": "Opacity (absolute)",
        "examples": ["high opacity vessels", "low opacity for skeleton"],
        "description": "Set a class's visibility to a fixed low/medium/high level.",
    },
    {
        "category": "Show only",
        "examples": ["show only skeleton", "show only skeleton and lungs", "show only heart",
                     "show only vessels", "show only liver", "show only kidneys",
                     "show only spleen", "show me the lungs"],
        "description": "Isolate one or more classes, crushing every other peak to zero. "
                       "Isolation uses approximate HU bands, not anatomical guarantees; "
                       "overlapping tissues can remain visible.",
    },
    {
        "category": "Compound",
        "examples": ["high opacity vessels, low opacity skeleton",
                     "more bone, a bit less soft tissue"],
        "description": "Set several classes' levels, or nudge several classes' opacity, in one command.",
    },
    {
        "category": "Width / sharpness",
        "examples": ["sharpen the skeleton peak", "soften soft tissue", "increase width for lungs"],
        "description": "Adjust how spread out (blended) or narrow (selective) a class's peak is.",
    },
    {
        "category": "Brightness",
        "examples": ["brighten skeleton", "darken lungs", "brighten heart", "brighten liver",
                     "brighten kidneys", "brighten spleen", "low brightness for vessels"],
        "description": "Adjust a class's color brightness.",
    },
    {
        "category": "Center position",
        "examples": ["shift skeleton's center up", "move lungs down", "shift skeleton's position down"],
        "description": "Nudge where in Hounsfield space a class's peak sits, clamped to that class's own band.",
    },
    {
        "category": "Camera",
        "examples": ["rotate left", "tilt down", "zoom in"],
        "description": "Adjust the viewing angle or zoom level. Doesn't change the transfer function.",
    },
    {
        "category": "Reset",
        "examples": ["reset"],
        "description": "Return the transfer function to its default state.",
    },
]

DEFAULT_LLM_MODEL = "qwen2.5:7b"
UTTERANCE_LOG_PATH = "out/utterances.jsonl"
STRENGTH_WORDS = {"slightly": 0.15, "moderately": 0.35, "strongly": 0.6}
NEAR_THRESHOLD_HU = 300.0

DIRECT_VERBS = {
    "sharpen": ("width", "decrease", "moderately"),
    "soften": ("width", "increase", "moderately"),
    "brighten": ("brightness", "increase", "moderately"),
    "darken": ("brightness", "decrease", "moderately"),
    # There is no "hide" direction (see _SYSTEM_PROMPT): hiding a class
    # completely is "decrease" with strength "strongly". The LLM parser
    # already knows this from the prompt; the rule parser had no vocabulary
    # for "hide"/"remove" at all, so it raised on a very natural phrasing
    # instead of falling back to it.
    "hide": ("opacity", "decrease", "strongly"),
    "remove": ("opacity", "decrease", "strongly"),
    "reduce": ("opacity", "decrease", "moderately"),
}

ATTRIBUTE_WORD_ALIASES = {"sharpness": "width"}


def _normalise_text(text: str) -> str:
    t = re.sub(r"\s+", " ", text.lower().strip())
    return re.sub(r"[.!?]+$", "", t).strip()


_SLIGHT_RE = lexicon.strength_pattern(_LEX, "slightly")
_STRONG_RE = lexicon.strength_pattern(_LEX, "strongly")


def _strength_in(text: str, default: str = "moderately") -> str:
    """The strength a phrase asks for: an explicit strength word, or a
    modifier ("a bit", "a lot", "extremely"), else `default`."""
    for word in STRENGTH_WORDS:
        if re.search(rf"\b{word}\b", text):
            return word
    if _SLIGHT_RE.search(text):
        return "slightly"
    if _STRONG_RE.search(text):
        return "strongly"
    return default


def _find_classes(text: str) -> list:
    """Every class named in `text`, in order of appearance: whole words
    (plurals allowed), longest phrase first, no overlaps ("pulmonary tissue"
    is lungs, not lungs + soft)."""
    taken, found = [], []
    for phrase, cls in _CLASS_LOOKUP:
        for m in re.finditer(rf"\b{re.escape(phrase)}(?:s|es)?\b", text):
            if any(m.start() < end and start < m.end() for start, end in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((m.start(), cls))
    classes = []
    for _, cls in sorted(found):
        if cls not in classes:
            classes.append(cls)
    return classes


_SINGLE_WORD_SYNONYMS = {phrase: cls for phrase, cls in _CLASS_LOOKUP if " " not in phrase}


def _fuzzy_classes(text: str) -> list:
    """Classes behind misspelt or misheard words ("langs" -> lungs), for a
    clause that names none exactly."""
    classes = []
    for word in re.findall(r"[a-z]+", text):
        if len(word) < 4:
            continue
        close = difflib.get_close_matches(word, _SINGLE_WORD_SYNONYMS, n=1, cutoff=0.8)
        if close and _SINGLE_WORD_SYNONYMS[close[0]] not in classes:
            classes.append(_SINGLE_WORD_SYNONYMS[close[0]])
    return classes


_RESET_RE = re.compile(r"\b(reset|revert|start over|start again|from scratch|defaults?|original settings)\b")
_PRONOUN_RE = re.compile(r"\b(it|them|they|those|these|that|this)\b")
_RELATIVE_CUE_RE = re.compile(r"\b(more|less|increase\w*|decrease\w*|brighte\w*|darke\w*|lighte\w*)\b")
_SHOW_ONLY_RE = re.compile(r"\b(only|isolate|nothing but|exclusively)\b|^just\b|\b(show|display)\b(?! (more|less)\b)")
_BRIGHTNESS_RE = re.compile(r"\b(bright\w*|light(?:en|er)\w*|dark\w*|dim\w*)\b")
_DECREASE_RE = re.compile(
    r"\b(less|decrease\w*|reduc\w*|lower\w*|fade\w*|hide|hiding|remove|get rid of|eliminate|weaken\w*|"
    r"transparent|translucent|see through|dominat\w*|drown\w*|obscur\w*|block\w*|cut back|"
    r"too (?:strong|much|visible|opaque|prominent|dense))\b"
    r"|\b(?:turn|tone|dial)\w*\b.*\bdown\b|\bpull\w*\b.*\bback\b")
_INCREASE_RE = re.compile(
    r"\b(more|increase\w*|add|boost\w*|raise|bring\w*|stand out|emphasi[sz]e\w*|highlight\w*|"
    r"strengthen\w*|pop|punch\w*|enhance\w*|better|can'?t see|cannot see|barely|faint|invisible|"
    r"washed out|missing|visible|opaque|stronger)\b|\b(?:turn|dial)\w*\b.*\bup\b")
_BRIGHTNESS_DOWN_RE = re.compile(r"\b(darken\w*|darker|dim\w*|too bright|decrease\w*|reduc\w*|less|lower\w*)\b")
_BRIGHTNESS_UP_RE = re.compile(r"\b(brighten\w*|brighter|lighten\w*|lighter|too dark|increase\w*|more|raise)\b")


def _free_direction(clause: str, attribute: str):
    if attribute == "brightness":
        if _BRIGHTNESS_DOWN_RE.search(clause):
            return "decrease"
        if _BRIGHTNESS_UP_RE.search(clause):
            return "increase"
        return "decrease" if re.search(r"\b(dark|dim)", clause) else "increase"
    # Decrease cues first: "more transparent" and "less visible" are decreases.
    if _DECREASE_RE.search(clause):
        return "decrease"
    if _INCREASE_RE.search(clause):
        return "increase"
    return None


def _parse_free_clause(clause: str, previous_target=None):
    """A clause in everyday words: a class (named, misspelt, or "it"/"them"
    for the previous clause's class) plus a cue for what should happen to it --
    a level ("high spleen", "set the liver to low"), a brightness word, or an
    opacity verb or description ("add", "tone down", "barely visible",
    "dominating the view"). Returns a list of sub-commands, or None."""
    classes = _find_classes(clause) or _fuzzy_classes(clause)
    if not classes:
        if previous_target is None or not _PRONOUN_RE.search(clause):
            return None
        classes = [previous_target]
    attribute = "brightness" if _BRIGHTNESS_RE.search(clause) else "opacity"
    level = re.search(r"\b(low|medium|high)\b", clause)
    if level:
        return [{"target": c, "attribute": attribute, "direction": "set", "level": level.group(1)}
                for c in classes]
    cue = lexicon.find_cue(clause, _LEX)
    if cue and cue.get("direction") in ("increase", "decrease"):
        attribute = cue.get("attribute", attribute)
        strength = _strength_in(clause, cue.get("strength", "moderately"))
        return [{"target": c, "attribute": attribute, "direction": cue["direction"], "strength": strength}
                for c in classes]
    direction = _free_direction(clause, attribute)
    if direction is None:
        return None
    default = "strongly" if re.search(r"\b(hide|remove|get rid of|eliminate)\b", clause) else "moderately"
    strength = _strength_in(clause, default)
    return [{"target": c, "attribute": attribute, "direction": direction, "strength": strength}
            for c in classes]


def _cue_direction(t: str):
    cue = lexicon.find_cue(t, _LEX)
    return cue.get("direction") if cue else None


def _parse_show_only(t: str):
    """"show only the heart", "just the kidneys", "isolate the liver", "I only
    want to see the vessels" -> show_only over every class named."""
    if not (_SHOW_ONLY_RE.search(t) or _cue_direction(t) == "show_only") or _RELATIVE_CUE_RE.search(t):
        return None
    classes = _find_classes(t) or _fuzzy_classes(t)
    if not classes:
        return None
    return {"target": classes[0] if len(classes) == 1 else classes,
            "attribute": "opacity", "direction": "show_only", "strength": None}


def _parse_camera(t: str):
    m = re.search(r"\b(?:rotate|turn)\b(?:\s+\w+){0,3}?\s+(left|right)\b", t)
    if m:
        return {"camera": {"action": "rotate", "direction": m.group(1), "strength": _strength_in(t)}}
    m = re.search(r"\btilt\b(?:\s+\w+){0,2}?\s+(up|down)\b", t)
    if m:
        return {"camera": {"action": "tilt", "direction": m.group(1), "strength": _strength_in(t)}}
    m = re.search(r"\bzoom\s+(in|out)\b", t)
    if m:
        return {"camera": {"action": "zoom", "direction": m.group(1), "strength": _strength_in(t)}}
    return None


def _parse_grammar_clause(t: str):
    """The fixed grammar (COMMANDS.md) for one clause; a list of
    sub-commands, or None."""
    m = re.search(r"show\s+(more|less)\s+([\w ]+)", t)
    if m:
        cls = _find_class(m.group(2))
        if cls:
            return [{"target": cls, "attribute": "opacity",
                     "direction": "increase" if m.group(1) == "more" else "decrease",
                     "strength": "moderately"}]

    m = re.search(r"\b(sharpen|soften|brighten|darken|hide|remove|reduce)\b\s+([\w ]+)", t)
    if m:
        verb, target_text = m.group(1), m.group(2)
        attribute, direction, strength = DIRECT_VERBS[verb]
        cls = _find_class(target_text)
        if cls:
            return [{"target": cls, "attribute": attribute, "direction": direction,
                     "strength": _strength_in(target_text, strength)}]

    m = re.search(r"\b(?:shift|move)\s+([\w ]+?)(?:'s)?\s+(?:center|position)?\s*(up|down|higher|lower)\b", t)
    if m:
        cls = _find_class(m.group(1))
        if cls:
            direction = "increase" if m.group(2) in ("up", "higher") else "decrease"
            return [{"target": cls, "attribute": "center", "direction": direction, "strength": "moderately"}]

    # "high opacity vessels", "low opacity for skeleton", "high sharpness bone":
    # an absolute level per class+attribute, not a relative delta.
    subcommands = []
    for lm in re.finditer(r"(low|medium|high)\s+(opacity|width|sharpness|brightness)\s+(?:for\s+)?(?:the\s+)?(\w+)", t):
        cls = _find_class(lm.group(3))
        if cls:
            subcommands.append({"target": cls, "attribute": ATTRIBUTE_WORD_ALIASES.get(lm.group(2), lm.group(2)),
                                "direction": "set", "level": lm.group(1)})
    if subcommands:
        return subcommands

    single = _parse_relative_clause(t)
    return [single] if single else None


_PURPOSE_RE = re.compile(r"\s*\b(so that|so|in order to|because)\b.*$")
_CLAUSE_SPLIT_RE = re.compile(r"\s*(?:[,;+]|\band\b|\bbut\b|\bthen\b|\bplus\b|\bwhile\b)\s*")


_FILLER_RE = re.compile(r"\b(please|again|even|the|a|an|just)\b")


def _is_modifier_only(clause: str) -> bool:
    """A clause with nothing but strength, verb or filler words ("extremely",
    "darken", "a lot please") belongs to its neighbour."""
    rest = clause
    for pattern in (_SLIGHT_RE, _STRONG_RE, _DECREASE_RE, _INCREASE_RE, _BRIGHTNESS_RE, _FILLER_RE):
        rest = pattern.sub(" ", rest)
    return not rest.strip() and not _find_classes(clause)


def _merge_modifier_clauses(clauses: list) -> list:
    if len(clauses) < 2:
        return clauses
    merged, pending = [], []
    for clause in clauses:
        if _is_modifier_only(clause):
            pending.append(clause)
            continue
        merged.append(" ".join(pending + [clause]))
        pending = []
    if pending:
        if not merged:
            return [" ".join(pending)]
        merged[-1] = " ".join([merged[-1]] + pending)
    return merged


def parse_command_rule(text: str) -> dict:
    """Text -> command, without a language model.

    The fixed grammar in COMMANDS.md parses exactly as before; everyday
    phrasings ("add bones", "set the spleen to high", "the aorta is
    dominating the view, pull it back", "darken langs extremely") go through
    `_parse_free_clause`. A sentence is split into clauses (commas, "and",
    "but"); every clause must parse, or the whole command is refused rather
    than silently dropping part of what was said."""
    t = _normalise_text(text)

    _check_retired(t)

    if _RESET_RE.search(t) or _cue_direction(t) == "reset":
        return {"target": None, "attribute": None, "direction": "reset", "strength": None}

    camera = _parse_camera(t)
    if camera:
        return camera

    t = _PURPOSE_RE.sub("", t) or t
    show_only = _parse_show_only(t)
    if show_only:
        return show_only

    clauses = _merge_modifier_clauses([c for c in _CLAUSE_SPLIT_RE.split(t) if c])
    subcommands, previous = [], None
    for clause in clauses:
        parsed = _parse_grammar_clause(clause) or _parse_free_clause(clause, previous)
        if not parsed:
            if len(clauses) > 1:
                raise ValueError(f"could not parse clause {clause!r} in compound command: {text!r}")
            raise ValueError(f"rule parser cannot parse: {text!r}")
        subcommands.extend(parsed)
        previous = parsed[-1]["target"]
    # "the heart is too strong, tone it down" says one thing twice.
    unique = []
    for sub in subcommands:
        if not any((u["target"], u["attribute"], u["direction"]) == (sub["target"], sub["attribute"], sub["direction"])
                   for u in unique):
            unique.append(sub)
    return unique[0] if len(unique) == 1 else {"compound": unique}


def parse_command_with_meta(text: str, parser: str = "rule", model: str = DEFAULT_LLM_MODEL):
    """`parse_command`, plus which parser actually produced the command.

    `parse_command_llm` falls back to the rule parser whenever Ollama is
    unreachable or answers with an invalid schema, and only says so on the
    server's stdout -- so a viewer with the llm toggle on could show a rule
    parse and call it an LLM one. Returns `(cmd, meta)` with meta =
    {"parser_requested", "parser_used", "parser_fallback"}, where
    `parser_fallback` is the reason string when llm was asked for and rule
    answered, and None otherwise."""
    meta = {"parser_requested": parser, "parser_used": parser, "parser_fallback": None}
    if parser == "hybrid":
        cmd, hmeta = parse_command_hybrid(text, model=model)
        return cmd, {"parser_requested": "hybrid", "parser_used": hmeta["path"],
                     "parser_fallback": hmeta["fallback"], "normalized": hmeta["normalized"],
                     "uncovered": hmeta["uncovered"], "parse_ms": hmeta["parse_ms"]}
    if parser != "llm":
        return parse_command_rule(text), meta

    reasons = []
    cmd = parse_command_llm(text, model=model, _on_fallback=reasons.append)
    if reasons:
        meta["parser_used"] = "rule"
        meta["parser_fallback"] = reasons[0]
    return cmd, meta


def _peak_center_hu(params: np.ndarray, i: int) -> float:
    return peak_internal(params, i)["center"]


# Keep retired direct-application names working, while anatomical targets use
# the same centres and widths as their transfer peaks.
CLASS_HU = {**TISSUE_HU, **ANATOMICAL_CENTRES_HU}
CLASS_BANDS = {**TISSUE_BANDS, **{
    name: (centre - width, centre + width)
    for name, centre in ANATOMICAL_CENTRES_HU.items()
    for width in (ANATOMICAL_WIDTHS_HU[name],)
}}

_LEGACY_APPLY_PEAK_INDEX = {"bone": 3, "spongy": 2, "fat": 1, "air": 0}


def _find_or_create_peak(params: np.ndarray, tissue: str):
    """Nearest peak to the tissue's HU; reseed the weakest peak if none is near."""
    target_hu = CLASS_HU[tissue]
    if tissue in _LEGACY_APPLY_PEAK_INDEX:
        return params.copy(), _LEGACY_APPLY_PEAK_INDEX[tissue]
    centers = [_peak_center_hu(params, i) for i in range(N_PEAKS)]
    distances = [abs(c - target_hu) for c in centers]
    idx = int(np.argmin(distances))
    params = params.copy()
    if distances[idx] > NEAR_THRESHOLD_HU:
        heights = [peak_internal(params, i)["height"] for i in range(N_PEAKS)]
        idx = int(np.argmin(heights))
        base = idx * PARAMS_PER_PEAK
        params[base + 0] = _from_range(target_hu, *CENTER_RANGE)
        params[base + 1] = _from_range(80.0, *WIDTH_RANGE)
        params[base + 2] = _from_unit(0.05)
    return params, idx


LEVEL_WORDS = {"low": 0.15, "medium": 0.5, "high": 0.85}

ATTRIBUTE_PARAM_INDEX = {"opacity": 2, "width": 1, "brightness": (3, 4, 5), "center": 0}


def _asymptotic_step(current_ext: float, direction: str, delta: float) -> float:
    # Move a fraction of the remaining headroom to the +1/-1 bound, not a
    # fixed absolute amount -- generalizes the height-only fix (a fixed step
    # saturated almost any starting value in one command) to any attribute
    # stored in the same normalized external [-1, 1] encoding.
    if direction == "increase":
        return current_ext + delta * (1.0 - current_ext)
    return current_ext - delta * (current_ext + 1.0)


def _center_band_clip(ext_value: float, tissue: str) -> float:
    # A center shift must stay within the target tissue's own HU band --
    # otherwise repeated shifts could walk a peak out of its own tissue
    # entirely and into a neighboring one's territory, silently.
    lo_hu, hi_hu = CLASS_BANDS[tissue]
    lo_ext = _from_range(lo_hu, *CENTER_RANGE)
    hi_ext = _from_range(hi_hu, *CENTER_RANGE)
    return float(np.clip(ext_value, lo_ext, hi_ext))


def apply_command(cmd: dict, params: np.ndarray, reset_params=None) -> np.ndarray:
    """`cmd` applied to `params`. "reset" returns `reset_params` (the viewer's
    start TF for its task version), `default_params()` when not given."""
    if "compound" in cmd:
        # Multiple tissues set to different absolute levels in one utterance
        # ("high opacity spongy, low opacity bone") -- fold each sub-command
        # through this same function in sequence. Not a search target: an
        # absolute multi-tissue assignment isn't something to hill-climb.
        for sub in cmd["compound"]:
            params = apply_command(sub, params, reset_params)
        return params

    params = params.copy()

    if cmd["direction"] == "reset":
        return default_params() if reset_params is None else np.array(reset_params, dtype=np.float64)

    if cmd["direction"] == "show_only":
        targets = cmd["target"] if isinstance(cmd["target"], list) else [cmd["target"]]
        target_idxs = set()
        for tissue in targets:
            params, idx = _find_or_create_peak(params, tissue)
            target_idxs.add(idx)
        for idx in target_idxs:
            # A wide peak still has real opacity reaching into a neighbour's
            # HU range, so boosting it alone lights up other tissue too --
            # cap (never widen) so an already-narrow peak is untouched.
            base = idx * PARAMS_PER_PEAK
            current_width = peak_internal(params, idx)["width"]
            shown_width = SHOW_ONLY_MAX_WIDTH_HU if tissue == "bone" else min(
                current_width, SHOW_ONLY_MAX_WIDTH_HU)
            params[base + 1] = _from_range(shown_width, *WIDTH_RANGE)
        for i in range(N_PEAKS):
            b = i * PARAMS_PER_PEAK
            params[b + 2] = _from_unit(0.7 if i in target_idxs else 0.0)
        return params

    params, idx = _find_or_create_peak(params, cmd["target"])
    base = idx * PARAMS_PER_PEAK
    indices = ATTRIBUTE_PARAM_INDEX[cmd["attribute"]]
    if isinstance(indices, int):
        indices = (indices,)

    if cmd["direction"] == "set":
        if cmd["attribute"] == "center":
            raise ValueError("center has no absolute 'set' -- only increase/decrease")
        new_ext = _from_unit(LEVEL_WORDS[cmd["level"]])
        for offset in indices:
            params[base + offset] = new_ext
        return params

    delta = STRENGTH_WORDS[cmd["strength"] or "moderately"]
    for offset in indices:
        new_ext = _asymptotic_step(float(params[base + offset]), cmd["direction"], delta)
        if cmd["attribute"] == "center":
            new_ext = _center_band_clip(new_ext, cmd["target"])
        params[base + offset] = float(np.clip(new_ext, -1.0, 1.0))
    return params


def apply_command_state(cmd: dict, params: np.ndarray, layers: dict) -> tuple[np.ndarray, dict]:
    """Apply a command to HU peaks and anatomical layer state independently."""
    layer_command = cmd.get("layer_action")
    if cmd.get("attribute") != "layers" and layer_command is None:
        return apply_command(cmd, params), normalize_layers(layers)
    result = normalize_layers(layers)
    action = layer_command["action"] if layer_command else cmd["action"]
    targets = set(layer_command["targets"] if layer_command else cmd["targets"])
    if action == "restore":
        result = default_layers()
    elif action == "show_only":
        for name, layer in result.items():
            layer["opacity"] = 1.0 if name in targets else 0.0
    elif action == "hide":
        for name in targets:
            result[name]["opacity"] = 0.0
    elif action == "show":
        for name in targets:
            result[name]["opacity"] = 1.0
    else:
        raise ValueError(f"unknown layer action: {action}")
    return np.asarray(params, dtype=np.float64).copy(), result


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
VALID_DIRECTIONS = {"increase", "decrease", "show_only", "reset"}
VALID_STRENGTHS = set(STRENGTH_WORDS) | {None}
VALID_LEVELS = set(LEVEL_WORDS)
VALID_CAMERA_ACTIONS = {
    "rotate": {"left", "right"},
    "tilt": {"up", "down"},
    "zoom": {"in", "out"},
}
VALID_CAMERA_STRENGTHS = set(STRENGTH_WORDS)  # camera strength has no null case, unlike top-level strength (reset/show_only)


def _validate_camera_cmd(obj) -> bool:
    if not isinstance(obj, dict) or set(obj.keys()) != {"camera"}:
        return False
    cam = obj["camera"]
    if not isinstance(cam, dict) or set(cam.keys()) != {"action", "direction", "strength"}:
        return False
    if cam["action"] not in VALID_CAMERA_ACTIONS:
        return False
    if cam["direction"] not in VALID_CAMERA_ACTIONS[cam["action"]]:
        return False
    if cam["strength"] not in VALID_CAMERA_STRENGTHS:
        return False
    return True


def _class_synonym_lines() -> str:
    # Generated from CLASS_SYNONYMS so the LLM and the rule parser can never
    # drift onto different anatomical vocabularies.
    lines = []
    for cls, phrases in CLASS_SYNONYMS.items():
        aliases = [p for p in phrases if p != cls]
        lines.append(f'- "{cls}": also called {", ".join(aliases)}' if aliases else f'- "{cls}"')
    return "\n".join(lines)


_SYSTEM_PROMPT = f"""You convert one spoken instruction about a volume-rendering \
transfer function into strict JSON, nothing else.

Anatomical classes, with the words people actually use for them -- map any of
these back to the exact class name on the left, never invent a different one:
{_class_synonym_lines()}

Promoted anatomy has dedicated classes: heart/cardiac, liver, kidney(s)/renal,
and spleen. "vessels" includes vessel, aorta, and vena cava. "soft" covers
unpromoted organs and muscle. "fat", "air" and "spongy" are not supported.

Output schema -- the usual case is a single command:
{{"target": "<class>|[<class>, ...]|null", "attribute": "opacity"|"width"|"brightness"|"center"|null,
 "direction": "increase"|"decrease"|"show_only"|"reset",
 "strength": "slightly"|"moderately"|"strongly"|null}}

"attribute" is usually "opacity", but can also be "width" (how spread out /
sharp a class's peak is -- "sharpen"/"soften" mean decrease/increase width),
"brightness" (how light/dark a class's color is -- "brighten"/"darken" mean
increase/decrease), or "center" (where in Hounsfield space the peak sits --
"shift up"/"shift down" mean increase/decrease). These follow the same
increase/decrease/strength shape as opacity.

"target" is a list only for "show_only" when the user names more than one
class ("show skeleton and lungs" -> target: ["skeleton", "lungs"]).

direction is from the user's goal, not their wording: if something is missing,
faint, invisible, or needs more presence, that is "increase" (more of it should
be visible) even if the sentence itself sounds negative ("barely there" ->
increase). If something is hiding, dominating, or should be reduced, that is
"decrease".

Never output numeric values. "reset" has target=null, attribute=null, strength=null.
"show_only" has strength=null.

strength comes from the user's modifier words: "a bit", "a little", "slightly"
-> "slightly"; "a lot", "much", "very", "extremely", "completely" ->
"strongly"; no modifier -> "moderately". "add X" means increase X.

The four directions above are the only ones that exist. There is no "hide"
direction: hiding a class completely is "decrease" with strength "strongly",
and showing only some classes is "show_only" naming the ones to keep.

When the user gives two or more instructions in one sentence, respond with a
compound command instead: a list of single-class sub-commands, each either a
relative change (with "direction": "increase"|"decrease" and "strength", like
the single-command shape above) or an absolute level (with "direction": "set"
and a "level" field instead of "strength") -- never mix "strength" and
"level" in the same sub-command:
{{"compound": [
  {{"target": "skeleton", "attribute": "opacity", "direction": "increase", "strength": "moderately"}},
  {{"target": "soft", "attribute": "opacity", "direction": "decrease", "strength": "slightly"}}
]}}
{{"compound": [
  {{"target": "vessels", "attribute": "opacity", "direction": "set", "level": "high"}},
  {{"target": "skeleton", "attribute": "opacity", "direction": "set", "level": "low"}}
]}}
"level" is "low"|"medium"|"high" -- an absolute target, not a relative change.
"center" never takes an absolute "set"/"level" -- only increase/decrease.

Camera movement is a completely separate command shape, not a variant of
the schema above -- it has no class target at all:
{{"camera": {{"action": "rotate"|"tilt"|"zoom",
 "direction": "left"|"right" (rotate) | "up"|"down" (tilt) | "in"|"out" (zoom),
 "strength": "slightly"|"moderately"|"strongly"}}}}
Use this whenever the user wants to change the viewing angle or zoom level,
not the transfer function itself (e.g. "look from the other side", "zoom in").

Examples:
- "show more liver" -> canonical target "liver"
- "hide the kidneys" -> canonical target "kidneys"
- "brighten the spleen" -> canonical target "spleen"
- "show heart" -> canonical target "heart"
- "reduce aorta" -> canonical target "vessels"

If the input is not an instruction about the rendering at all -- small talk,
a filler word, a fragment the speech recogniser picked up ("thank you",
"okay", "bye", "you") -- respond {{"none": true}}. Never guess a command for it,
and never answer "reset" unless the user asks to reset, revert or start over.

Respond with JSON only, no prose."""


_STRENGTH_ALIASES = {"slight": "slightly", "little": "slightly", "a bit": "slightly", "a little": "slightly",
                     "moderate": "moderately", "medium": "moderately", "normal": "moderately",
                     "strong": "strongly", "very": "strongly", "extremely": "strongly", "extreme": "strongly",
                     "a lot": "strongly", "much": "strongly", "completely": "strongly"}


def _repair_single(obj, text: str):
    """Fill in what a near-miss answer left out, the way the prompt defines
    it: a missing strength (null for show_only/reset, else the text's own
    modifier), a null attribute on a relative change ("opacity"), and a
    strength word outside the schema ("extremely" -> "strongly")."""
    if not isinstance(obj, dict) or "camera" in obj:
        return obj
    obj = dict(obj)
    direction = obj.get("direction")
    if direction == "set":
        obj.pop("strength", None)
        if obj.get("attribute") is None:
            obj["attribute"] = "opacity"
        return obj
    if direction in ("show_only", "reset"):
        obj["strength"] = None
        obj.setdefault("attribute", None)
        return obj
    if direction in ("increase", "decrease"):
        if obj.get("attribute") is None:
            obj["attribute"] = "opacity"
        strength = obj.get("strength")
        if strength not in STRENGTH_WORDS:
            obj["strength"] = _STRENGTH_ALIASES.get(str(strength).lower()) if strength else None
            if obj["strength"] is None:
                obj["strength"] = _strength_in(_normalise_text(text))
    return obj


def _repair_cmd(obj, text: str):
    if isinstance(obj, dict) and isinstance(obj.get("compound"), list):
        return {"compound": [_repair_single(sub, text) for sub in obj["compound"]]}
    if isinstance(obj, dict) and obj.get("direction") == "set":
        return {"compound": [_repair_single(obj, text)]}
    return _repair_single(obj, text)


def _command_targets(cmd) -> list:
    subs = cmd["compound"] if isinstance(cmd, dict) and "compound" in cmd else [cmd]
    targets = []
    for sub in subs:
        target = sub.get("target") if isinstance(sub, dict) else None
        targets.extend(target if isinstance(target, list) else [target] if target else [])
    return targets


def _ungrounded_reason(cmd, text: str):
    """Why `cmd` is not supported by the words of `text`, or None. The model
    turned speech-recognition noise into commands ("bye" -> reset, "you" ->
    more skeleton); a command is only kept when every class it changes is
    named in the text and a reset was actually asked for."""
    t = _normalise_text(text)
    if isinstance(cmd, dict) and cmd.get("direction") == "reset" and not _RESET_RE.search(t):
        return "reset without a reset word"
    named = set(_find_classes(t)) | set(_fuzzy_classes(t))
    missing = [c for c in _command_targets(cmd) if c not in named]
    if missing:
        return f"target(s) {missing} not named in the text"
    return None


def _validate_set_cmd(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    if set(obj.keys()) != {"target", "attribute", "direction", "level"}:
        return False
    if obj["direction"] != "set":
        return False
    if obj["attribute"] == "center":
        return False
    if not isinstance(obj["target"], str) or obj["target"] not in CLASS_SYNONYMS:
        return False
    if obj["level"] not in VALID_LEVELS:
        return False
    return True


def _validate_relative_cmd(obj) -> bool:
    # A compound sub-command's relative-change shape -- same fields as a
    # top-level single command, but restricted to increase/decrease (no
    # reset/show_only/null target inside a compound).
    if not isinstance(obj, dict):
        return False
    if set(obj.keys()) != {"target", "attribute", "direction", "strength"}:
        return False
    if obj["direction"] not in ("increase", "decrease"):
        return False
    if not isinstance(obj["target"], str) or obj["target"] not in CLASS_SYNONYMS:
        return False
    if obj["strength"] not in STRENGTH_WORDS:
        return False
    return True


def _validate_single_cmd(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    if set(obj.keys()) != {"target", "attribute", "direction", "strength"}:
        return False
    if obj["direction"] not in VALID_DIRECTIONS:
        return False
    if obj["direction"] == "show_only":
        targets = obj["target"] if isinstance(obj["target"], list) else [obj["target"]]
        if not targets or any(t is None or t not in CLASS_SYNONYMS for t in targets):
            return False
    else:
        if obj["target"] is not None and obj["target"] not in CLASS_SYNONYMS:
            return False
        if obj["direction"] == "increase" or obj["direction"] == "decrease":
            if obj["target"] is None:
                return False
    if obj["strength"] not in VALID_STRENGTHS:
        return False
    return True


def _normalise_cmd(obj):
    """A compound of one sub-command is that sub-command.

    `parse_command_rule` already folds a single clause this way, but the LLM
    frequently wraps one clause in a compound regardless of the prompt. Left
    alone, the same instruction reaches `goals.goal_from_command` in two
    different shapes depending on which parser produced it.
    """
    if isinstance(obj, dict) and set(obj.keys()) == {"compound"}:
        subs = obj["compound"]
        if isinstance(subs, list) and len(subs) == 1 and isinstance(subs[0], dict):
            return subs[0]
    return obj


def _validate_cmd(obj) -> bool:
    if isinstance(obj, dict) and set(obj.keys()) == {"compound"}:
        subs = obj["compound"]
        if not isinstance(subs, list) or not subs:
            return False
        # A one-clause compound is only a wrapper -- `_normalise_cmd` unwraps
        # it right after this -- so judge it as the command it will become.
        # Without this, "show only the lungs" (which qwen2.5:7b returns as a
        # single-element compound) failed validation, because a compound
        # sub-command may only be a set/relative clause, and the viewer fell
        # back to the rule parser on a parse that was already correct.
        if len(subs) == 1:
            return _validate_single_cmd(subs[0]) or _validate_set_cmd(subs[0]) \
                or _validate_relative_cmd(subs[0])
        return all(_validate_set_cmd(sub) or _validate_relative_cmd(sub) for sub in subs)
    if isinstance(obj, dict) and set(obj.keys()) == {"camera"}:
        return _validate_camera_cmd(obj)
    return _validate_single_cmd(obj)


LLM_LOG_PATH = "out/llm_requests.jsonl"


def _log_llm_request(text, model, host, raw_response, parsed_cmd, final_cmd, fallback_reason):
    import os
    from evaluate import jsonl_append  # local import: avoids a module-load-order dependency on evaluate.py
    os.makedirs(os.path.dirname(LLM_LOG_PATH) or ".", exist_ok=True)
    jsonl_append(LLM_LOG_PATH, {
        "timestamp": _dt.datetime.now().isoformat(),
        "text": text,
        "model": model,
        "host": host,
        "raw_response": raw_response,
        "parsed_cmd": parsed_cmd,
        "final_cmd": final_cmd,
        "fell_back": fallback_reason is not None,
        "fallback_reason": fallback_reason,
    })


def parse_command_llm(text: str, model: str = DEFAULT_LLM_MODEL, host: str = None,
                       _on_fallback=None) -> dict:
    # `host` defaults at call time, not at import time, so a test (or a
    # deployment) that points OLLAMA_HOST somewhere else is honoured by
    # callers that never pass it explicitly.
    host = host if host is not None else OLLAMA_HOST
    body = _json.dumps({
        "model": model,
        "system": _SYSTEM_PROMPT,
        "prompt": text,
        "format": "json",
        "stream": False,
        # Keep the model resident between commands -- Ollama's default 5-minute
        # idle unload means any gap between commands pays an 8-10s reload cost
        # on the next one, which is what made this feel slow in practice.
        "keep_alive": "30m",
        # Parsing must be reproducible: the same instruction has to produce the
        # same command every time, or the viewer answers a retry differently
        # and a parser evaluation cannot be repeated. Ollama's default is 0.8,
        # which made the same phrase parse correctly on one run and wrongly on
        # the next -- two of the seven "failures" in the first evaluation of
        # this parser were only sampling noise.
        "options": {"temperature": 0, "seed": 0},
    }).encode()
    req = Request(f"{host}/api/generate", data=body,
                   headers={"Content-Type": "application/json"})
    try:
        # 30s, not 10s: a cold start (first request after idle-unload, or the
        # very first request of a session) can take ~8-10s on its own before
        # generation even begins -- a tight timeout risked silently falling
        # back to the rule parser on exactly the requests keep_alive can't
        # help (the first one).
        with urlopen(req, timeout=30) as resp:
            raw = _json.loads(resp.read().decode())
        raw_response = raw["response"]
        cmd = _json.loads(raw_response)
    except (URLError, OSError, ValueError, KeyError) as exc:
        print(f"[llm-parser] falling back to rule parser: {exc}")
        reason = f"request failed: {exc}"
        if _on_fallback:
            _on_fallback(reason)
        try:
            final = parse_command_rule(text)
        except ValueError:
            _log_llm_request(text, model, host, None, None, None, reason)
            raise
        _log_llm_request(text, model, host, None, None, final, reason)
        return final

    # The model sometimes echoes an alias ("bones") instead of the canonical
    # class name ("skeleton") despite the prompt asking it not to -- normalize
    # through the same synonym table the rule parser uses (`_find_class`)
    # before validating, rather than discarding an otherwise-correct answer
    # over a naming mismatch. Applies to a plain string target, each entry of
    # a list target (multi-class show_only), and each sub-command's target
    # inside a compound command. A retired class word (fat/air/spongy) has no
    # entry in CLASS_SYNONYMS, so it normalizes to itself, fails validation
    # below, and falls back to the rule parser -- which rejects it with the
    # same message `_check_retired` raises for the rule parser directly.
    def _normalize_target(value):
        if isinstance(value, str) and value not in CLASS_SYNONYMS:
            return _find_class(value) or value
        if isinstance(value, list):
            return [_normalize_target(v) for v in value]
        return value

    if isinstance(cmd, dict):
        if isinstance(cmd.get("compound"), list):
            for sub in cmd["compound"]:
                if isinstance(sub, dict) and "target" in sub:
                    sub["target"] = _normalize_target(sub["target"])
        elif "target" in cmd:
            cmd["target"] = _normalize_target(cmd["target"])

    # "Not an instruction" goes through the same fallback as every other
    # refusal: the rule parser refuses noise too, and answers what it knows.
    if isinstance(cmd, dict) and cmd.get("none"):
        reason = "model: not an instruction"
    else:
        cmd = _repair_cmd(cmd, text)
        reason = None if _validate_cmd(cmd) else f"invalid schema: {cmd!r}"
    if reason is None:
        ungrounded = _ungrounded_reason(cmd, text)
        reason = f"ungrounded: {ungrounded}: {cmd!r}" if ungrounded else None
    if reason is not None:
        print(f"[llm-parser] falling back to rule parser: {reason}")
        try:
            final = parse_command_rule(text)
        except ValueError:
            _log_llm_request(text, model, host, raw_response, cmd, None, reason)
            if _on_fallback:
                _on_fallback(reason)
            raise
        _log_llm_request(text, model, host, raw_response, cmd, final, reason)
        if _on_fallback:
            _on_fallback(reason)
        return final

    # Validate first, then unwrap: a one-clause compound is checked as the
    # compound the model actually returned, and only then folded to the shape
    # the rule parser would have produced for the same sentence.
    cmd = _normalise_cmd(cmd)
    _log_llm_request(text, model, host, raw_response, cmd, cmd, None)
    return cmd


def _log_utterance(text, meta, cmd, error, start):
    meta["parse_ms"] = round((time.perf_counter() - start) * 1000, 1)
    from evaluate import jsonl_append  # same lazy import as _log_llm_request
    os.makedirs(os.path.dirname(UTTERANCE_LOG_PATH) or ".", exist_ok=True)
    jsonl_append(UTTERANCE_LOG_PATH, {
        "timestamp": _dt.datetime.now().isoformat(), "text": text, **meta, "cmd": cmd, "error": error,
    })


def parse_command_hybrid(text: str, model: str = DEFAULT_LLM_MODEL):
    """The lexicon pipeline: normalize, answer with the rule parser when it
    parses and every word is accounted for, else ask the LLM (which falls back
    to the rule parser itself). Returns (cmd, meta) with meta = {"normalized",
    "uncovered", "path" ("rule"|"llm"), "fallback", "parse_ms"}; raises
    ValueError (RetiredClassError for fat/air/spongy) when nothing parses."""
    start = time.perf_counter()
    normalized = lexicon.normalize(text, _LEX)
    meta = {"normalized": normalized, "uncovered": lexicon.uncovered_words(normalized, _LEX),
            "path": None, "fallback": None}
    rule_cmd = None
    try:
        rule_cmd = parse_command_rule(normalized)
    except RetiredClassError as exc:
        meta["path"] = "rejected"
        _log_utterance(text, meta, None, str(exc), start)
        raise
    except ValueError:
        pass
    if rule_cmd is not None and not meta["uncovered"]:
        meta["path"] = "rule"
        _log_utterance(text, meta, rule_cmd, None, start)
        return rule_cmd, meta

    reasons = []
    try:
        cmd = parse_command_llm(normalized, model=model, _on_fallback=reasons.append)
    except ValueError as exc:
        reason = reasons[0] if reasons else str(exc)
        meta["path"], meta["fallback"] = "rejected", reason
        message = (f"LLM offline and the rule parser cannot parse {text!r}"
                   if reason.startswith("request failed") else f"could not parse {text!r}")
        _log_utterance(text, meta, None, message, start)
        raise ValueError(message) from exc
    meta["path"] = "rule" if reasons else "llm"
    meta["fallback"] = reasons[0] if reasons else None
    _log_utterance(text, meta, cmd, None, start)
    return cmd, meta
