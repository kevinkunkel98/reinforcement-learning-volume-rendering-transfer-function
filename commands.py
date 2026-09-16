"""Text -> command dict (two implementations) and command -> params."""
import datetime as _dt
import json as _json
import os
import re
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np

from transfer import (
    CENTER_RANGE, N_PEAKS, PARAMS_PER_PEAK, TISSUE_BANDS, TISSUE_HU,
    WIDTH_RANGE, _from_range, _from_unit, _unit, default_params, peak_internal,
)

# --- goal-class vocabulary: the four RL v2 goal classes ---------------------
# Both the rule parser and the LLM parser (prompt + validator) speak the same
# four anatomical goal classes the trained policy does. An organ name maps to
# "soft" because no transfer function can isolate one organ from the rest of
# soft tissue -- the parser must not promise what the renderer cannot deliver.
CLASS_SYNONYMS = {
    "skeleton": ["bone", "bones", "skeleton", "ribs", "rib", "spine", "vertebrae",
                 "hip", "femur", "skull"],
    "lungs": ["lung", "lungs", "pulmonary"],
    "soft": ["soft tissue", "soft", "organs", "organ", "muscle", "muscles",
             "liver", "kidney", "spleen"],
    "vessels": ["vessel", "vessels", "artery", "arteries", "vein", "veins",
                "aorta", "contrast"],
}
_CLASS_LOOKUP = sorted(
    ((phrase, cls) for cls, phrases in CLASS_SYNONYMS.items() for phrase in phrases),
    key=lambda t: -len(t[0]),
)

# fat/air/spongy are retired -- no goal class isolates them any more, and
# silently remapping them to something else would corrupt collected
# preference data. The rule parser rejects them outright.
RETIRED_CLASS_WORDS = {
    "fat": ["fatty", "fat", "adipose"],
    "air": ["air", "background"],
    "spongy": ["spongy bone", "spongy", "cancellous", "trabecular"],
}
_RETIRED_LOOKUP = sorted(
    ((phrase, name) for name, phrases in RETIRED_CLASS_WORDS.items() for phrase in phrases),
    key=lambda t: -len(t[0]),
)


def _check_retired(t: str) -> None:
    for phrase, name in _RETIRED_LOOKUP:
        if re.search(rf"\b{re.escape(phrase)}\b", t):
            raise ValueError(
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
    m = re.search(r"\b(increase|decrease)\s+(opacity|width|sharpness|brightness)\s+for\s+([\w ]+)", clause)
    if m:
        direction, attr_word, target_text = m.group(1), m.group(2), m.group(3)
        attribute = ATTRIBUTE_WORD_ALIASES.get(attr_word, attr_word)
        strength = "moderately"
        for word in STRENGTH_WORDS:
            if word in target_text:
                strength = word
                target_text = target_text.replace(word, "")
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
            strength = _STRENGTH_MODIFIER_WORDS.get(modifier, "moderately")
            direction = "increase" if verb == "more" else "decrease"
            return {"target": cls, "attribute": "opacity",
                    "direction": direction, "strength": strength}
    return None

COMMAND_REFERENCE = [
    {
        "category": "Opacity (relative)",
        "examples": ["increase opacity for skeleton strongly", "decrease opacity for lungs slightly",
                     "more bone", "a bit less soft tissue"],
        "description": "Nudge a class's visibility up or down by a relative amount.",
    },
    {
        "category": "Opacity (absolute)",
        "examples": ["high opacity vessels", "low opacity for skeleton"],
        "description": "Set a class's visibility to a fixed low/medium/high level.",
    },
    {
        "category": "Show only",
        "examples": ["show only skeleton", "show only skeleton and lungs", "show me the lungs"],
        "description": "Isolate one or more classes, crushing every other peak to zero.",
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
        "examples": ["brighten skeleton", "darken lungs", "low brightness for vessels"],
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

STRENGTH_WORDS = {"slightly": 0.15, "moderately": 0.35, "strongly": 0.6}
NEAR_THRESHOLD_HU = 300.0

DIRECT_VERBS = {
    "sharpen": ("width", "decrease"),
    "soften": ("width", "increase"),
    "brighten": ("brightness", "increase"),
    "darken": ("brightness", "decrease"),
}

ATTRIBUTE_WORD_ALIASES = {"sharpness": "width"}


def parse_command_rule(text: str) -> dict:
    t = text.lower().strip()

    _check_retired(t)

    if "reset" in t:
        return {"target": None, "attribute": None, "direction": "reset", "strength": None}

    m = re.search(r"show(?:\s+only|\s+me(?:\s+the)?)\s+([\w ,\+]+)", t)
    if m:
        classes = _split_class_list(m.group(1))
        if classes:
            return {"target": classes[0] if len(classes) == 1 else classes,
                     "attribute": "opacity", "direction": "show_only", "strength": None}

    m = re.search(r"\b(sharpen|soften|brighten|darken)\b\s+([\w ]+)", t)
    if m:
        verb, target_text = m.group(1), m.group(2)
        attribute, direction = DIRECT_VERBS[verb]
        strength = "moderately"
        for word in STRENGTH_WORDS:
            if word in target_text:
                strength = word
                target_text = target_text.replace(word, "")
        cls = _find_class(target_text)
        if cls:
            return {"target": cls, "attribute": attribute,
                     "direction": direction, "strength": strength}

    m = re.search(r"\b(?:shift|move)\s+([\w ]+?)(?:'s)?\s+(?:center|position)?\s*(up|down|higher|lower)\b", t)
    if m:
        target_text, word = m.group(1), m.group(2)
        cls = _find_class(target_text)
        if cls:
            direction = "increase" if word in ("up", "higher") else "decrease"
            return {"target": cls, "attribute": "center",
                     "direction": direction, "strength": "moderately"}

    m = re.search(r"\b(rotate|turn)\s+(left|right)\b", t)
    if m:
        strength = "moderately"
        for word in STRENGTH_WORDS:
            if word in t:
                strength = word
        return {"camera": {"action": "rotate", "direction": m.group(2), "strength": strength}}

    m = re.search(r"\btilt\s+(up|down)\b", t)
    if m:
        strength = "moderately"
        for word in STRENGTH_WORDS:
            if word in t:
                strength = word
        return {"camera": {"action": "tilt", "direction": m.group(1), "strength": strength}}

    m = re.search(r"\bzoom\s+(in|out)\b", t)
    if m:
        strength = "moderately"
        for word in STRENGTH_WORDS:
            if word in t:
                strength = word
        return {"camera": {"action": "zoom", "direction": m.group(1), "strength": strength}}

    # "high opacity vessels", "low opacity for skeleton", "high sharpness bone" --
    # an absolute level per class+attribute, not a relative delta. One or
    # more may appear in the same sentence, each becomes its own
    # sub-command, folded together into one compound command.
    level_matches = list(re.finditer(
        r"(low|medium|high)\s+(opacity|width|sharpness|brightness)\s+(?:for\s+)?(\w+)", t))
    if level_matches:
        subcommands = []
        for lm in level_matches:
            level, attr_word, class_text = lm.group(1), lm.group(2), lm.group(3)
            attribute = ATTRIBUTE_WORD_ALIASES.get(attr_word, attr_word)
            cls = _find_class(class_text)
            if cls:
                subcommands.append({"target": cls, "attribute": attribute,
                                      "direction": "set", "level": level})
        if subcommands:
            return subcommands[0] if len(subcommands) == 1 else {"compound": subcommands}

    # Relative clauses: "increase opacity for skeleton strongly", "more bone",
    # "a bit less soft tissue". Several comma/semicolon-separated clauses
    # become a compound of all of them; if some clauses in a multi-clause
    # sentence parse as relative commands and others don't, that's an error
    # -- never silently drop a clause the user actually said.
    clauses = [c.strip() for c in re.split(r"\s*[,;]\s*", t) if c.strip()]
    if len(clauses) > 1:
        parsed = [_parse_relative_clause(c) for c in clauses]
        if any(p is not None for p in parsed):
            if all(p is not None for p in parsed):
                return {"compound": parsed}
            bad = clauses[parsed.index(None)]
            raise ValueError(f"could not parse clause {bad!r} in compound command: {text!r}")
    else:
        single = _parse_relative_clause(t)
        if single:
            return single

    raise ValueError(f"rule parser cannot parse: {text!r}")


def parse_command(text: str, parser: str = "rule", model: str = "qwen2.5:7b") -> dict:
    if parser == "llm":
        return parse_command_llm(text, model=model)
    return parse_command_rule(text)


def _peak_center_hu(params: np.ndarray, i: int) -> float:
    return peak_internal(params, i)["center"]


# apply_command's peak-placement vocabulary: the legacy single-tissue names
# (bone, spongy, fat, air, soft -- still used directly by apply_command's own
# unit tests) plus the four anatomical goal classes both parsers now speak.
# "soft" and "bone"/"spongy"'s HU already coincide with
# "soft"/"skeleton"/"vessels", so those just reuse the same peak; only
# "lungs" has no legacy equivalent.
CLASS_HU = {**TISSUE_HU, "lungs": -800.0, "vessels": 300.0, "skeleton": 900.0}
CLASS_BANDS = {**TISSUE_BANDS, "lungs": (-1050.0, -550.0),
                "vessels": (170.0, 600.0), "skeleton": (600.0, 2000.0)}


def _find_or_create_peak(params: np.ndarray, tissue: str):
    """Nearest peak to the tissue's HU; reseed the weakest peak if none is near."""
    target_hu = CLASS_HU[tissue]
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


def apply_command(cmd: dict, params: np.ndarray) -> np.ndarray:
    if "compound" in cmd:
        # Multiple tissues set to different absolute levels in one utterance
        # ("high opacity spongy, low opacity bone") -- fold each sub-command
        # through this same function in sequence. Not a search target: an
        # absolute multi-tissue assignment isn't something to hill-climb.
        for sub in cmd["compound"]:
            params = apply_command(sub, params)
        return params

    params = params.copy()

    if cmd["direction"] == "reset":
        return default_params()

    if cmd["direction"] == "show_only":
        targets = cmd["target"] if isinstance(cmd["target"], list) else [cmd["target"]]
        target_idxs = set()
        for tissue in targets:
            params, idx = _find_or_create_peak(params, tissue)
            target_idxs.add(idx)
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

"soft" covers organs and muscle together (liver, kidney, spleen, muscle, ...) --
no transfer function can isolate one organ from the rest, so never invent a
narrower target such as "liver". "vessels" is only meaningful on contrast-
enhanced scans. "fat", "air" and "spongy" (cancellous/trabecular bone) are not
supported classes any more -- if the user asks for one of those, still map it
to the nearest of the four classes above if there plainly is one (e.g. spongy
bone is part of the skeleton); otherwise do your best with what is available.

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

Respond with JSON only, no prose."""


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


def _validate_cmd(obj) -> bool:
    if isinstance(obj, dict) and set(obj.keys()) == {"compound"}:
        subs = obj["compound"]
        if not isinstance(subs, list) or not subs:
            return False
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


def parse_command_llm(text: str, model: str = "qwen2.5:7b", host: str = OLLAMA_HOST) -> dict:
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
        final = parse_command_rule(text)
        _log_llm_request(text, model, host, None, None, final, f"request failed: {exc}")
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

    if not _validate_cmd(cmd):
        print(f"[llm-parser] falling back to rule parser: invalid schema {cmd!r}")
        final = parse_command_rule(text)
        _log_llm_request(text, model, host, raw_response, cmd, final, f"invalid schema: {cmd!r}")
        return final

    _log_llm_request(text, model, host, raw_response, cmd, cmd, None)
    return cmd
