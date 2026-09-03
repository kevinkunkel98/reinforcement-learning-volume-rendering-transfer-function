"""Text -> command dict (two implementations) and command -> params."""
import datetime as _dt
import json as _json
import re
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np

from transfer import (
    CENTER_RANGE, N_PEAKS, PARAMS_PER_PEAK, TISSUE_HU, WIDTH_RANGE,
    _from_range, _from_unit, _unit, default_params, peak_internal,
)

TISSUE_SYNONYMS = {
    "bone": ["bone", "bones", "cortical bone", "cortical", "skeleton"],
    "spongy": ["spongy bone", "spongy", "cancellous", "trabecular"],
    "soft": ["soft tissue", "soft", "tissue", "muscle"],
    "fat": ["fatty", "fat", "adipose"],
    "air": ["air", "background"],
}
# longest phrase first so "cortical bone" matches before "bone"
_SYNONYM_LOOKUP = sorted(
    ((phrase, tissue) for tissue, phrases in TISSUE_SYNONYMS.items() for phrase in phrases),
    key=lambda t: -len(t[0]),
)

STRENGTH_WORDS = {"slightly": 0.15, "moderately": 0.35, "strongly": 0.6}
NEAR_THRESHOLD_HU = 300.0


def _find_tissue(text: str):
    for phrase, tissue in _SYNONYM_LOOKUP:
        if phrase in text:
            return tissue
    return None


def _split_tissue_list(text: str) -> list:
    """'bone and spongy' / 'bone, spongy' / 'bone + spongy' -> ['bone', 'spongy']."""
    parts = re.split(r"\s*(?:,|\+|\band\b)\s*", text.strip())
    tissues = []
    for part in parts:
        tissue = _find_tissue(part)
        if tissue and tissue not in tissues:
            tissues.append(tissue)
    return tissues


def parse_command_rule(text: str) -> dict:
    t = text.lower().strip()

    if "reset" in t:
        return {"target": None, "attribute": None, "direction": "reset", "strength": None}

    m = re.search(r"show only ([\w ,\+]+)", t)
    if m:
        tissues = _split_tissue_list(m.group(1))
        if tissues:
            return {"target": tissues[0] if len(tissues) == 1 else tissues,
                     "attribute": "opacity", "direction": "show_only", "strength": None}

    # "high opacity spongy", "low opacity for bone" -- an absolute level per
    # tissue, not a relative delta. One or more may appear in the same
    # sentence ("high opacity spongy, low opacity bone"); each becomes its
    # own sub-command, folded together into one compound command.
    level_matches = list(re.finditer(r"(low|medium|high)\s+opacity\s+(?:for\s+)?(\w+)", t))
    if level_matches:
        subcommands = []
        for lm in level_matches:
            level, tissue_text = lm.group(1), lm.group(2)
            tissue = _find_tissue(tissue_text)
            if tissue:
                subcommands.append({"target": tissue, "attribute": "opacity",
                                      "direction": "set", "level": level})
        if subcommands:
            return subcommands[0] if len(subcommands) == 1 else {"compound": subcommands}

    m = re.search(r"(increase|decrease)\s+opacity\s+for\s+([\w ]+)", t)
    if m:
        direction, target_text = m.group(1), m.group(2)
        strength = "moderately"
        for word in STRENGTH_WORDS:
            if word in target_text:
                strength = word
                target_text = target_text.replace(word, "")
        tissue = _find_tissue(target_text)
        if tissue:
            return {"target": tissue, "attribute": "opacity",
                     "direction": direction, "strength": strength}

    raise ValueError(f"rule parser cannot parse: {text!r}")


def parse_command(text: str, parser: str = "rule", model: str = "qwen2.5:7b") -> dict:
    if parser == "llm":
        return parse_command_llm(text, model=model)
    return parse_command_rule(text)


def _peak_center_hu(params: np.ndarray, i: int) -> float:
    return peak_internal(params, i)["center"]


def _find_or_create_peak(params: np.ndarray, tissue: str):
    """Nearest peak to the tissue's HU; reseed the weakest peak if none is near."""
    target_hu = TISSUE_HU[tissue]
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

    if cmd["direction"] == "set":
        new_h = LEVEL_WORDS[cmd["level"]]
        params[base + 2] = _from_unit(new_h)
        return params

    delta = STRENGTH_WORDS[cmd["strength"] or "moderately"]
    current_h = _unit(params[base + 2])
    # Move a fraction of the remaining headroom to the bound, not a fixed
    # absolute amount -- an additive step this large (e.g. "strongly" = 0.6)
    # saturated almost any starting height to 0/1 in a single command, after
    # which every repeated increase/decrease was a silent no-op.
    if cmd["direction"] == "increase":
        new_h = current_h + delta * (1.0 - current_h)
    else:
        new_h = current_h - delta * current_h
    new_h = float(np.clip(new_h, 0.0, 1.0))
    params[base + 2] = _from_unit(new_h)
    return params


OLLAMA_HOST = "http://localhost:11434"
VALID_DIRECTIONS = {"increase", "decrease", "show_only", "reset"}
VALID_STRENGTHS = set(STRENGTH_WORDS) | {None}
VALID_LEVELS = set(LEVEL_WORDS)

def _tissue_synonym_lines() -> str:
    # Generated from TISSUE_SYNONYMS so the LLM and the rule parser can never
    # drift onto different tissue vocabularies.
    lines = []
    for tissue, phrases in TISSUE_SYNONYMS.items():
        aliases = [p for p in phrases if p != tissue]
        lines.append(f'- "{tissue}": also called {", ".join(aliases)}' if aliases else f'- "{tissue}"')
    return "\n".join(lines)


_SYSTEM_PROMPT = f"""You convert one spoken instruction about a volume-rendering \
transfer function into strict JSON, nothing else.

Tissues, with the words people actually use for them -- map any of these back to
the exact tissue name on the left, never invent a different one:
{_tissue_synonym_lines()}

"spongy" and "bone" are different tissues in this system even though everyday
speech calls both of them "bone" -- spongy/cancellous/trabecular bone is its own
target, distinct from cortical bone/skeleton.

Output schema -- the usual case is a single command:
{{"target": "<tissue>|[<tissue>, ...]|null", "attribute": "opacity"|null,
 "direction": "increase"|"decrease"|"show_only"|"reset",
 "strength": "slightly"|"moderately"|"strongly"|null}}

"target" is a list only for "show_only" when the user names more than one
tissue ("show bone and spongy" -> target: ["bone", "spongy"]).

direction is from the user's goal, not their wording: if something is missing,
faint, invisible, or needs more presence, that is "increase" (more of it should
be visible) even if the sentence itself sounds negative ("barely there" ->
increase). If something is hiding, dominating, or should be reduced, that is
"decrease".

Never output numeric values. "reset" has target=null, attribute=null, strength=null.
"show_only" has strength=null.

When the user wants two or more tissues set to different absolute levels in
one sentence ("high opacity spongy, low opacity bone"), respond with a
compound command instead: a list of single-tissue "set" commands, each with a
"level" field (not "strength"):
{{"compound": [
  {{"target": "spongy", "attribute": "opacity", "direction": "set", "level": "high"}},
  {{"target": "bone", "attribute": "opacity", "direction": "set", "level": "low"}}
]}}
"level" is "low"|"medium"|"high" -- an absolute target, not a relative change.
Use "set"/"level" only inside a compound command, never "strength" there.

Respond with JSON only, no prose."""


def _validate_set_cmd(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    if set(obj.keys()) != {"target", "attribute", "direction", "level"}:
        return False
    if obj["direction"] != "set":
        return False
    if not isinstance(obj["target"], str) or obj["target"] not in TISSUE_HU:
        return False
    if obj["level"] not in VALID_LEVELS:
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
        if not targets or any(t is None or t not in TISSUE_HU for t in targets):
            return False
    else:
        if obj["target"] is not None and obj["target"] not in TISSUE_HU:
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
        return all(_validate_set_cmd(sub) for sub in subs)
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
    }).encode()
    req = Request(f"{host}/api/generate", data=body,
                   headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=10) as resp:
            raw = _json.loads(resp.read().decode())
        raw_response = raw["response"]
        cmd = _json.loads(raw_response)
    except (URLError, OSError, ValueError, KeyError) as exc:
        print(f"[llm-parser] falling back to rule parser: {exc}")
        final = parse_command_rule(text)
        _log_llm_request(text, model, host, None, None, final, f"request failed: {exc}")
        return final

    # The model sometimes echoes an alias ("bones") instead of the canonical
    # key ("bone") despite the prompt asking it not to -- normalize through the
    # same synonym table the rule parser uses before validating, rather than
    # discarding an otherwise-correct answer over a naming mismatch. Applies
    # to a plain string target, each entry of a list target (multi-tissue
    # show_only), and each sub-command's target inside a compound command.
    def _normalize_target(value):
        if isinstance(value, str) and value not in TISSUE_HU:
            return _find_tissue(value) or value
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
