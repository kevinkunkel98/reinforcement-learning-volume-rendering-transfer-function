"""Score the rule and LLM parsers on `data/parser_eval_set.json`.

Each phrase carries the full expected command (or null: the parser should
refuse it -- speech-recognition noise, retired classes). "dev" phrases are the
ones parser changes are made against: real commands from the viewer's LLM log
(`out/llm_requests.jsonl`) and the earlier free-form set. "heldout" phrases
were written before any of those changes and are only for reporting; their
failures are hidden unless `--show-heldout` is given.

Two scores per parser: "exact" compares target(s), attribute, direction,
strength and level; "intent" ignores strength. An expected strength may be a
list when the phrase does not pin one down ("the lungs are barely visible").

    .venv/bin/python eval_parsers.py                    # both parsers, both splits
    .venv/bin/python eval_parsers.py --parsers rule --split dev
"""
import argparse
import json

import commands
from commands import parse_command_llm, parse_command_rule

EVAL_SET_PATH = "data/parser_eval_set.json"
EVAL_SETS = {"base": EVAL_SET_PATH, "realistic": "data/parser_eval_realistic.json"}


def _canonical_single(cmd: dict, with_strength: bool):
    if "camera" in cmd:
        cam = cmd["camera"]
        return ("camera", cam.get("action"), cam.get("direction"),
                cam.get("strength") if with_strength else None)
    direction = cmd.get("direction")
    target = cmd.get("target")
    targets = tuple(sorted(target)) if isinstance(target, list) else (target,)
    if direction in ("show_only", "reset"):
        # The attribute of a show_only/reset carries no meaning (rule: "opacity", LLM: null).
        return (direction, targets)
    if direction == "set":
        return ("set", targets, cmd.get("attribute"), cmd.get("level"))
    return (direction, targets, cmd.get("attribute"), cmd.get("strength") if with_strength else None)


def _canonical(cmd, with_strength: bool):
    if cmd is None:
        return None
    if set(cmd) == {"compound"}:
        subs = sorted(_canonical_single(s, with_strength) for s in cmd["compound"])
        return ("compound", tuple(subs)) if len(subs) > 1 else subs[0]
    return _canonical_single(cmd, with_strength)


def _strength_options(expected: dict):
    """Every concrete command `expected` allows (a list-valued strength fans out)."""
    def fan(cmd):
        if "camera" in cmd:
            return [cmd]
        s = cmd.get("strength")
        return [{**cmd, "strength": v} for v in s] if isinstance(s, list) else [cmd]

    if expected is None:
        return [None]
    if "compound" in expected:
        options = [[]]
        for sub in expected["compound"]:
            options = [o + [f] for o in options for f in fan(sub)]
        return [{"compound": o} for o in options]
    return fan(expected)


def matches(expected, actual, with_strength: bool = True) -> bool:
    got = _canonical(actual, with_strength)
    return any(_canonical(option, with_strength) == got for option in _strength_options(expected))


def run_parser(fn, text):
    try:
        return fn(text)
    except Exception:  # a refusal is an answer: correct when null is expected
        return None


def score(rows, parsers: dict) -> dict:
    """{parser: [(row, actual, exact_ok, intent_ok), ...]}"""
    results = {}
    for name, fn in parsers.items():
        results[name] = []
        for row in rows:
            actual = run_parser(fn, row["text"])
            results[name].append((row, actual, matches(row["expected"], actual, True),
                                  matches(row["expected"], actual, False)))
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm-model", default="qwen2.5:7b")
    ap.add_argument("--parsers", default="rule,llm")
    ap.add_argument("--split", default="dev,heldout")
    ap.add_argument("--show-heldout", action="store_true", help="also list held-out failures")
    args = ap.parse_args(argv)

    available = {"rule": parse_command_rule,
                 "llm": lambda t: parse_command_llm(t, model=args.llm_model)}
    parsers = {name: available[name] for name in args.parsers.split(",")}
    # The request log is a record of real viewer use; evaluation calls stay out of it.
    commands.LLM_LOG_PATH = "out/llm_requests_eval.jsonl"
    with open(EVAL_SET_PATH) as f:
        rows = [r for r in json.load(f) if r["split"] in args.split.split(",")]
    results = score(rows, parsers)

    for name, rs in results.items():
        for split in args.split.split(","):
            if split == "heldout" and not args.show_heldout:
                continue
            failures = [(r, a) for r, a, ok, _ in rs if r["split"] == split and not ok]
            if failures:
                print(f"\n{name} failures ({split}):")
                for r, a in failures:
                    print(f"  {r['text']!r}\n     expected {r['expected']}\n     got      {a}")

    print(f"\n{'parser':<6} {'split':<8} {'n':>3}  {'exact':>6}  {'intent':>6}")
    for name, rs in results.items():
        for split in args.split.split(","):
            part = [x for x in rs if x[0]["split"] == split]
            if part:
                n = len(part)
                print(f"{name:<6} {split:<8} {n:>3}  {sum(x[2] for x in part) / n:6.0%}  "
                      f"{sum(x[3] for x in part) / n:6.0%}")


if __name__ == "__main__":
    main()
