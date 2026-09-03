"""Compare rule vs LLM parser accuracy on free-form phrases the rule parser
is expected to fail on. Prints a table; this is the point of the change."""
import argparse
import json

from commands import parse_command_rule, parse_command_llm

PHRASES_PATH = "data/parser_eval_phrases.json"


def _check(text, expected_target, expected_direction, fn):
    try:
        cmd = fn(text)
    except Exception:
        return False, None
    ok = cmd.get("target") == expected_target and cmd.get("direction") == expected_direction
    return ok, cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-model", default="qwen2.5:7b")
    args = ap.parse_args()

    with open(PHRASES_PATH) as f:
        phrases = json.load(f)

    rule_correct = llm_correct = 0
    rows = []
    for item in phrases:
        text, exp_t, exp_d = item["text"], item["expected_target"], item["expected_direction"]
        r_ok, r_cmd = _check(text, exp_t, exp_d, parse_command_rule)
        l_ok, l_cmd = _check(text, exp_t, exp_d,
                              lambda t: parse_command_llm(t, model=args.llm_model))
        rule_correct += r_ok
        llm_correct += l_ok
        rows.append((text, exp_t, exp_d, r_ok, l_ok))

    print(f"{'phrase':<45} {'expected':<20} {'rule':<6} {'llm':<6}")
    for text, exp_t, exp_d, r_ok, l_ok in rows:
        print(f"{text:<45} {exp_t}/{exp_d:<10} {'ok' if r_ok else 'X':<6} {'ok' if l_ok else 'X':<6}")

    n = len(phrases)
    print(f"\nrule parser:  {rule_correct}/{n} correct")
    print(f"llm parser:   {llm_correct}/{n} correct")


if __name__ == "__main__":
    main()
