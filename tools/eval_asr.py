"""Speech recognition on the scripted recordings (data/asr_eval.json): word
error rate, command accuracy through the hybrid parser, and time per clip,
for each backend x model size x vocabulary prompt.

    .venv/bin/python tools/eval_asr.py --backends faster-whisper,mlx --sizes small,turbo
"""
import argparse
import json
import re
import sys
import time

sys.path.insert(0, ".")
import asr  # noqa: E402
import commands  # noqa: E402
import lexicon  # noqa: E402
from eval_parsers import matches, percentile  # noqa: E402


def _words(text):
    return re.sub(r"[^a-z' ]", " ", text.lower()).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = [w.strip("'") for w in _words(reference)], [w.strip("'") for w in _words(hypothesis)]
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)] / max(1, len(ref))


def _parse(text):
    try:
        return commands.parse_command_hybrid(text)[0]
    except ValueError:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="data/asr_eval.json")
    ap.add_argument("--backends", default="faster-whisper")
    ap.add_argument("--sizes", default="small")
    args = ap.parse_args(argv)
    commands.LLM_LOG_PATH = "out/llm_requests_eval.jsonl"
    commands.UTTERANCE_LOG_PATH = "out/utterances_eval.jsonl"
    with open(args.pairs) as f:
        pairs = json.load(f)
    expected = [_parse(p["reference"]) for p in pairs]
    prompt = lexicon.get()["asr_prompt"]
    print(f"{'backend':<15} {'size':<6} {'prompt':<6} {'WER':>6} {'cmd ok':>7} {'p50 s':>6} {'p90 s':>6}")
    for backend in args.backends.split(","):
        for size in args.sizes.split(","):
            asr._transcribe_path(pairs[0]["wav"], "en", size, backend=backend)  # load + warm up
            for use_prompt in (False, True):
                wers, oks, secs = [], [], []
                for p, exp in zip(pairs, expected):
                    start = time.perf_counter()
                    text = asr._transcribe_path(p["wav"], "en", size, initial_prompt=prompt if use_prompt else None,
                                                backend=backend)
                    secs.append(time.perf_counter() - start)
                    wers.append(word_error_rate(p["reference"], text))
                    got = None if asr.is_hallucination(text) else _parse(text)
                    oks.append(matches(exp, got))
                n = len(pairs)
                print(f"{backend:<15} {size:<6} {'yes' if use_prompt else 'no':<6} {sum(wers) / n:6.1%} "
                      f"{sum(oks) / n:7.0%} {percentile(secs, 50):6.2f} {percentile(secs, 90):6.2f}")


if __name__ == "__main__":
    main()
