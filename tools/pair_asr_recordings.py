"""Pair the recordings of one scripted session with the script lines.

Read data/asr_script.txt aloud in the app, one trigger press per line, after
noting the time. Then:

    .venv/bin/python tools/pair_asr_recordings.py --since 20261022-100000

writes data/asr_eval.json. Refuses when the number of recordings differs from
the number of lines -- delete the extra .wav/.txt pairs (retakes) first.
"""
import argparse
import glob
import json
import os


def pair(audio_dir: str, since: str, lines: list) -> list:
    wavs = sorted(p for p in glob.glob(os.path.join(audio_dir, "*.wav"))
                  if os.path.basename(p)[:-4] >= since)
    if len(wavs) != len(lines):
        raise ValueError(f"{len(wavs)} recordings for {len(lines)} script lines")
    return [{"wav": w, "reference": line} for w, line in zip(wavs, lines)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", required=True, help="first recording's stamp, YYYYMMDD-HHMMSS")
    ap.add_argument("--script", default="data/asr_script.txt")
    ap.add_argument("--audio-dir", default="out/audio")
    ap.add_argument("--out", default="data/asr_eval.json")
    args = ap.parse_args(argv)
    with open(args.script) as f:
        lines = [line.strip() for line in f if line.strip()]
    rows = pair(args.audio_dir, args.since, lines)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {len(rows)} pairs to {args.out}")


if __name__ == "__main__":
    main()
