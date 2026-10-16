"""Add every remaining eligible TotalSegmentator subject as split "train_extra".

The RL v2 manifest uses 30 of the zip's 100 subjects that pass the extent
filter (20 / 4 / 6 train / val / test). A learned renderer model
(`rl.model_based`) has to generalise across patients, and 20 training patients
is what limits it; this extracts the other ~70 without touching the existing
entries, so val and test -- and every result scored on them -- stay as they are.
SAC experiments keep reading split "train"; only callers that ask for
"train_extra" see the new volumes.

    python -m tools.extend_totalseg
    python -m tools.build_visibility_cache --splits train_extra
"""
import argparse
import json
import os
import zipfile

from tools.select_totalseg import (DEFAULT_MANIFEST, DEFAULT_OUT_DIR, DEFAULT_ZIP, eligible_candidates,
                                   extract_subject)

EXTRA_SPLIT = "train_extra"


def extend_manifest(manifest: dict, zip_path: str, out_dir: str = DEFAULT_OUT_DIR, limit: int = None) -> dict:
    """`manifest` plus every eligible subject it does not list yet, as EXTRA_SPLIT."""
    known = {s["id"] for s in manifest["subjects"]}
    with zipfile.ZipFile(zip_path) as zf:
        candidates, info = eligible_candidates(zf)
        new_ids = sorted(c["id"] for c in candidates if c["id"] not in known)[:limit]
        added = [extract_subject(zf, sid, info[sid], EXTRA_SPLIT, out_dir) for sid in new_ids]
    extended = dict(manifest)
    extended["subjects"] = list(manifest["subjects"]) + added
    extended["selection"] = {**manifest.get("selection", {}), "extra_split": EXTRA_SPLIT,
                             "n_extra": len(manifest_extra(extended))}
    return extended


def manifest_extra(manifest: dict) -> list:
    return [s for s in manifest["subjects"] if s["split"] == EXTRA_SPLIT]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", default=DEFAULT_ZIP)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--limit", type=int, default=None, help="add at most this many subjects")
    args = parser.parse_args()
    with open(args.manifest) as f:
        manifest = json.load(f)
    extended = extend_manifest(manifest, args.zip, args.out_dir, args.limit)
    tmp = args.manifest + ".tmp"
    with open(tmp, "w") as f:
        json.dump(extended, f, indent=2)
    os.replace(tmp, args.manifest)
    print(f"[extend_totalseg] {len(manifest_extra(extended))} {EXTRA_SPLIT} subjects in {args.manifest}")


if __name__ == "__main__":
    main()
