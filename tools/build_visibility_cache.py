"""Precompute visibility caches for the RL v2 volumes.

    python -m tools.build_visibility_cache                  # train + val + test + out_of_source
    python -m tools.build_visibility_cache --splits train
"""
import argparse
import time

import datasets
import visibility

SPLITS = ("train", "val", "test", "out_of_source")


def build(splits=SPLITS, cache_dir=None) -> list:
    built = []
    for split in splits:
        for name in datasets.volumes_for_split(split):
            started = time.time()
            model = visibility.for_volume(name, cache_dir)
            built.append((split, name, time.time() - started))
            print(f"[visibility] {split:13s} {name:12s} "
                  f"step={model.step_mm:5.2f}mm {built[-1][2]:5.2f}s")
    return built


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    built = build(tuple(args.splits), args.cache_dir)
    print(f"[visibility] {len(built)} volumes cached in {sum(b[2] for b in built):.1f}s")


if __name__ == "__main__":
    main()
