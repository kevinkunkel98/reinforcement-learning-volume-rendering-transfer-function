"""Precompute the visibility caches (`visibility.for_volume` builds one on
first use otherwise, a few seconds per scan).

    python -m tools.build_visibility_cache                  # train + val + test
    python -m tools.build_visibility_cache --splits train_extra
"""
import argparse
import time

import scans
import visibility

SPLITS = ("train", "val", "test")
OPTIONAL_SPLITS = ("train_extra",)


def build(splits=SPLITS, cache_dir=None) -> list:
    built = []
    for split in splits:
        for name in scans.split_names(split):
            started = time.time()
            model = visibility.for_volume(name, cache_dir)
            built.append((split, name, time.time() - started))
            print(f"[visibility] {split:13s} {name:12s} "
                  f"step={model.step_mm:5.2f}mm labels={model.label_source:9s} {built[-1][2]:5.2f}s")
    return built


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS) + list(OPTIONAL_SPLITS))
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    built = build(tuple(args.splits), args.cache_dir)
    print(f"[visibility] {len(built)} volumes cached in {sum(b[2] for b in built):.1f}s")


if __name__ == "__main__":
    main()
