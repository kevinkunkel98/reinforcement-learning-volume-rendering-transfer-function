"""Merge several raters' `vis_preferences.jsonl` files and measure how well
they agree, so the thesis can report inter-rater agreement -- the ceiling any
reward model trained on this data can be expected to reach.

    .venv/bin/python -m tools.merge_preferences out/*.jsonl --out out/preferences_merged.jsonl

Agreement is measured on the anchor items every rater judges (see
`rl.candidates.anchor_items`, `collect.py`'s `Collector`): items whose sides
("A"/"B") are shuffled independently per rater, so a raw `choice` letter
means different things to different people. Every agreement metric here
resolves a judgment to *which candidate* was chosen -- identified by its
stored `params`/`source`, never by the displayed letter -- before comparing
across raters. Two raters who both chose the same underlying candidate but
saw it on opposite sides must, and do, count as agreeing (see
`tests/test_merge_preferences.py`'s
`test_agreement_table_counts_raters_who_chose_the_same_candidate_on_opposite_sides`).

`scipy` is not installed here, so both Cohen's kappa and Krippendorff's alpha
are implemented directly, the same house style as `rl.vis_eval.wilcoxon`
(hand-written, with a worked example in a comment and a test that checks it).
"""
import argparse
import collections
import json
import os

import numpy as np

ORDINAL_CATEGORIES = ("x", "equal", "y")   # chosen-candidate-A / equal / chosen-candidate-B, canonicalised


# --- reading/writing JSONL ----------------------------------------------------

def _read_jsonl(path: str) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# --- merge ---------------------------------------------------------------------

def merge(paths, out_path: str = None) -> dict:
    """Concatenate rows from `paths` (each a rater's preference file),
    de-duplicating on `(pair_id, rater_id)` -- when the same pair appears
    more than once for the same rater (e.g. a file was merged before and is
    being merged again), the row with the later `timestamp` wins. Writes the
    merged rows to `out_path` (one JSON object per line) when given.

    Returns `{"total", "per_rater": {rater_id: n_rows}, "anchor_coverage":
    {rater_id: n_distinct_anchor_items_judged}}`.
    """
    by_key = {}
    for path in paths:
        for row in _read_jsonl(path):
            key = (row["pair_id"], row["rater_id"])
            existing = by_key.get(key)
            if existing is None or row["timestamp"] > existing["timestamp"]:
                by_key[key] = row

    rows = sorted(by_key.values(), key=lambda row: (row["rater_id"], row["timestamp"]))

    per_rater = collections.Counter(row["rater_id"] for row in rows)
    anchor_sets = {rater_id: set() for rater_id in per_rater}
    for row in rows:
        if row.get("anchor_id") is not None:
            anchor_sets[row["rater_id"]].add(row["anchor_id"])
    anchor_coverage = {rater_id: len(ids) for rater_id, ids in anchor_sets.items()}

    if out_path is not None:
        _write_jsonl(out_path, rows)

    return {"total": len(rows), "per_rater": dict(per_rater), "anchor_coverage": anchor_coverage}


# --- candidate identity: resolving a judgment past the displayed letter ------

def _candidate_key(candidate: dict) -> tuple:
    return (candidate["source"], tuple(round(float(v), 10) for v in candidate["params"]))


def _canonical_pair(a: dict, b: dict) -> tuple:
    """The two candidates of one anchor item, ordered the same way
    regardless of which rater's row (and therefore which display side) we
    happened to read them from -- the fixed points "x" and "y" that every
    rater's choice gets resolved against."""
    return tuple(sorted((a, b), key=_candidate_key))


def _match(candidate: dict, reference: dict) -> bool:
    return _candidate_key(candidate) == _candidate_key(reference)


def _categorize_choice(row: dict, x: dict, y: dict):
    """This row's `choice` resolved to `"x"`/`"equal"`/`"y"` against the
    canonical pair `(x, y)`, independent of whether this row displayed `x`
    as "a" or "b". `None` for `"skip"` (no candidate chosen) or a choice
    that matches neither `x` nor `y` (should not happen for a real anchor
    row; treated as missing rather than raising)."""
    choice = row["choice"]
    if choice == "equal":
        return "equal"
    if choice == "a":
        chosen = row["a"]
    elif choice == "b":
        chosen = row["b"]
    else:
        return None
    if _match(chosen, x):
        return "x"
    if _match(chosen, y):
        return "y"
    return None


def _anchor_categories(rows: list) -> dict:
    """anchor_id -> list of categories, one per rater who judged it (missing
    for a rater who skipped it or never saw it)."""
    by_anchor = collections.defaultdict(list)
    for row in rows:
        if row.get("anchor_id") is not None:
            by_anchor[row["anchor_id"]].append(row)

    result = {}
    for anchor_id, anchor_rows in by_anchor.items():
        x, y = _canonical_pair(anchor_rows[0]["a"], anchor_rows[0]["b"])
        categories = [c for c in (_categorize_choice(row, x, y) for row in anchor_rows) if c is not None]
        result[anchor_id] = categories
    return result


# --- Cohen's kappa (weighted and unweighted), hand-written -------------------

def _cohen_kappa(pairs: list, categories: tuple, weighted: str = "linear"):
    """Cohen's kappa between two raters' categorical judgments, `pairs`
    being a list of `(category_1, category_2)` tuples over `categories` (an
    ordered tuple for `weighted="linear"`; order is irrelevant for
    `weighted="none"`). `weighted="linear"` gives linearly-weighted kappa
    (disagreement weight `|i - j|` between ordinal positions `i`, `j`);
    `weighted="none"` gives plain (unweighted) kappa, algebraically the same
    formula with disagreement weight 1 for any mismatch, 0 for a match --
    see the worked example in `tests/test_merge_preferences.py`
    (`test_weighted_kappa_matches_a_hand_computed_example`):

        pairs = [(x,x), (x,equal), (equal,equal), (y,y), (y,x)]
        confusion matrix (rows=rater1, cols=rater2), categories (x,equal,y):
                 x  equal  y
            x  [ 1    1    0 ]   row total 2
            eq [ 0    1    0 ]   row total 1
            y  [ 1    0    1 ]   row total 2
                 col totals: x=2, equal=2, y=1        n = 5
        linear weights w_ij = |i-j|:
            observed = sum w_ij*count_ij = (x,equal):1*1 + (y,x):2*1 = 3
            expected = sum w_ij*row_i*col_j / n = 23 / 5 = 4.6
            kappa = 1 - observed/expected = 1 - 3/4.6 = 8/23 ~= 0.347826

    Returns `None` when there are no pairs; a perfect-agreement `pairs` with
    zero variance (expected disagreement is also zero) returns `1.0`.
    """
    n = len(pairs)
    if n == 0:
        return None

    index = {category: i for i, category in enumerate(categories)}
    k = len(categories)
    counts = np.zeros((k, k))
    for c1, c2 in pairs:
        counts[index[c1], index[c2]] += 1

    row_totals = counts.sum(axis=1)
    col_totals = counts.sum(axis=0)
    positions = np.arange(k)
    if weighted == "linear":
        weights = np.abs(np.subtract.outer(positions, positions)).astype(float)
    elif weighted == "none":
        weights = (np.subtract.outer(positions, positions) != 0).astype(float)
    else:
        raise ValueError(f"unknown weighted={weighted!r}, expected 'linear' or 'none'")

    observed = float(np.sum(weights * counts))
    expected = float(np.sum(weights * np.outer(row_totals, col_totals) / n))
    if expected == 0.0:
        return 1.0 if observed == 0.0 else 0.0
    return 1.0 - observed / expected


# --- agreement_table: pairwise, over the anchor items -------------------------

def agreement_table(rows: list) -> list:
    """For every pair of raters, the anchor items both judged (excluding
    `"skip"`), raw agreement, and linearly-weighted Cohen's kappa over the
    ordered outcomes (chosen-candidate-A / equal / chosen-candidate-B,
    resolved by candidate identity -- see module docstring). One entry per
    rater pair, `{"rater_a", "rater_b", "n_shared", "raw_agreement",
    "weighted_kappa"}`; `raw_agreement`/`weighted_kappa` are `None` when the
    pair shares no judged anchor items."""
    rater_categories = collections.defaultdict(dict)
    for anchor_id, categories_by_row in _anchor_categories_with_raters(rows).items():
        for rater_id, category in categories_by_row.items():
            rater_categories[rater_id][anchor_id] = category

    # Every rater who appears at all, not just those with resolvable anchor
    # judgments -- a pair with no shared anchor data still gets a row
    # (n_shared=0), rather than being silently dropped from the table.
    raters = sorted({row["rater_id"] for row in rows})
    table = []
    for i, rater_a in enumerate(raters):
        for rater_b in raters[i + 1:]:
            shared = sorted(set(rater_categories[rater_a]) & set(rater_categories[rater_b]))
            pairs = [(rater_categories[rater_a][anchor_id], rater_categories[rater_b][anchor_id])
                     for anchor_id in shared]
            n = len(pairs)
            raw_agreement = (sum(1 for c1, c2 in pairs if c1 == c2) / n) if n else None
            weighted_kappa = _cohen_kappa(pairs, ORDINAL_CATEGORIES, weighted="linear") if n else None
            table.append({"rater_a": rater_a, "rater_b": rater_b, "n_shared": n,
                          "raw_agreement": raw_agreement, "weighted_kappa": weighted_kappa})
    return table


def _anchor_categories_with_raters(rows: list) -> dict:
    """anchor_id -> {rater_id: category}, the per-rater breakdown
    `_anchor_categories` collapses into a plain list."""
    by_anchor = collections.defaultdict(list)
    for row in rows:
        if row.get("anchor_id") is not None:
            by_anchor[row["anchor_id"]].append(row)

    result = {}
    for anchor_id, anchor_rows in by_anchor.items():
        x, y = _canonical_pair(anchor_rows[0]["a"], anchor_rows[0]["b"])
        per_rater = {}
        for row in anchor_rows:
            category = _categorize_choice(row, x, y)
            if category is not None:
                per_rater[row["rater_id"]] = category
        result[anchor_id] = per_rater
    return result


# --- Krippendorff's alpha (ordinal), hand-written -----------------------------

def krippendorff_alpha(rows: list):
    """Krippendorff's alpha for ordinal data (categories chosen-candidate-A /
    equal / chosen-candidate-B, resolved by candidate identity -- see module
    docstring) over the anchor items, tolerating missing judgments: a rater
    who never judged a given anchor item (or skipped it) simply contributes
    nothing for it, via Krippendorff's coincidence-matrix formulation, no
    imputation needed.

    Worked example, verified in `tests/test_merge_preferences.py`
    (`test_krippendorff_alpha_matches_a_hand_computed_example`):

        Two anchor items, ordinal categories x=0/equal=1/y=2:
            anchor 0: r1 -> x, r2 -> equal
            anchor 1: r1 -> y, r2 -> y
        Coincidence matrix o_ck (each unit's within-unit pairs, weighted
        1/(m_u - 1), added symmetrically for c != d):
            o_{x,equal} = o_{equal,x} = 1     (unit "anchor 0", m=2, weight 1)
            o_{y,y} = 2                       (unit "anchor 1": 2*1/(2-1))
        marginals (row sums): n_x=1, n_equal=1, n_y=2, n=4
        ordinal difference function delta^2(c,d), summing marginals from
        min(c,d) to max(c,d):
            delta^2(x,equal) = (n_x+n_eq - (n_x+n_eq)/2)^2        = (2-1)^2    = 1
            delta^2(x,y)     = (n_x+n_eq+n_y - (n_x+n_y)/2)^2     = (4-1.5)^2  = 6.25
            delta^2(eq,y)    = (n_eq+n_y - (n_eq+n_y)/2)^2        = (3-1.5)^2  = 2.25
        Do = (1/n) * sum_{c!=d} o_cd * delta^2(c,d) = (1/4)*(1*1 + 1*1) = 0.5
        De = (1/(n(n-1))) * sum_{c!=d} n_c*n_d*delta^2(c,d)
           = (1*1*1 + 1*1*1 + 1*2*6.25 + 2*1*6.25 + 1*2*2.25 + 2*1*2.25) / 12
           = 36 / 12 = 3.0
        alpha = 1 - Do/De = 1 - 0.5/3.0 = 1 - 1/6 = 5/6 ~= 0.833333

    Returns `None` when fewer than 2 pairable values exist (nothing to
    compare reliability against).
    """
    index = {category: i for i, category in enumerate(ORDINAL_CATEGORIES)}
    k = len(ORDINAL_CATEGORIES)
    coincidence = np.zeros((k, k))
    n = 0
    for categories in _anchor_categories(rows).values():
        m = len(categories)
        if m < 2:
            continue
        counts = np.zeros(k)
        for category in categories:
            counts[index[category]] += 1
        n += m
        weight = 1.0 / (m - 1)
        for c in range(k):
            if counts[c] == 0.0:
                continue
            coincidence[c, c] += counts[c] * (counts[c] - 1.0) * weight
            for d in range(c + 1, k):
                if counts[d] == 0.0:
                    continue
                pair_weight = counts[c] * counts[d] * weight
                coincidence[c, d] += pair_weight
                coincidence[d, c] += pair_weight

    if n < 2:
        return None

    marginals = coincidence.sum(axis=1)

    def _delta2(c: int, d: int) -> float:
        lo, hi = (c, d) if c <= d else (d, c)
        cumulative = float(marginals[lo:hi + 1].sum())
        return (cumulative - (marginals[lo] + marginals[hi]) / 2.0) ** 2

    do = 0.0
    de = 0.0
    for c in range(k):
        for d in range(k):
            if c == d:
                continue
            delta2 = _delta2(c, d)
            do += coincidence[c, d] * delta2
            de += marginals[c] * marginals[d] * delta2
    do /= n
    de /= n * (n - 1)

    if de == 0.0:
        return 1.0 if do == 0.0 else 0.0
    return 1.0 - do / de


# --- self_consistency: repeats vs. originals, per rater -----------------------

def self_consistency(rows: list) -> dict:
    """Per rater, raw agreement and unweighted Cohen's kappa between each
    repeated item (`repeat_of` not null) and its original. A repeat is shown
    with sides swapped from the original (`collect.py`'s `_swap_sides`), so
    this resolves both judgments by candidate identity, the same rule as
    `agreement_table`, rather than comparing raw `choice` letters.

    Returns `{rater_id: {"n", "raw_agreement", "kappa"}}`; a rater with no
    judged repeats is absent from the result."""
    by_pair_id = {row["pair_id"]: row for row in rows}
    pairs_by_rater = collections.defaultdict(list)
    for row in rows:
        repeat_of = row.get("repeat_of")
        if not repeat_of:
            continue
        original = by_pair_id.get(repeat_of)
        if original is None or original["rater_id"] != row["rater_id"]:
            continue
        x, y = _canonical_pair(original["a"], original["b"])
        cat_original = _categorize_choice(original, x, y)
        cat_repeat = _categorize_choice(row, x, y)
        if cat_original is None or cat_repeat is None:
            continue
        pairs_by_rater[row["rater_id"]].append((cat_original, cat_repeat))

    result = {}
    for rater_id, pairs in pairs_by_rater.items():
        n = len(pairs)
        result[rater_id] = {
            "n": n,
            "raw_agreement": sum(1 for c1, c2 in pairs if c1 == c2) / n,
            "kappa": _cohen_kappa(pairs, ORDINAL_CATEGORIES, weighted="none"),
        }
    return result


# --- CLI -----------------------------------------------------------------------

def _print_counts(summary: dict) -> None:
    print(f"total: {summary['total']}")
    print(f"{'rater':20s} {'n':>6s} {'anchors covered':>16s}")
    for rater_id in sorted(summary["per_rater"]):
        print(f"{rater_id:20s} {summary['per_rater'][rater_id]:6d} "
              f"{summary['anchor_coverage'].get(rater_id, 0):16d}")


def _fmt(value) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def _print_agreement_table(rows: list) -> None:
    print("\ninter-rater agreement (anchor items):")
    print(f"{'rater a':16s} {'rater b':16s} {'n shared':>10s} {'raw agree':>10s} {'weighted kappa':>15s}")
    for entry in agreement_table(rows):
        print(f"{entry['rater_a']:16s} {entry['rater_b']:16s} {entry['n_shared']:10d} "
              f"{_fmt(entry['raw_agreement']):>10s} {_fmt(entry['weighted_kappa']):>15s}")

    alpha = krippendorff_alpha(rows)
    print(f"\nKrippendorff's alpha (ordinal, anchor items): {_fmt(alpha)}")


def _print_self_consistency(rows: list) -> None:
    print("\nself-consistency (repeats vs. originals):")
    print(f"{'rater':20s} {'n':>6s} {'raw agree':>10s} {'kappa':>10s}")
    for rater_id, entry in sorted(self_consistency(rows).items()):
        print(f"{rater_id:20s} {entry['n']:6d} {_fmt(entry['raw_agreement']):>10s} {_fmt(entry['kappa']):>10s}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="one or more raters' vis_preferences.jsonl files")
    parser.add_argument("--out", default="out/preferences_merged.jsonl")
    args = parser.parse_args()

    summary = merge(args.paths, args.out)
    _print_counts(summary)

    rows = _read_jsonl(args.out)
    _print_agreement_table(rows)
    _print_self_consistency(rows)

    print(f"\n[merge_preferences] wrote {args.out}")


if __name__ == "__main__":
    main()
