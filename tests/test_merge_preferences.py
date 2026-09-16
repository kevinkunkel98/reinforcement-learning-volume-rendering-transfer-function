import json

import pytest

from tools.merge_preferences import (
    _cohen_kappa,
    agreement_table,
    krippendorff_alpha,
    merge,
    self_consistency,
)

P = {"params": [0.1, 0.2], "source": "B1_current_executor"}   # sorts before Q ("B" < "p")
Q = {"params": [0.9, 0.9], "source": "perturbation"}


def _row(pair_id, rater_id, timestamp="2024-01-01T00:00:00", anchor_id=None, a=None, b=None,
         choice="a", repeat_of=None):
    return {
        "pair_id": pair_id,
        "rater_id": rater_id,
        "timestamp": timestamp,
        "anchor_id": anchor_id,
        "a": a if a is not None else P,
        "b": b if b is not None else Q,
        "choice": choice,
        "repeat_of": repeat_of,
    }


# --- merge: de-duplication, counts, anchor coverage --------------------------

def test_merge_deduplicates_on_pair_id_and_rater_id_keeping_later_timestamp(tmp_path):
    path_a = tmp_path / "rater1_old.jsonl"
    path_b = tmp_path / "rater1_new.jsonl"
    old_row = _row("p1", "kk", timestamp="2024-01-01T00:00:00", choice="a")
    new_row = _row("p1", "kk", timestamp="2024-01-02T00:00:00", choice="b")  # same pair, later, different choice
    path_a.write_text(json.dumps(old_row) + "\n")
    path_b.write_text(json.dumps(new_row) + "\n")

    summary = merge([str(path_a), str(path_b)], str(tmp_path / "merged.jsonl"))

    assert summary["total"] == 1
    with open(tmp_path / "merged.jsonl") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 1
    assert rows[0]["choice"] == "b"  # the later row wins


def test_merge_keeps_rows_with_the_same_pair_id_from_different_raters(tmp_path):
    path = tmp_path / "combined.jsonl"
    row1 = _row("p1", "kk")
    row2 = _row("p1", "other")  # same pair_id, different rater -- both are real rows (an anchor item)
    path.write_text(json.dumps(row1) + "\n" + json.dumps(row2) + "\n")

    summary = merge([str(path)], str(tmp_path / "merged.jsonl"))

    assert summary["total"] == 2
    assert summary["per_rater"] == {"kk": 1, "other": 1}


def test_merge_reports_anchor_coverage_per_rater(tmp_path):
    path = tmp_path / "combined.jsonl"
    rows = [
        _row("p1", "kk", anchor_id=0),
        _row("p2", "kk", anchor_id=1),
        _row("p3", "kk", anchor_id=None),   # not an anchor item -- doesn't count
        _row("p4", "other", anchor_id=0),
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    summary = merge([str(path)], str(tmp_path / "merged.jsonl"))

    assert summary["anchor_coverage"] == {"kk": 2, "other": 1}


def test_merge_without_out_path_still_returns_a_summary(tmp_path):
    path = tmp_path / "combined.jsonl"
    path.write_text(json.dumps(_row("p1", "kk")) + "\n")

    summary = merge([str(path)], None)

    assert summary["total"] == 1


# --- agreement_table: candidate identity, not displayed letter ---------------

def test_agreement_table_counts_raters_who_chose_the_same_candidate_on_opposite_sides(tmp_path):
    # THE critical case: sides are shuffled per rater. rater1 saw P as "a" and
    # chose "a" (P); rater2 saw the very same pair with sides swapped -- P as
    # "b" -- and chose "b" (P too). They chose the SAME underlying candidate
    # and must count as agreeing, even though their raw "choice" strings
    # ("a" vs "b") differ.
    rows = [
        _row("p1", "r1", anchor_id=0, a=P, b=Q, choice="a"),
        _row("p2", "r2", anchor_id=0, a=Q, b=P, choice="b"),
    ]

    table = agreement_table(rows)

    assert len(table) == 1
    entry = table[0]
    assert {entry["rater_a"], entry["rater_b"]} == {"r1", "r2"}
    assert entry["n_shared"] == 1
    assert entry["raw_agreement"] == pytest.approx(1.0)
    assert entry["weighted_kappa"] == pytest.approx(1.0)


def test_agreement_table_a_naive_letter_comparison_would_get_this_wrong(tmp_path):
    # Same setup, but framed as a regression test: if agreement were computed
    # on the raw "choice" letters ("a" == "a"? "b" == "b"?) instead of
    # resolved candidate identity, rater1's "a" and rater2's "b" would look
    # like disagreement. They must not.
    rows = [
        _row("p1", "r1", anchor_id=5, a=P, b=Q, choice="a"),   # chose P
        _row("p2", "r2", anchor_id=5, a=Q, b=P, choice="b"),   # chose P (opposite side)
    ]
    assert rows[0]["choice"] != rows[1]["choice"]  # letters disagree...

    table = agreement_table(rows)

    assert table[0]["raw_agreement"] == pytest.approx(1.0)  # ...but the candidates agree


def test_agreement_table_detects_real_disagreement(tmp_path):
    rows = [
        _row("p1", "r1", anchor_id=0, a=P, b=Q, choice="a"),   # chose P
        _row("p2", "r2", anchor_id=0, a=P, b=Q, choice="b"),   # chose Q
    ]

    table = agreement_table(rows)

    assert table[0]["raw_agreement"] == pytest.approx(0.0)
    assert table[0]["weighted_kappa"] == pytest.approx(0.0)


def test_agreement_table_only_uses_anchor_items(tmp_path):
    rows = [
        _row("p1", "r1", anchor_id=None, a=P, b=Q, choice="a"),
        _row("p2", "r2", anchor_id=None, a=P, b=Q, choice="a"),
    ]

    table = agreement_table(rows)

    assert table == [{"rater_a": "r1", "rater_b": "r2", "n_shared": 0,
                       "raw_agreement": None, "weighted_kappa": None}]


def test_agreement_table_excludes_skip_choices_as_missing(tmp_path):
    rows = [
        _row("p1", "r1", anchor_id=0, a=P, b=Q, choice="skip"),
        _row("p2", "r2", anchor_id=0, a=P, b=Q, choice="a"),
    ]

    table = agreement_table(rows)

    assert table[0]["n_shared"] == 0


# --- Cohen's kappa: worked example (house style, see rl/vis_eval.wilcoxon) ---

def test_weighted_kappa_matches_a_hand_computed_example():
    # categories, ordinal: x=0, equal=1, y=2. Five judged pairs (rater1, rater2):
    #   (x,x) (x,equal) (equal,equal) (y,y) (y,x)
    # confusion matrix (rows=rater1, cols=rater2):
    #        x  equal  y
    #   x  [ 1    1    0 ]   row total 2
    #   eq [ 0    1    0 ]   row total 1
    #   y  [ 1    0    1 ]   row total 2
    #        col totals: x=2, equal=2, y=1        n = 5
    # linear weights w_ij = |i-j|:
    #   observed = sum w_ij * count_ij = (x,equal): 1*1 + (y,x): 2*1 = 1 + 2 = 3
    #   expected = sum w_ij * row_i*col_j / n
    #     (x,equal): 1*2*2=4  (x,y): 2*2*1=4  (eq,x): 1*1*2=2  (eq,y): 1*1*1=1
    #     (y,x): 2*2*2=8      (y,equal): 1*2*2=4
    #     sum = 4+4+2+1+8+4 = 23  ->  expected = 23/5 = 4.6
    #   kappa = 1 - observed/expected = 1 - 3/4.6 = 1 - 15/23 = 8/23 ~= 0.347826
    pairs = [("x", "x"), ("x", "equal"), ("equal", "equal"), ("y", "y"), ("y", "x")]

    kappa = _cohen_kappa(pairs, ("x", "equal", "y"), weighted="linear")

    assert kappa == pytest.approx(8.0 / 23.0, abs=1e-9)


def test_unweighted_kappa_is_one_for_perfect_agreement():
    pairs = [("x", "x"), ("equal", "equal"), ("y", "y")]
    assert _cohen_kappa(pairs, ("x", "equal", "y"), weighted="none") == pytest.approx(1.0)


# --- krippendorff_alpha: worked example, ordinal, tolerant of missing --------

def test_krippendorff_alpha_matches_a_hand_computed_example():
    # Two anchor items, ordinal categories x=0/equal=1/y=2:
    #   anchor 0: r1 -> x, r2 -> equal
    #   anchor 1: r1 -> y, r2 -> y
    # Coincidence matrix (Krippendorff, ordinal difference function):
    #   o_{x,equal} = o_{equal,x} = 1   (from anchor 0, m=2, weight 1/(2-1)=1)
    #   o_{y,y} = 2                     (from anchor 1: 2*1/(2-1) = 2)
    #   marginals: n_x=1, n_equal=1, n_y=2, n=4
    #   delta^2(x,equal)   = (n_x+n_equal - (n_x+n_equal)/2)^2 = (2-1)^2 = 1
    #   delta^2(x,y)       = (n_x+n_equal+n_y - (n_x+n_y)/2)^2 = (4-1.5)^2 = 6.25
    #   delta^2(equal,y)   = (n_equal+n_y - (n_equal+n_y)/2)^2 = (3-1.5)^2 = 2.25
    #   Do = (1/n) * [o_{x,equal}*d(x,eq) + o_{eq,x}*d(eq,x)] = (1/4)*(1+1) = 0.5
    #   De = (1/(n(n-1))) * sum_{c!=d} n_c*n_d*delta^2(c,d)
    #      pairs: (x,eq)=1*1*1=1 (eq,x)=1  (x,y)=1*2*6.25=12.5 (y,x)=12.5
    #             (eq,y)=1*2*2.25=4.5 (y,eq)=4.5  -> sum=1+1+12.5+12.5+4.5+4.5=36
    #      De = 36 / (4*3) = 36/12 = 3.0
    #   alpha = 1 - Do/De = 1 - 0.5/3.0 = 1 - 1/6 = 5/6 ~= 0.833333
    x_candidate = {"params": [0.0], "source": "A"}
    y_candidate = {"params": [1.0], "source": "B"}
    rows = [
        _row("p1", "r1", anchor_id=0, a=x_candidate, b=y_candidate, choice="a"),      # -> x
        _row("p2", "r2", anchor_id=0, a=x_candidate, b=y_candidate, choice="equal"),  # -> equal
        _row("p3", "r1", anchor_id=1, a=y_candidate, b=x_candidate, choice="a"),      # chose y_candidate -> y
        _row("p4", "r2", anchor_id=1, a=x_candidate, b=y_candidate, choice="b"),      # chose y_candidate -> y
    ]

    alpha = krippendorff_alpha(rows)

    assert alpha == pytest.approx(1.0 - 0.5 / 3.0, abs=1e-9)
    assert alpha == pytest.approx(5.0 / 6.0, abs=1e-9)


def test_krippendorff_alpha_is_one_for_perfect_agreement():
    x_candidate = {"params": [0.0], "source": "A"}
    y_candidate = {"params": [1.0], "source": "B"}
    rows = [
        _row("p1", "r1", anchor_id=0, a=x_candidate, b=y_candidate, choice="a"),
        _row("p2", "r2", anchor_id=0, a=x_candidate, b=y_candidate, choice="a"),
        _row("p3", "r1", anchor_id=1, a=x_candidate, b=y_candidate, choice="b"),
        _row("p4", "r2", anchor_id=1, a=y_candidate, b=x_candidate, choice="a"),  # opposite side, same candidate
    ]

    assert krippendorff_alpha(rows) == pytest.approx(1.0)


def test_krippendorff_alpha_tolerates_missing_judgments(tmp_path):
    # A third anchor item judged by only one rater contributes nothing (it
    # can't be paired) but must not raise or distort the other units.
    x_candidate = {"params": [0.0], "source": "A"}
    y_candidate = {"params": [1.0], "source": "B"}
    rows = [
        _row("p1", "r1", anchor_id=0, a=x_candidate, b=y_candidate, choice="a"),
        _row("p2", "r2", anchor_id=0, a=x_candidate, b=y_candidate, choice="a"),
        _row("p3", "r1", anchor_id=1, a=x_candidate, b=y_candidate, choice="a"),
        _row("p4", "r2", anchor_id=1, a=x_candidate, b=y_candidate, choice="a"),
        _row("p5", "r1", anchor_id=2, a=x_candidate, b=y_candidate, choice="a"),  # only r1 judged anchor 2
    ]

    assert krippendorff_alpha(rows) == pytest.approx(1.0)


def test_krippendorff_alpha_is_none_when_there_is_not_enough_paired_anchor_data():
    assert krippendorff_alpha([]) is None


# --- self_consistency: repeats vs. originals, same candidate-identity rule ---

def test_self_consistency_agrees_when_the_repeat_chose_the_same_candidate_on_a_swapped_side():
    rows = [
        _row("orig", "kk", a=P, b=Q, choice="a", repeat_of=None),          # chose P
        _row("rep", "kk", a=Q, b=P, choice="b", repeat_of="orig"),         # chose P, sides swapped
    ]

    result = self_consistency(rows)

    assert result["kk"]["n"] == 1
    assert result["kk"]["raw_agreement"] == pytest.approx(1.0)
    assert result["kk"]["kappa"] == pytest.approx(1.0)


def test_self_consistency_detects_a_real_inconsistency():
    rows = [
        _row("orig", "kk", a=P, b=Q, choice="b", repeat_of=None),          # chose Q
        _row("rep", "kk", a=Q, b=P, choice="b", repeat_of="orig"),         # chose P: inconsistent
    ]

    result = self_consistency(rows)

    assert result["kk"]["n"] == 1
    assert result["kk"]["raw_agreement"] == pytest.approx(0.0)


def test_self_consistency_ignores_rows_without_a_matching_original():
    rows = [_row("rep", "kk", repeat_of="missing")]
    assert self_consistency(rows) == {}
