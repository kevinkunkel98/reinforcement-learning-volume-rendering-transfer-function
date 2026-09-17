from tools.preference_agreement import agreement


def _row(choice, objective_choice, assisted=False, kind="relative"):
    return {
        "choice": choice,
        "objective_choice": objective_choice,
        "assisted": assisted,
        "instruction": {"kind": kind, "text": "more bone"},
    }


def test_agreement_counts_only_decided_unassisted_rows():
    rows = [
        _row("a", "a"),              # agree
        _row("b", "a"),              # disagree
        _row("skip", "a"),           # dropped: no judgment
        _row("equal", "a"),          # dropped from the rate, counted separately
        _row("a", "a", assisted=True),  # dropped: not independent
    ]
    result = agreement(rows)
    assert result["n"] == 2
    assert result["agree"] == 1
    assert result["rate"] == 0.5
    assert result["ties"] == 1
    assert result["skipped"] == 1
    assert result["assisted_excluded"] == 1


def test_confidence_interval_brackets_the_rate_and_stays_in_bounds():
    rows = [_row("a", "a") for _ in range(20)]
    result = agreement(rows)
    lo, hi = result["ci95"]
    assert 0.0 <= lo <= result["rate"] <= hi <= 1.0
    assert lo > 0.5  # 20/20 agreement is not consistent with a coin flip


def test_per_kind_breakdown_splits_by_instruction_kind():
    rows = [_row("a", "a", kind="relative"), _row("b", "a", kind="show_only")]
    per_kind = agreement(rows)["per_kind"]
    assert per_kind["relative"]["rate"] == 1.0
    assert per_kind["show_only"]["rate"] == 0.0


def test_empty_input_reports_no_rate_instead_of_dividing_by_zero():
    result = agreement([])
    assert result["n"] == 0
    assert result["rate"] is None


def test_wilson_interval_matches_published_reference_values():
    # Reference: Wilson 95% CI for 21/38 successes is ~(0.397, 0.700)
    # (e.g. Newcombe 1998, Table II). If this disagrees, the implementation
    # is wrong -- fix the math, not the expected numbers.
    rows = [_row("a", "a") for _ in range(21)] + [_row("b", "a") for _ in range(17)]
    lo, hi = agreement(rows)["ci95"]
    assert round(lo, 2) == 0.40
    assert round(hi, 2) == 0.70


def test_missing_assisted_key_is_treated_as_unassisted():
    # Rows written before flag_assisted_rows.py existed have no "assisted"
    # key at all -- they must not be silently dropped.
    row = {
        "choice": "a",
        "objective_choice": "a",
        "instruction": {"kind": "relative", "text": "more bone"},
    }
    result = agreement([row])
    assert result["n"] == 1
    assert result["assisted_excluded"] == 0
