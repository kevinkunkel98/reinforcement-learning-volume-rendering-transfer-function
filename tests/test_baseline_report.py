import pytest

from tools.baseline_report import summarise


def _rows():
    return [
        {"volume": "v1", "kind": "relative", "baseline": "B1", "attainment": 0.5},
        {"volume": "v1", "kind": "relative", "baseline": "B1", "attainment": 1.0},
        {"volume": "v1", "kind": "show_only", "baseline": "B1", "attainment": 0.0},
        {"volume": "v1", "kind": "relative", "baseline": "B2", "attainment": -0.5},
        {"volume": "v1", "kind": "relative", "baseline": "B1", "attainment": None},
    ]


def test_summarise_means_and_counts_per_baseline():
    # by_baseline now reports goals.summarise_attainment's robust stats
    # rather than a plain mean -- B1's values are [0.5, 1.0, 0.0].
    summary = summarise(_rows())
    assert summary["by_baseline"]["B1"]["median"] == pytest.approx(0.5)
    assert summary["by_baseline"]["B1"]["mean_raw"] == pytest.approx(0.5)
    assert summary["by_baseline"]["B1"]["mean_clipped"] == pytest.approx(0.5)
    assert summary["by_baseline"]["B1"]["share_positive"] == pytest.approx(2.0 / 3.0)
    assert summary["by_baseline"]["B1"]["n"] == 3
    assert summary["by_baseline"]["B2"]["mean_raw"] == pytest.approx(-0.5)
    assert summary["by_baseline"]["B2"]["n"] == 1


def test_summarise_means_and_counts_per_kind():
    summary = summarise(_rows())
    assert summary["by_kind"]["B1"]["relative"]["mean"] == pytest.approx(0.75)
    assert summary["by_kind"]["B1"]["relative"]["n"] == 2
    assert summary["by_kind"]["B1"]["show_only"]["mean"] == pytest.approx(0.0)
    assert summary["by_kind"]["B1"]["show_only"]["n"] == 1
    assert summary["by_kind"]["B2"]["relative"]["n"] == 1


def test_summarise_ignores_none_entirely():
    summary = summarise(_rows())
    total_n = sum(entry["n"] for entry in summary["by_baseline"].values())
    assert total_n == 4  # the one None row is not counted anywhere


def test_summarise_handles_empty_rows():
    summary = summarise([])
    assert summary["by_baseline"] == {}
    assert summary["by_kind"] == {}
