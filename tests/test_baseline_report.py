import json
import pytest

from tools.baseline_report import run_volume, summarise
from tools.per_class_eval import _class_row, summarise_per_class
import transfer


def test_per_class_cli_parser_accepts_layer_file(tmp_path):
    from tools import per_class_eval

    path = tmp_path / "layers.json"
    path.write_text('{"liver": {"opacity": 0.0}}')

    args = per_class_eval.parse_args(["policy.zip", "--layers-file", str(path)])

    assert args.layers_file == str(path)


def test_baseline_report_scoring_receives_active_layers(monkeypatch):
    from tools import baseline_report

    seen = {}
    class Model:
        def features(self, params, active_layers=None):
            seen["features"] = active_layers
            return {"vis": {}, "bright": {}, "coverage": 0.0}
    monkeypatch.setattr(baseline_report.visibility, "for_volume", lambda name: Model())
    monkeypatch.setattr(baseline_report.goals, "starting_params",
                        lambda: [0.0] * transfer.TOTAL_PARAMS)
    def aggregate(features, active_layers=None):
        seen["aggregate"] = active_layers
        return {"vis": {}, "bright": {}}
    monkeypatch.setattr(baseline_report.goals, "aggregate", aggregate)
    monkeypatch.setattr(baseline_report.goals, "sample_instruction",
                        lambda *args: {"kind": "relative", "goal": {}, "targets": {}})
    monkeypatch.setattr(baseline_report, "BASELINES", {"B0": lambda *args, **kwargs: (
        seen.setdefault("baseline", kwargs.get("active_layers")) or
        [0.0] * transfer.TOTAL_PARAMS)})
    monkeypatch.setattr(baseline_report, "_attainment_or_none", lambda *args: 0.0)

    layers = {"liver": {"opacity": 0.0}}
    run_volume("stub", 1, active_layers=layers)

    assert seen["aggregate"] == layers
    assert seen["baseline"] == layers


def test_baseline_report_persists_active_layer_provenance(tmp_path, monkeypatch):
    from tools import baseline_report

    monkeypatch.setattr(baseline_report, "run_volume", lambda *args, **kwargs: [])
    monkeypatch.setattr(baseline_report, "_print_table", lambda summary: None)
    out = tmp_path / "report.json"

    baseline_report.main(["--volumes", "stub", "--instructions", "0", "--layers",
                          '{"liver": {"opacity": 0.0}}', "--out", str(out)])

    payload = json.loads(out.read_text())
    assert payload["provenance"]["anatomy_layers"]["liver"]["opacity"] == 0.0


def test_baseline_report_rejects_existing_provenance_free_report(tmp_path, monkeypatch):
    from tools import baseline_report

    out = tmp_path / "old.json"
    out.write_text(json.dumps({"summary": {}, "rows": []}))
    monkeypatch.setattr(baseline_report, "run_volume", lambda *args, **kwargs: [])
    monkeypatch.setattr(baseline_report, "_print_table", lambda summary: None)

    with pytest.raises(ValueError, match="provenance"):
        baseline_report.main(["--volumes", "stub", "--instructions", "0", "--out", str(out)])


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


def test_per_class_report_keeps_reachability_counts_and_attainment():
    rows = [
        {"class": "liver", "status": "reachable", "attainment": 0.8},
        {"class": "liver", "status": "unsupported", "attainment": None},
        {"class": "heart", "status": "unreachable", "attainment": None},
        {"class": "heart", "status": "reachable", "attainment": -0.2},
    ]

    report = summarise_per_class(rows)

    assert report["liver"] == {
        "median": 0.8,
        "share_positive": 1.0,
        "n": 1,
        "supported": 1,
        "reachable": 1,
        "unsupported": 1,
        "unreachable": 0,
    }
    assert report["heart"]["median"] == pytest.approx(-0.2)
    assert report["heart"]["unsupported"] == 0
    assert report["heart"]["unreachable"] == 1


def test_per_class_report_counts_all_statuses_even_without_scores():
    rows = [
        {"class": "liver", "status": "unsupported", "attainment": None},
        {"class": "liver", "status": "unreachable", "attainment": None},
    ]

    report = summarise_per_class(rows)

    assert report["liver"]["supported"] == 1
    assert report["liver"]["reachable"] == 0
    assert report["liver"]["unsupported"] == 1
    assert report["liver"]["unreachable"] == 1
    assert report["liver"]["n"] == 0


def test_per_class_row_drops_attainment_for_nonreachable_class():
    assert _class_row("liver", "unsupported", 0.8)["attainment"] is None
    assert _class_row("liver", "unreachable", 0.8)["attainment"] is None
    assert _class_row("liver", "reachable", 0.8)["attainment"] == 0.8
