import json
import pytest

from tools.baseline_report import run_volume, summarise
from tools.per_class_eval import _class_row, summarise_per_class
from tools import per_class_eval
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


def test_expanded_hill_climb_baseline_uses_fixed_episode_budget():
    assert per_class_eval.BASELINE_EVALUATIONS["expanded_hill_climb"] > 200


def test_expanded_hill_climb_name_is_reportable():
    assert per_class_eval.baseline_name("expanded_hill_climb") == "expanded_hill_climb"


def test_expanded_hill_climb_passes_fixed_budget(monkeypatch):
    seen = {}
    monkeypatch.setattr(per_class_eval, "hill_climb",
                        lambda *args, **kwargs: seen.update(kwargs) or "params")
    assert per_class_eval.expanded_hill_climb("model", "start", "instruction") == "params"
    assert seen["evaluations"] == per_class_eval.BASELINE_EVALUATIONS["expanded_hill_climb"]


def test_per_class_cli_accepts_selectable_baseline():
    args = per_class_eval.parse_args(["policy.zip", "--baseline", "expanded_hill_climb"])
    assert args.baseline == ["expanded_hill_climb"]


def test_per_class_policy_score_uses_episode_action_mode(monkeypatch):
    seen = {}
    monkeypatch.setattr(per_class_eval, "_apply_action",
                        lambda start, action, action_mode="absolute": seen.setdefault("mode", action_mode) or start)
    episode = {"start_params": [0.0], "instruction": {"goal": {}, "targets": { }, "kind": "relative"},
               "policy_metadata": {"action_mode": "residual"}}
    model = type("Model", (), {"features": lambda self, params: {"vis": {}, "bright": {}, "coverage": 0.0}})()
    monkeypatch.setattr(per_class_eval.goals, "aggregate", lambda *args, **kwargs: {})
    monkeypatch.setattr(per_class_eval, "_observation_for", lambda *args: [])
    monkeypatch.setattr(per_class_eval, "_predict", lambda *args: [0.0])
    monkeypatch.setattr(per_class_eval.goals, "attainment", lambda *args: 0.0)
    per_class_eval._score(object(), episode, model)
    assert seen["mode"] == "residual"


def test_baseline_rows_use_actual_volume_reachability(monkeypatch):
    monkeypatch.setattr(per_class_eval.goals, "goal_classes_for_volume", lambda volume: ["skeleton"])
    monkeypatch.setattr(per_class_eval.goals, "reachable_goal_classes", lambda volume, model, layers: [])
    rows = per_class_eval._baseline_class_rows(
        {"volume": "stub", "instruction": {"targets": {"skeleton": {"vis": 0.3}}}},
        object(), None, 0.0)
    assert rows[0]["status"] == "unreachable"


def test_per_class_eval_uses_each_policy_sidecar_contract(monkeypatch):
    calls = []
    episodes = [{"volume": "stub", "start_params": [0.0],
                 "instruction": {"targets": {}, "goal": {}, "kind": "relative"},
                 "policy_metadata": {"policy_version": "oneshot-v6",
                                     "action_mode": "absolute", "reward_mode": "attainment"}}]
    monkeypatch.setattr(per_class_eval, "load_policy_metadata", lambda path: {
        "policy_version": "oneshot-v7" if path == "v7.zip" else "oneshot-v6",
        "action_mode": "residual" if path == "v7.zip" else "absolute",
        "reward_mode": "target" if path == "v7.zip" else "attainment"})
    def make_episodes(*args, **kwargs):
        calls.append((args, kwargs["policy_metadata"]))
        return [dict(episodes[0], policy_metadata=kwargs["policy_metadata"])]
    monkeypatch.setattr(per_class_eval, "fixed_episodes", make_episodes)
    monkeypatch.setattr(per_class_eval.visibility, "for_volume", lambda name: object())
    monkeypatch.setattr(per_class_eval.goals, "goal_classes_for_volume", lambda volume: [])
    monkeypatch.setattr(per_class_eval.goals, "reachable_goal_classes", lambda volume, model, layers: [])
    import stable_baselines3
    monkeypatch.setattr(stable_baselines3.SAC, "load", lambda path, device: object())
    monkeypatch.setattr(per_class_eval, "_score", lambda *args: 0.0)
    per_class_eval.main(argv=["v6.zip", "v7.zip", "--episodes", "1"])

    assert [metadata["policy_version"] for _, metadata in calls] == ["oneshot-v6", "oneshot-v7"]
    assert calls[0][0][1:3] == calls[1][0][1:3]
