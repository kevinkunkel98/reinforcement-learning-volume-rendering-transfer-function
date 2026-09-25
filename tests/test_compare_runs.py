import json

import pytest

from anatomy_layers import normalize_layers
from tools.compare_runs import compare_groups, seed_average


def _detail(attainments, kinds=None, volumes=None):
    kinds = kinds if kinds is not None else ["relative"] * len(attainments)
    volumes = volumes if volumes is not None else [f"ts_{i}" for i in range(len(attainments))]
    return [{"attainment": a, "kind": k, "volume": v}
            for a, k, v in zip(attainments, kinds, volumes)]


def _result_file(tmp_path, name, attainments, kinds=None, volumes=None, detail=True,
                 starts=None, targets=None, goals=None, provenance=None):
    payload = {"split": "test", "seed": 0, "episodes": len(attainments),
               "policy": f"out/{name}/best.zip"}
    if detail:
        rows = _detail(attainments, kinds, volumes)
        for index, row in enumerate(rows):
            row["start_params"] = (starts or [[0.0] for _ in rows])[index]
            row["targets"] = (targets or [{"skeleton": {"vis": 0.1}} for _ in rows])[index]
            row["goal"] = (goals or [[0.1] for _ in rows])[index]
        payload["episodes_detail"] = {"policy": rows}
    if provenance is not None:
        payload["provenance"] = provenance
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload))
    return str(path)


def test_seed_average_averages_each_episode_across_the_files():
    groups = [_detail([0.0, 1.0]), _detail([1.0, 3.0]), _detail([2.0, 2.0])]

    averaged = seed_average(groups)

    assert [row["attainment"] for row in averaged] == [1.0, 2.0]
    assert [row["kind"] for row in averaged] == ["relative", "relative"]


def test_compare_reports_both_medians_and_the_delta_between_them(tmp_path):
    a = [_result_file(tmp_path, "a0", [0.0, 0.2, 0.4])]
    b = [_result_file(tmp_path, "b0", [0.1, 0.5, 0.9])]

    result = compare_groups(a, b)

    assert result["a"]["median"] == pytest.approx(0.2)
    assert result["b"]["median"] == pytest.approx(0.5)
    assert result["delta"] == pytest.approx(0.3)
    assert result["n_episodes"] == 3


def test_compare_counts_the_share_of_episodes_the_second_group_wins(tmp_path):
    a = [_result_file(tmp_path, "a0", [0.0, 0.5, 0.2, 0.9])]
    b = [_result_file(tmp_path, "b0", [0.1, 0.4, 0.3, 1.0])]

    result = compare_groups(a, b)

    assert result["share_b_ahead"] == pytest.approx(0.75)
    assert 0.0 <= result["p_value"] <= 1.0


def test_compare_reports_the_delta_per_instruction_kind(tmp_path):
    kinds = ["absolute", "absolute", "compound", "compound"]
    a = [_result_file(tmp_path, "a0", [0.1, 0.3, 0.0, 0.2], kinds=kinds)]
    b = [_result_file(tmp_path, "b0", [0.5, 0.7, 0.1, 0.1], kinds=kinds)]

    result = compare_groups(a, b)

    assert result["by_kind"]["absolute"]["delta"] == pytest.approx(0.4)
    assert result["by_kind"]["compound"]["delta"] == pytest.approx(0.0)
    assert result["by_kind"]["absolute"]["n"] == 2


def test_compare_refuses_a_result_file_written_before_per_episode_rows(tmp_path):
    a = [_result_file(tmp_path, "a0", [0.1, 0.2], detail=False)]
    b = [_result_file(tmp_path, "b0", [0.3, 0.4])]

    with pytest.raises(ValueError, match="a0.json"):
        compare_groups(a, b)


def test_compare_refuses_groups_that_were_not_scored_on_the_same_episodes(tmp_path):
    a = [_result_file(tmp_path, "a0", [0.1, 0.2], volumes=["ts_0", "ts_1"])]
    b = [_result_file(tmp_path, "b0", [0.3, 0.4], volumes=["ts_0", "ts_9"])]

    with pytest.raises(ValueError, match="episode 1"):
        compare_groups(a, b)


def test_compare_refuses_different_episode_start_or_instruction_identity(tmp_path):
    provenance = {"scoring_fingerprint": "a", "label_layout": "anatomy-v2",
                  "anatomy_layer_layout": "anatomy-layers-v1", "visibility_renderer": "visibility-v1",
                  "anatomy_layers": None}
    a = [_result_file(tmp_path, "a0", [0.1], starts=[[0.0]], provenance=provenance)]
    b = [_result_file(tmp_path, "b0", [0.3], starts=[[1.0]], provenance=provenance)]

    with pytest.raises(ValueError, match="episode 0"):
        compare_groups(a, b)


def test_compare_refuses_incompatible_provenance_or_active_layers(tmp_path):
    base = {"scoring_fingerprint": "a", "label_layout": "anatomy-v2",
            "anatomy_layer_layout": "anatomy-layers-v1", "visibility_renderer": "visibility-v1",
            "anatomy_layers": None}
    a = [_result_file(tmp_path, "a0", [0.1], provenance=base)]
    b = [_result_file(tmp_path, "b0", [0.3], provenance={**base, "anatomy_layers": {"liver": {"opacity": 0.0}}})]

    with pytest.raises(ValueError, match="provenance"):
        compare_groups(a, b)


def test_compare_accepts_matching_custom_active_layers(tmp_path):
    layers = {"liver": {"opacity": 0.0}}
    base = {"scoring_fingerprint": "a", "label_layout": "anatomy-v2",
            "anatomy_layer_layout": "anatomy-layers-v1", "visibility_renderer": "visibility-v1",
            "anatomy_layers": normalize_layers(layers)}
    a = [_result_file(tmp_path, "a-custom", [0.1], provenance=base)]
    b = [_result_file(tmp_path, "b-custom", [0.3], provenance=base)]

    assert compare_groups(a, b)["n_episodes"] == 1
