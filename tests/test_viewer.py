"""The viewer: a session answering commands in every mode, two-stage policy
answers, restores and hints, the comparison panel and the routes. Renders
with VTK on the synthetic scans (main thread, as the server does)."""
import asyncio

import numpy as np
import pytest

import answers
import goals
import server
import transfer
from session import Session, no_visible_change_hint, restore_hidden, scan_label, scan_list


@pytest.fixture
def session(tmp_path, policy):
    return Session("ts_s9001", path=str(tmp_path / "session.json"), policy_provider=lambda: policy)


def _height(step, name):
    return step["params"][transfer.ANATOMICAL_PEAK_INDEX[name] * transfer.PARAMS_PER_PEAK + 2]


def test_a_session_starts_on_the_task_start_and_resumes(session):
    first = session.state()
    assert first["total"] == 1 and first["current"]["params"] == goals.starting_params().tolist()
    assert first["current"]["image_b64"] and first["anatomy"]["label_available"]
    session.command("more liver", mode="exact")
    resumed = Session("ts_s9001", path=session.path)
    assert resumed.state()["total"] == 2 and resumed.session_id == session.session_id


def test_exact_search_and_camera_commands(session):
    exact = session.command("more liver", mode="exact")["current"]
    assert exact["mode"] == "exact" and _height(exact, "liver") > goals.starting_params()[
        transfer.ANATOMICAL_PEAK_INDEX["liver"] * transfer.PARAMS_PER_PEAK + 2]
    searched = session.command("a bit less lungs", mode="search", steps=30)["current"]
    assert searched["mode"] == "search" and searched["search"]
    turned = session.command("rotate left", mode="exact")["current"]
    assert turned["mode"] == "camera" and turned["params"] == searched["params"]
    assert turned["camera"] != searched["camera"]


def test_policy_answers_in_two_stages(session):
    fast = session.command("more liver", mode="policy", defer_refine=True)["current"]
    assert fast["mode"] == "policy" and fast["refining"]
    job = session.pending_refines[fast["id"]]
    assert session.apply_refined(fast["id"], answers.full_answer(job), job)
    final = session.state()["current"]
    assert final["id"] == fast["id"] and final["refined"] and not final["refining"]


def test_a_stale_refine_is_discarded(session):
    fast = session.command("more liver", mode="policy", defer_refine=True)["current"]
    job = session.pending_refines[fast["id"]]
    session.command("less lungs", mode="exact")       # the user moved on
    assert not session.apply_refined(fast["id"], answers.full_answer(job), job)


def test_policy_mode_without_a_checkpoint_applies_the_command(tmp_path):
    session = Session("ts_s9001", path=str(tmp_path / "s.json"), policy_provider=lambda: None)
    step = session.command("more liver", mode="policy")["current"]
    assert step["mode"] == "exact" and step["message"].startswith(answers.NO_POLICY_MESSAGE)


def test_adding_a_hidden_class_restores_it(session):
    session.command("show only the skeleton", mode="exact")
    step = session.command("add liver", mode="policy")["current"]
    assert "liver was hidden -- restored to its start" in step["message"]
    start = goals.starting_params()
    base = transfer.ANATOMICAL_PEAK_INDEX["liver"] * transfer.PARAMS_PER_PEAK
    assert step["params"][base:base + 6] == start[base:base + 6].tolist()


def test_restore_rewrites_only_hidden_increases():
    cmd = {"compound": [{"target": "liver", "attribute": "opacity", "direction": "increase", "strength": "moderately"},
                        {"target": "lungs", "attribute": "opacity", "direction": "increase", "strength": "moderately"}]}
    new, restored = restore_hidden(cmd, {"liver": 0.0, "lungs": 0.05})
    assert restored == ["liver"] and new["compound"][0]["direction"] == "reset"
    assert new["compound"][1] == cmd["compound"][1]


def test_no_visible_change_hint(session):
    step = session.state()["current"]
    hint = no_visible_change_hint({"target": "kidneys", "attribute": "brightness", "direction": "increase"}, step, step)
    assert hint == "no visible change -- kidneys mostly hidden behind other tissue; try 'show only kidneys'"


def test_comparison_panel_answers_every_arm(session, policy):
    model = session.model()
    params = goals.starting_params()
    cmd = {"target": "liver", "attribute": "opacity", "direction": "increase", "strength": "moderately"}
    start_agg, instruction = session.instruction_for(cmd, model, params)
    arms = answers.compare(model, policy, cmd, instruction, start_agg, params,
                           lambda p: session.view(p, session.state()["current"]["camera"]), cheap=5, thorough=20)
    assert set(arms) == {"exact", "search_cheap", "search_thorough", "policy", "policy_keep"}
    assert all(np.isfinite(arm["attainment"]) and arm["image_b64"] for arm in arms.values())
    summary = answers.sweep(model, policy, "ts_s9001", params, episodes=3)
    assert summary["arms"]["policy"]["n"] == 3


def test_scan_list_and_labels():
    assert scan_list() == ["ts_s9001", "ts_s9002"]
    assert scan_label("ts_s9001") == "s9001 · thorax–pelvis"


def test_routes(session, monkeypatch):
    monkeypatch.setattr(server, "_session", session)
    listing = asyncio.run(server.datasets_list())
    assert listing["current"] == "ts_s9001" and listing["labels"]["ts_s9002"] == "s9002 · thorax–pelvis"
    metadata = asyncio.run(server.volume_metadata("ts_s9001"))
    chunk = asyncio.run(server.volume_chunk("ts_s9001", "0"))
    assert metadata["dimensions"] == [40, 32, 48] and len(chunk.body) == metadata["chunks"][0]["byte_length"]
    with pytest.raises(server.HTTPException):
        asyncio.run(server.command(server.CommandRequest(text="what a lovely day", parser="rule")))
    result = asyncio.run(server.command(server.CommandRequest(text="more liver", parser="rule", mode="exact")))
    assert result["current"]["cmd_text"] == "more liver"
    switched = asyncio.run(server.dataset_switch(server.DatasetRequest(name="ts_s9002")))
    assert switched["dataset"] == "ts_s9002" and switched["total"] == 1
