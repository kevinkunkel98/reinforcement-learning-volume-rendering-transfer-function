"""Tests for `provenance`: what a result file has to say about the code that
produced it, and how a stale recording is caught."""
import subprocess
import types

import provenance


def _module_at(path, source: str, name: str = "fake_scoring"):
    """A stand-in module object whose `__file__` points at `path` -- enough
    for the fingerprint, which only ever reads the file, never the module."""
    path.write_text(source)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    return module


# --- provenance() ---------------------------------------------------------------

def test_provenance_records_commit_dirtiness_fingerprint_python_and_time():
    record = provenance.provenance()

    assert set(record) >= {"git_commit", "git_dirty", "scoring_fingerprint",
                           "python", "timestamp"}
    # This repo is a git checkout, so the commit must actually resolve here.
    assert isinstance(record["git_commit"], str) and len(record["git_commit"]) == 40
    assert isinstance(record["git_dirty"], bool)
    assert isinstance(record["python"], str) and record["python"]
    assert isinstance(record["timestamp"], str) and "T" in record["timestamp"]
    assert "anatomy_layers" in record


def test_scoring_fingerprint_is_twelve_hex_characters():
    fingerprint = provenance.provenance()["scoring_fingerprint"]
    assert len(fingerprint) == 12
    assert all(character in "0123456789abcdef" for character in fingerprint)


def test_provenance_can_record_the_active_anatomy_layers():
    record = provenance.provenance(anatomy_layers={"liver": {"opacity": 0.0}})

    assert record["anatomy_layers"]["liver"]["opacity"] == 0.0
    assert record["label_layout"] == "anatomy-v2"


def test_result_provenance_overlays_active_layers_on_import_snapshot():
    record = provenance.result_provenance({"liver": {"opacity": 0.0}})

    assert record["anatomy_layers"]["liver"]["opacity"] == 0.0
    assert record["scoring_fingerprint"] == provenance.IMPORT_TIME_PROVENANCE["scoring_fingerprint"]
    assert record["timestamp"] == provenance.IMPORT_TIME_PROVENANCE["timestamp"]


def test_result_provenance_marks_unavailable_active_layers_explicitly():
    assert provenance.result_provenance()["anatomy_layers"] is None


def test_fingerprint_is_read_from_disk_so_editing_a_scoring_module_changes_it(tmp_path):
    path = tmp_path / "scoring.py"
    module = _module_at(path, "def score(x):\n    return x\n")

    before = provenance.provenance(modules=[module])["scoring_fingerprint"]
    path.write_text("def score(x):\n    return x + 1\n")   # the 22:48 bug fix
    after = provenance.provenance(modules=[module])["scoring_fingerprint"]

    assert before != after


def test_fingerprint_is_stable_for_unchanged_sources(tmp_path):
    module = _module_at(tmp_path / "scoring.py", "def score(x):\n    return x\n")
    assert (provenance.provenance(modules=[module])["scoring_fingerprint"]
            == provenance.provenance(modules=[module])["scoring_fingerprint"])


def test_fingerprint_covers_every_module_not_just_the_first(tmp_path):
    first = _module_at(tmp_path / "a.py", "A = 1\n", name="a")
    second = _module_at(tmp_path / "b.py", "B = 1\n", name="b")

    before = provenance.provenance(modules=[first, second])["scoring_fingerprint"]
    (tmp_path / "b.py").write_text("B = 2\n")
    after = provenance.provenance(modules=[first, second])["scoring_fingerprint"]

    assert before != after


def test_default_modules_are_the_ones_that_compute_a_score():
    assert set(provenance.DEFAULT_SCORING_MODULES) == {
        "goals", "visibility", "rl.vis_eval", "rl.oneshot_env",
        "tools.per_class_eval"}
    assert provenance.provenance()["scoring_modules"] == list(
        provenance.DEFAULT_SCORING_MODULES)


def test_unreadable_module_source_does_not_raise(tmp_path):
    missing = types.ModuleType("gone")
    missing.__file__ = str(tmp_path / "never_written.py")
    record = provenance.provenance(modules=[missing])
    assert len(record["scoring_fingerprint"]) == 12


# --- git, degrading gracefully ---------------------------------------------------

def test_git_fields_are_none_when_git_is_not_installed(monkeypatch):
    def _no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(provenance.subprocess, "run", _no_git)
    record = provenance.provenance()

    assert record["git_commit"] is None
    assert record["git_dirty"] is None
    # The rest still has to be recorded: provenance must never be all-or-nothing.
    assert len(record["scoring_fingerprint"]) == 12


def test_git_fields_are_none_outside_a_repository(monkeypatch):
    def _not_a_repo(*args, **kwargs):
        return subprocess.CompletedProcess(args, 128, "", "fatal: not a git repository")

    monkeypatch.setattr(provenance.subprocess, "run", _not_a_repo)
    record = provenance.provenance()

    assert record["git_commit"] is None
    assert record["git_dirty"] is None


def test_git_dirty_is_true_when_status_reports_a_modified_file(monkeypatch):
    def _fake_git(command, **kwargs):
        text = "abc\n" if "rev-parse" in command else " M goals.py\n"
        return subprocess.CompletedProcess(command, 0, text, "")

    monkeypatch.setattr(provenance.subprocess, "run", _fake_git)
    assert provenance.provenance()["git_dirty"] is True


def test_git_dirty_is_false_when_status_is_empty(monkeypatch):
    def _fake_git(command, **kwargs):
        text = "abc\n" if "rev-parse" in command else ""
        return subprocess.CompletedProcess(command, 0, text, "")

    monkeypatch.setattr(provenance.subprocess, "run", _fake_git)
    assert provenance.provenance()["git_dirty"] is False


# --- import-time capture ----------------------------------------------------------

def test_import_time_provenance_is_captured_once_at_import():
    assert isinstance(provenance.IMPORT_TIME_PROVENANCE, dict)
    assert set(provenance.IMPORT_TIME_PROVENANCE) >= {
        "git_commit", "git_dirty", "scoring_fingerprint", "python", "timestamp"}
    # Captured, not recomputed: it must not follow the working tree while a
    # long job runs, so reading it twice cannot produce two timestamps.
    assert (provenance.IMPORT_TIME_PROVENANCE["timestamp"]
            == provenance.IMPORT_TIME_PROVENANCE["timestamp"])
    assert provenance.IMPORT_TIME_PROVENANCE["scoring_fingerprint"] is not None


# --- compare() ---------------------------------------------------------------------

def test_compare_reports_fresh_when_nothing_changed():
    record = provenance.provenance()
    report = provenance.compare(record, record)
    assert report["stale"] is False
    assert report["reasons"] == []


def test_compare_flags_a_changed_scoring_fingerprint_with_both_hashes():
    recorded = {"git_commit": "a" * 40, "git_dirty": False,
                "scoring_fingerprint": "aaaaaaaaaaaa"}
    current = {"git_commit": "a" * 40, "git_dirty": False,
               "scoring_fingerprint": "bbbbbbbbbbbb"}

    report = provenance.compare(recorded, current)

    assert report["stale"] is True
    [reason] = report["reasons"]
    assert "aaaaaaaaaaaa" in reason and "bbbbbbbbbbbb" in reason
    assert "fingerprint" in reason


def test_compare_flags_a_changed_commit():
    recorded = {"git_commit": "a" * 40, "git_dirty": False, "scoring_fingerprint": "f" * 12}
    current = {"git_commit": "b" * 40, "git_dirty": False, "scoring_fingerprint": "f" * 12}

    report = provenance.compare(recorded, current)

    assert report["stale"] is True
    assert any("commit" in reason for reason in report["reasons"])


def test_compare_flags_a_changed_anatomy_layer_layout():
    recorded = provenance.provenance()
    current = {**recorded, "anatomy_layer_layout": "anatomy-layers-v2"}

    report = provenance.compare(recorded, current)

    assert report["stale"] is True
    assert any("anatomy_layer_layout" in reason for reason in report["reasons"])


def test_compare_ignores_runtime_layer_difference_without_expected_layers():
    recorded = provenance.result_provenance({"liver": {"opacity": 0.0}})
    current = provenance.result_provenance()

    report = provenance.compare(recorded, current)

    assert report["stale"] is False


def test_compare_checks_runtime_layers_when_expected_layers_are_explicit():
    recorded = provenance.result_provenance({"liver": {"opacity": 0.0}})
    current = provenance.result_provenance()

    report = provenance.compare(recorded, current, expected_layers={"liver": {"opacity": 1.0}})

    assert report["stale"] is True
    assert any("anatomy_layers" in reason for reason in report["reasons"])


def test_compare_treats_a_result_without_provenance_as_stale():
    current = {"git_commit": "a" * 40, "git_dirty": False, "scoring_fingerprint": "f" * 12}
    for recorded in (None, {}):
        report = provenance.compare(recorded, current)
        assert report["stale"] is True
        assert report["reasons"]


def test_compare_defaults_to_current_provenance():
    report = provenance.compare({"scoring_fingerprint": "deadbeefcafe",
                                 "git_commit": "0" * 40})
    assert report["stale"] is True
    assert any("deadbeefcafe" in reason for reason in report["reasons"])


def test_compare_reports_that_the_recorded_run_had_a_dirty_tree():
    record = {"git_commit": "a" * 40, "git_dirty": True, "scoring_fingerprint": "f" * 12}
    report = provenance.compare(record, dict(record))
    # Not stale -- the code matches -- but the recording itself is untrustworthy,
    # because the tree it ran from was never committed anywhere.
    assert report["stale"] is False
    assert report["recorded_dirty"] is True


def test_compare_does_not_guess_when_a_side_is_missing_a_field():
    recorded = {"git_commit": None, "git_dirty": None, "scoring_fingerprint": "f" * 12}
    current = {"git_commit": "b" * 40, "git_dirty": False, "scoring_fingerprint": "f" * 12}
    report = provenance.compare(recorded, current)
    assert report["stale"] is False


def test_staleness_banner_is_loud_and_names_the_reasons():
    report = {"stale": True, "reasons": ["scoring code changed: fingerprint aaa -> bbb"],
              "recorded_dirty": False}
    banner = provenance.format_staleness(report)
    assert "STALE" in banner
    assert "scoring code changed: fingerprint aaa -> bbb" in banner


def test_staleness_banner_is_empty_for_a_fresh_result():
    assert provenance.format_staleness(
        {"stale": False, "reasons": [], "recorded_dirty": False}) == ""


def test_staleness_banner_stays_empty_for_a_dirty_but_matching_result():
    # A dirty tree is the normal state during development; a banner that fires
    # on every run would be trained away within a day.
    assert provenance.format_staleness(
        {"stale": False, "reasons": [], "recorded_dirty": True}) == ""


def test_staleness_banner_mentions_a_dirty_tree_once_something_is_already_wrong():
    banner = provenance.format_staleness(
        {"stale": True, "reasons": ["git commit changed: aaa -> bbb"], "recorded_dirty": True})
    assert "uncommitted" in banner
