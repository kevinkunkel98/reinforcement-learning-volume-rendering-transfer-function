import json
import os
import stat

from tools.flag_assisted_rows import flag_rows, main


def _row(timestamp, **extra):
    row = {"pair_id": "p", "timestamp": timestamp, "choice": "a"}
    row.update(extra)
    return row


def test_rows_inside_the_window_are_flagged():
    rows = [_row("2026-09-17T13:20:00")]
    assert flag_rows(rows, "2026-09-17T13:00:00", "2026-09-17T16:00:00")[0]["assisted"] is True


def test_rows_outside_the_window_are_flagged_false_not_dropped():
    rows = [_row("2026-09-16T09:00:00"), _row("2026-09-17T18:00:00")]
    flagged = flag_rows(rows, "2026-09-17T13:00:00", "2026-09-17T16:00:00")
    assert [r["assisted"] for r in flagged] == [False, False]
    assert len(flagged) == 2


def test_an_existing_true_flag_is_never_cleared():
    # Re-running the tool must not un-flag rows flagged by an earlier run with
    # a different window.
    rows = [_row("2026-09-15T09:00:00", assisted=True)]
    assert flag_rows(rows, "2026-09-17T13:00:00", "2026-09-17T16:00:00")[0]["assisted"] is True


def test_flagging_is_idempotent():
    rows = [_row("2026-09-17T13:20:00")]
    once = flag_rows(rows, "2026-09-17T13:00:00", "2026-09-17T16:00:00")
    twice = flag_rows(once, "2026-09-17T13:00:00", "2026-09-17T16:00:00")
    assert once == twice


def test_cli_rewrite_is_atomic_and_preserves_all_rows(tmp_path, monkeypatch):
    # Exercises the real file-rewrite path (temp file + os.replace) against a
    # throwaway fixture -- never the live out/vis_preferences.jsonl.
    pref_path = tmp_path / "vis_preferences.jsonl"
    rows = [
        _row("2026-09-17T13:20:00"),
        _row("2026-09-16T09:00:00"),
        _row("2026-09-15T09:00:00", assisted=True),
    ]
    with open(pref_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    monkeypatch.setattr(
        "sys.argv",
        [
            "flag_assisted_rows.py",
            "--path", str(pref_path),
            "--start", "2026-09-17T13:00:00",
            "--end", "2026-09-17T16:00:00",
        ],
    )
    main()

    with open(pref_path) as f:
        written = [json.loads(line) for line in f if line.strip()]

    assert len(written) == len(rows)
    assert all("assisted" in row for row in written)
    # Original pair_ids/timestamps must survive the rewrite untouched.
    assert [row["timestamp"] for row in written] == [row["timestamp"] for row in rows]
    # No stray temp file left behind in the directory.
    assert list(tmp_path.iterdir()) == [pref_path]


def test_cli_rewrite_preserves_file_permissions(tmp_path, monkeypatch):
    # tempfile.NamedTemporaryFile creates its file at mode 0600 regardless of
    # umask, and os.replace does not carry over the destination's permissions
    # -- so a naive rewrite silently narrows the world-readable preference
    # file to owner-only on every run. The rewrite must go through a fixed
    # path + ".tmp" opened with plain open(), which respects umask instead.
    pref_path = tmp_path / "vis_preferences.jsonl"
    with open(pref_path, "w") as f:
        f.write(json.dumps(_row("2026-09-17T13:20:00")) + "\n")
    os.chmod(pref_path, 0o644)

    monkeypatch.setattr(
        "sys.argv",
        [
            "flag_assisted_rows.py",
            "--path", str(pref_path),
            "--start", "2026-09-17T13:00:00",
            "--end", "2026-09-17T16:00:00",
        ],
    )
    main()

    mode = stat.S_IMODE(os.stat(pref_path).st_mode)
    assert mode == 0o644
