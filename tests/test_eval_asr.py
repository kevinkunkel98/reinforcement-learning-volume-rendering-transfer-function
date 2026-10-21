import pytest

from tools import eval_asr, pair_asr_recordings


def test_word_error_rate():
    assert eval_asr.word_error_rate("show only bones", "show only bones") == 0
    assert eval_asr.word_error_rate("show only bones", "show bones") == pytest.approx(1 / 3)
    assert eval_asr.word_error_rate("darken skeleton", "Darkin' Skeleton.") == pytest.approx(1 / 2)


def test_pairing_takes_recordings_after_the_start_stamp_in_order(tmp_path):
    for stamp in ["20261022-100000", "20261022-100500", "20261022-100600"]:
        (tmp_path / f"{stamp}.wav").write_bytes(b"x")
    rows = pair_asr_recordings.pair(str(tmp_path), "20261022-100100", ["add kidneys", "reset"])
    assert [r["reference"] for r in rows] == ["add kidneys", "reset"]
    assert rows[0]["wav"].endswith("20261022-100500.wav")


def test_pairing_refuses_a_count_mismatch(tmp_path):
    (tmp_path / "20261022-100500.wav").write_bytes(b"x")
    with pytest.raises(ValueError, match="1 recordings for 2 script lines"):
        pair_asr_recordings.pair(str(tmp_path), "20261022-100000", ["a", "b"])
