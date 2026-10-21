import pytest

import asr


@pytest.mark.parametrize("text", ["", "You", "you.", "Thank you.", "Thanks for watching!", "Bye",
                                  "We're in a small town and I've just tried to do the right things. Should I go "
                                  "to high school? No, no. Should I go to high school? Yeah, yeah, yeah. It was good."])
def test_hallucinations_are_recognised(text):
    assert asr.is_hallucination(text)


@pytest.mark.parametrize("text", ["Show only bones", "REVERT", "Darkin' Skeleton", "add kidneys"])
def test_commands_are_not_hallucinations(text):
    assert not asr.is_hallucination(text)


def test_faster_whisper_gets_prompt_and_vad(monkeypatch):
    seen = {}

    class Model:
        def transcribe(self, path, **kwargs):
            seen.update(kwargs)
            return [type("S", (), {"text": " show only bones "})()], None
    monkeypatch.setitem(asr._MODEL_CACHE, ("faster-whisper", "small"), ("faster-whisper", Model()))
    text = asr._transcribe_path("x.wav", "en", "small", initial_prompt="skeleton, lungs", backend="faster-whisper")
    assert text == "show only bones"
    assert seen["initial_prompt"] == "skeleton, lungs"
    assert seen["vad_filter"] is True
    assert seen["condition_on_previous_text"] is False


def test_mlx_uses_the_requested_model_size(monkeypatch):
    seen = {}

    class FakeMlx:
        @staticmethod
        def transcribe(path, **kwargs):
            seen.update(kwargs)
            return {"text": " reset "}
    monkeypatch.setitem(__import__("sys").modules, "mlx_whisper", FakeMlx)
    assert asr._transcribe_path("x.wav", "en", "turbo", initial_prompt="p", backend="mlx") == "reset"
    assert seen["path_or_hf_repo"] == asr.MLX_REPOS["turbo"]
    assert seen["initial_prompt"] == "p"
