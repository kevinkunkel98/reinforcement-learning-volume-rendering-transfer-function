"""faster-whisper wrapper: transcribe an audio file to text.

Audio is captured in the browser and POSTed to `/api/transcribe`; the
Python-side microphone capture this module used to carry went with the
CLI that called it, and with it the `sounddevice` dependency the web
server was importing at start-up purely to support dead code.
"""

_MODEL_CACHE = {}
SAMPLE_RATE = 16000
AUDIO_DIR = "out/audio"


def _load_model(model_size: str = "small"):
    if model_size in _MODEL_CACHE:
        return _MODEL_CACHE[model_size]
    try:
        import mlx_whisper  # noqa: F401  # pyright: ignore[reportMissingImports]
        _MODEL_CACHE[model_size] = ("mlx", model_size)
        print(f"[asr] using mlx-whisper ({model_size})")
    except ImportError:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        _MODEL_CACHE[model_size] = ("faster-whisper", model)
        print(f"[asr] using faster-whisper ({model_size}, cpu, int8)")
    return _MODEL_CACHE[model_size]


def _transcribe_path(path: str, lang: str, model_size: str) -> str:
    kind, model = _load_model(model_size)
    if kind == "mlx":
        import mlx_whisper  # pyright: ignore[reportMissingImports]
        result = mlx_whisper.transcribe(path, language=lang)
        return result["text"].strip()
    segments, _ = model.transcribe(path, language=lang)
    return " ".join(seg.text.strip() for seg in segments).strip()
