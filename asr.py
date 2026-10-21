"""Speech recognition for voice commands: audio file -> text.

Audio is captured in the browser and POSTed to `/api/transcribe`. Two
backends: mlx-whisper (Apple GPU) and faster-whisper (CPU, int8); ASR_BACKEND
picks one, otherwise mlx-whisper when installed. ASR_MODEL picks the size.
"""
import os
import re

_MODEL_CACHE = {}
SAMPLE_RATE = 16000
AUDIO_DIR = "out/audio"
ASR_BACKEND = os.environ.get("ASR_BACKEND")  # "mlx" | "faster-whisper" | None
ASR_MODEL = os.environ.get("ASR_MODEL", "small")
MLX_REPOS = {"small": "mlx-community/whisper-small-mlx", "turbo": "mlx-community/whisper-large-v3-turbo"}
FASTER_WHISPER_NAMES = {"small": "small", "turbo": "large-v3-turbo"}

# What Whisper says on silence or noise (2026-10-21 VR session: "You" eleven
# times, invented sentences twice). Commands are short.
HALLUCINATIONS = {"you", "thank you", "thanks", "thank you for watching", "thanks for watching", "bye",
                  "okay", "ok", "so", "um", "uh"}
MAX_COMMAND_WORDS = 20


def is_hallucination(text: str) -> bool:
    t = re.sub(r"[^a-z' ]", "", text.lower()).strip()
    return not t or t in HALLUCINATIONS or len(t.split()) > MAX_COMMAND_WORDS


def _default_backend() -> str:
    if ASR_BACKEND:
        return ASR_BACKEND
    try:
        import mlx_whisper  # noqa: F401  # pyright: ignore[reportMissingImports]
        return "mlx"
    except ImportError:
        return "faster-whisper"


def _load_model(model_size: str, backend: str):
    key = (backend, model_size)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if backend == "mlx":
        _MODEL_CACHE[key] = ("mlx", MLX_REPOS[model_size])
        print(f"[asr] using mlx-whisper ({MLX_REPOS[model_size]})")
    else:
        from faster_whisper import WhisperModel
        model = WhisperModel(FASTER_WHISPER_NAMES[model_size], device="cpu", compute_type="int8")
        _MODEL_CACHE[key] = ("faster-whisper", model)
        print(f"[asr] using faster-whisper ({FASTER_WHISPER_NAMES[model_size]}, cpu, int8)")
    return _MODEL_CACHE[key]


def _transcribe_path(path: str, lang: str, model_size: str = ASR_MODEL, initial_prompt: str = None,
                     backend: str = None) -> str:
    kind, model = _load_model(model_size, backend or _default_backend())
    if kind == "mlx":
        import mlx_whisper  # pyright: ignore[reportMissingImports]
        result = mlx_whisper.transcribe(path, path_or_hf_repo=model, language=lang,
                                        initial_prompt=initial_prompt, condition_on_previous_text=False)
        return result["text"].strip()
    segments, _ = model.transcribe(path, language=lang, initial_prompt=initial_prompt, vad_filter=True,
                                   condition_on_previous_text=False)
    return " ".join(seg.text.strip() for seg in segments).strip()
