"""Speech recognition for voice commands: audio file -> text.

Audio is captured in the browser and POSTed to `/api/transcribe`. Two
backends: mlx-whisper (Apple GPU) and faster-whisper; ASR_BACKEND picks one,
otherwise mlx-whisper when installed. ASR_MODEL picks the size.

faster-whisper runs on an NVIDIA GPU (float16) when one is usable and on the
CPU (int8) otherwise; ASR_DEVICE=cpu or =cuda overrides that. Whisper small on
an RTX 5080 takes 40-90 ms per command against 480-590 ms on a Ryzen 9 9950X3D
(2026-11-26, same transcripts).
"""
import ctypes
import gc
import os
import re
import sys
import threading

import numpy as np

_MODEL_CACHE = {}
_MODEL_LOCK = threading.Lock()
SAMPLE_RATE = 16000
AUDIO_DIR = "out/audio"
ASR_BACKEND = os.environ.get("ASR_BACKEND")  # "mlx" | "faster-whisper" | None
ASR_MODEL = os.environ.get("ASR_MODEL", "small")
ASR_DEVICE = os.environ.get("ASR_DEVICE", "auto")  # "auto" | "cuda" | "cpu" (faster-whisper only)
MLX_REPOS = {"small": "mlx-community/whisper-small-mlx", "turbo": "mlx-community/whisper-large-v3-turbo"}
FASTER_WHISPER_NAMES = {"small": "small", "turbo": "large-v3-turbo"}

# CTranslate2 (faster-whisper's engine) is built against CUDA 12 and opens
# cuBLAS 12 and cuDNN 9 by name. torch's CUDA 13 wheels bring cuDNN 9 but only
# cuBLAS 13, so `make setup` adds the nvidia-cublas-cu12 wheel; loading both
# from the wheels here means no LD_LIBRARY_PATH. Not nvidia-cudnn-cu12: it
# writes the same nvidia/cudnn/lib files as torch's nvidia-cudnn-cu13.
_CUDA_LIBRARIES = (("nvidia.cublas", ("libcublasLt.so.12", "libcublas.so.12")),
                   ("nvidia.cudnn", ("libcudnn.so.9",)))

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


def _load_cuda_libraries() -> str:
    """Load the CUDA libraries CTranslate2 needs from the pip wheels; None when
    that worked, else why not (Linux only: macOS has no CUDA)."""
    if not sys.platform.startswith("linux"):
        return "no CUDA on this platform"
    import importlib
    for package, names in _CUDA_LIBRARIES:
        try:
            directory = os.path.join(list(importlib.import_module(package).__path__)[0], "lib")
        except ImportError:
            return f"{package} is not installed (make setup installs it on Linux with an NVIDIA GPU)"
        for name in names:
            try:
                ctypes.CDLL(os.path.join(directory, name), mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:
                return f"cannot load {name}: {exc}"
    return None


def _faster_whisper_model(name: str):
    """(WhisperModel, device, compute type): CUDA when usable (checked with one
    warm-up transcription, since a missing library only shows when the
    encoder first runs), else the CPU."""
    import ctranslate2
    from faster_whisper import WhisperModel
    if ASR_DEVICE != "cpu":
        problem = _load_cuda_libraries()
        if problem is None and ctranslate2.get_cuda_device_count() == 0:
            problem = "no CUDA device"
        if problem is None:
            try:
                model = WhisperModel(name, device="cuda", compute_type="float16")
                silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
                list(model.transcribe(silence, language="en", vad_filter=False)[0])
                return model, "cuda", "float16"
            except (RuntimeError, ValueError) as exc:
                problem = str(exc)
        if ASR_DEVICE == "cuda":
            raise RuntimeError(f"ASR_DEVICE=cuda, but CUDA is not usable: {problem}")
        if sys.platform.startswith("linux") and ctranslate2.get_cuda_device_count() > 0:
            print(f"[asr] GPU not usable, falling back to the CPU: {problem}")
    return WhisperModel(name, device="cpu", compute_type="int8"), "cpu", "int8"


def _load_model(model_size: str, backend: str):
    key = (backend, model_size)
    with _MODEL_LOCK:   # the server's start-up warm-up and a first request may race
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        if backend == "mlx":
            _MODEL_CACHE[key] = ("mlx", MLX_REPOS[model_size])
            print(f"[asr] using mlx-whisper ({MLX_REPOS[model_size]})")
        else:
            name = FASTER_WHISPER_NAMES[model_size]
            model, device, compute_type = _faster_whisper_model(name)
            _MODEL_CACHE[key] = ("faster-whisper", model)
            print(f"[asr] using faster-whisper ({name}, {device}, {compute_type})")
        return _MODEL_CACHE[key]


def warm(model_size: str = ASR_MODEL, backend: str = None) -> None:
    """Load the model before the first voice command needs it."""
    _load_model(model_size, backend or _default_backend())


def _decode(path: str) -> np.ndarray:
    """The clip as float32 mono at SAMPLE_RATE: the same samples as
    faster_whisper.audio.decode_audio, which ends with a full gc.collect() to
    free PyAV resampler cycles (faster-whisper issue 390). With torch and vtk
    loaded that full collection took ~60 ms per command; the cycles are new,
    so collecting the young generation frees them too (no growth over 300
    decodes, 12 ms per call; 2026-11-26)."""
    import av
    from av.audio.resampler import AudioResampler
    from av.error import InvalidDataError
    resampler = AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    chunks = []
    with av.open(path, mode="r") as container:
        frames = container.decode(audio=0)
        while True:
            try:
                frame = next(frames)
            except StopIteration:
                break
            except InvalidDataError:   # skipped, as faster-whisper does
                continue
            chunks += [out.to_ndarray().reshape(-1) for out in resampler.resample(frame)]
        chunks += [out.to_ndarray().reshape(-1) for out in resampler.resample(None)]
    del resampler
    gc.collect(0)
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _transcribe_path(path: str, lang: str, model_size: str = ASR_MODEL, initial_prompt: str = None,
                     backend: str = None) -> str:
    kind, model = _load_model(model_size, backend or _default_backend())
    if kind == "mlx":
        import mlx_whisper  # pyright: ignore[reportMissingImports]
        result = mlx_whisper.transcribe(path, path_or_hf_repo=model, language=lang,
                                        initial_prompt=initial_prompt, condition_on_previous_text=False)
        return result["text"].strip()
    segments, _ = model.transcribe(_decode(path), language=lang, initial_prompt=initial_prompt, vad_filter=True,
                                   condition_on_previous_text=False)
    return " ".join(seg.text.strip() for seg in segments).strip()
