"""faster-whisper wrapper: push-to-talk mic capture and file transcription."""
import datetime
import os
import time
import wave

import numpy as np
import sounddevice as sd

_MODEL_CACHE = {}
SAMPLE_RATE = 16000
AUDIO_DIR = "out/audio"


def _load_model(model_size: str = "small"):
    if model_size in _MODEL_CACHE:
        return _MODEL_CACHE[model_size]
    try:
        import mlx_whisper  # noqa: F401
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
        import mlx_whisper
        result = mlx_whisper.transcribe(path, language=lang)
        return result["text"].strip()
    segments, _ = model.transcribe(path, language=lang)
    return " ".join(seg.text.strip() for seg in segments).strip()


def _save_wav(path: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    audio_i16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_i16.tobytes())


def transcribe_mic(lang: str = "en", model_size: str = "small") -> str:
    input("Enter druecken zum Start der Aufnahme...")
    print("[asr] recording... Enter zum Stoppen.")
    frames = []
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                             callback=lambda indata, *_: frames.append(indata.copy()))
    with stream:
        input()
    audio = np.concatenate(frames, axis=0).flatten() if frames else np.zeros(0, dtype=np.float32)

    os.makedirs(AUDIO_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    wav_path = os.path.join(AUDIO_DIR, f"{stamp}.wav")
    txt_path = os.path.join(AUDIO_DIR, f"{stamp}.txt")
    _save_wav(wav_path, audio)

    start = time.time()
    text = _transcribe_path(wav_path, lang, model_size)
    duration = time.time() - start
    with open(txt_path, "w") as f:
        f.write(text)
    print(f"[asr] '{text}' ({duration:.2f}s)")
    return text


def transcribe_file(path: str, lang: str = "en", model_size: str = "small") -> str:
    start = time.time()
    text = _transcribe_path(path, lang, model_size)
    duration = time.time() - start
    txt_path = os.path.splitext(path)[0] + ".txt"
    with open(txt_path, "w") as f:
        f.write(text)
    print(f"[asr] '{text}' ({duration:.2f}s)")
    return text
