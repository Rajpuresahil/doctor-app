import io
import os
import wave
import asr
import numpy as np

BACKEND = os.getenv("ASR_BACKEND", "faster-whisper")
MODEL_NAME = os.getenv("WHISPER_MODEL", "base.en")
SR = 16000

# Biases the decoder toward clinical vocabulary (helps drug names a lot).
MED_PRIMER = ("Medical consultation. Terms: paracetamol, metformin, azithromycin, amoxicillin, "
              "hypertension, diabetes mellitus, CBC, HbA1c, ECG, mg, blood pressure.")

# Whisper tends to hallucinate these on near-silence; drop them on very short clips.
HALLUCINATIONS = {"thank you", "thanks for watching", "you", "bye", "thank you."}

_model = None


def _load():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
    return _model


def _wav_bytes(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    return buf.getvalue()


def transcribe(pcm: bytes, context: str = "") -> str:
    """pcm: raw 16 kHz mono int16 little-endian. context: tail of transcript so far."""
    if len(pcm) < SR * 2 * 0.25:  # <250 ms
        return ""
    seconds = len(pcm) / 2 / SR
    prompt = (MED_PRIMER + " " + context).strip()

    if BACKEND == "openai":
        from openai import OpenAI
        r = OpenAI().audio.transcriptions.create(model="whisper-1", file=("seg.wav", _wav_bytes(pcm)),
                                                 prompt=prompt[-500:], language="en")
        text = r.text.strip()
    else:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segs, _ = _load().transcribe(audio, language="en", beam_size=1, vad_filter=False,
                                     condition_on_previous_text=False, initial_prompt=prompt[-500:])
        text = " ".join(s.text.strip() for s in segs).strip()

    if seconds < 1.5 and text.lower().strip(" .!") in {h.strip(" .!") for h in HALLUCINATIONS}:
        return ""
    return text
