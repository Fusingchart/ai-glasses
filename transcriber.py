"""Whisper transcription wrapper using faster-whisper."""
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel


@dataclass
class Transcript:
    text: str
    language: str
    duration_s: float


class Transcriber:
    def __init__(self, model_size: str = "base.en", device: str = "cpu"):
        self._model = WhisperModel(model_size, device=device, compute_type="int8")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        segments, info = self._model.transcribe(
            audio,
            beam_size=5,
            language="en",
            vad_filter=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return Transcript(
            text=text,
            language=info.language,
            duration_s=info.duration,
        )
