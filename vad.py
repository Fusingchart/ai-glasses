"""Voice Activity Detection using Silero VAD."""
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class VADResult:
    speech_detected: bool
    audio_chunk: np.ndarray
    timestamp: float


class VADDetector:
    SAMPLE_RATE = 16000
    CHUNK_SAMPLES = 512

    def __init__(self, threshold: float = 0.5, silence_duration_ms: int = 700):
        self.threshold = threshold
        self.silence_duration_ms = silence_duration_ms
        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        self._get_speech_timestamps = self._utils[0]
        self._buffer: deque[np.ndarray] = deque()
        self._in_speech = False
        self._silence_start: float | None = None

    def process_chunk(self, audio: np.ndarray) -> VADResult | None:
        tensor = torch.from_numpy(audio).float()
        prob = self._model(tensor, self.SAMPLE_RATE).item()
        now = time.monotonic()

        if prob >= self.threshold:
            self._in_speech = True
            self._silence_start = None
            self._buffer.append(audio)
            return None

        if self._in_speech:
            self._buffer.append(audio)
            if self._silence_start is None:
                self._silence_start = now
            elapsed_ms = (now - self._silence_start) * 1000
            if elapsed_ms >= self.silence_duration_ms:
                chunk = np.concatenate(list(self._buffer))
                self._buffer.clear()
                self._in_speech = False
                self._silence_start = None
                return VADResult(speech_detected=True, audio_chunk=chunk, timestamp=now)

        return None

    def reset(self) -> None:
        self._buffer.clear()
        self._in_speech = False
        self._silence_start = None
