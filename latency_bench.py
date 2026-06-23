"""Measure end-to-end pipeline latency on sample audio snippets."""
import json
import time
from pathlib import Path

import numpy as np

from config import Config
from pipeline import Pipeline

SAMPLE_RATE = 16000
SAMPLE_DURATION_S = 5


def _make_silence(duration_s: float = 0.5) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * duration_s), dtype=np.float32)


def _load_or_generate_audio(path: Path | None) -> np.ndarray:
    if path and path.exists():
        import torchaudio
        waveform, sr = torchaudio.load(str(path))
        if sr != SAMPLE_RATE:
            import torchaudio.functional as F
            waveform = F.resample(waveform, sr, SAMPLE_RATE)
        return waveform.mean(0).numpy()
    # Fallback: white noise (triggers Whisper but produces garbage text)
    return np.random.randn(SAMPLE_RATE * SAMPLE_DURATION_S).astype(np.float32) * 0.01


def run_bench(audio_paths: list[Path | None], runs: int = 3) -> dict:
    cfg = Config()
    pipeline = Pipeline(cfg)
    results = []

    for audio_path in audio_paths:
        audio = _load_or_generate_audio(audio_path)
        latencies = []
        for _ in range(runs):
            cards: list = []
            pipeline.register_card_callback(lambda c: cards.append(c))
            t0 = time.monotonic()
            # Simulate VAD: feed audio in chunks, then silence to flush
            chunk_size = VADDetector.CHUNK_SAMPLES if hasattr(__builtins__, "VADDetector") else 512
            for i in range(0, len(audio), chunk_size):
                pipeline._vad.process_chunk(audio[i : i + chunk_size])
            # Force flush with silence
            silence = _make_silence()
            for i in range(0, len(silence), 512):
                pipeline._vad.process_chunk(silence[i : i + 512])
            elapsed = time.monotonic() - t0
            latencies.append(elapsed)

        results.append({
            "file": str(audio_path) if audio_path else "synthetic",
            "mean_s": round(sum(latencies) / len(latencies), 3),
            "min_s": round(min(latencies), 3),
            "max_s": round(max(latencies), 3),
            "target_s": cfg.latency_target_s,
            "pass": max(latencies) <= cfg.latency_target_s,
        })

    print(json.dumps(results, indent=2))
    return {"results": results}


if __name__ == "__main__":
    import sys
    paths = [Path(p) for p in sys.argv[1:]] or [None]
    run_bench(paths)
