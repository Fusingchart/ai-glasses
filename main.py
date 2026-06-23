"""Entry point: start mic capture, pipeline, and overlay together."""
import sys
import threading

import numpy as np
import sounddevice as sd
from PyQt6.QtWidgets import QApplication

from config import Config
from overlay import OverlayBridge, run_overlay
from pipeline import Pipeline

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512


def main() -> None:
    cfg = Config()
    pipeline = Pipeline(cfg)
    app = QApplication(sys.argv)
    bridge = OverlayBridge()
    pipeline.register_card_callback(bridge.push_card)

    def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        mono = indata[:, 0].copy()
        pipeline.feed(mono)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SAMPLES,
        callback=audio_callback,
    )

    with stream:
        print("[main] listening... speak to trigger cards. Ctrl+C to quit.")
        run_overlay(bridge)


if __name__ == "__main__":
    main()
