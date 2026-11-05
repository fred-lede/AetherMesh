from __future__ import annotations

import numpy as np

SAMPLE_RATE = 24_000


class TurnDetector:
    """Energy VAD over 24 kHz PCM16: reports where a spoken turn starts and ends.

    A turn starts when a 20 ms window's RMS crosses ``threshold`` and ends
    after ``silence_ms`` of quiet. The quiet is counted in samples, so the
    decision belongs to the audio, not to how the client cut its buffers.
    A turn shorter than ``min_speech_ms`` is a cough, not a sentence.
    """

    def __init__(
        self,
        threshold: float = 0.02,
        silence_ms: float = 700,
        min_speech_ms: float = 100,
        preroll_ms: float = 300,
    ) -> None:
        self.threshold = threshold
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.preroll_ms = preroll_ms
        self.speaking = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._preroll: list[int] = []
        self._window = int(SAMPLE_RATE * 0.02)

    def feed(self, pcm: bytes) -> list[str]:
        if len(pcm) % 2:
            raise ValueError("PCM16 audio must contain complete samples")
        if not pcm:
            return []
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        events: list[str] = []
        for start in range(0, samples.size, self._window):
            piece = samples[start : start + self._window]
            if piece.size == 0:
                break
            loud = float(np.sqrt(np.mean(np.square(piece))))
            event = self._step(piece.size, loud)
            if event:
                events.append(event)
        return events

    def end_turn(self) -> None:
        self.speaking = False
        self._speech_samples = 0
        self._silence_samples = 0

    def _step(self, count: int, loud: float) -> str | None:
        if not self.speaking:
            if loud >= self.threshold:
                self.speaking = True
                self._speech_samples = count
                self._silence_samples = 0
                return "start"
            keep = int(self.preroll_ms * SAMPLE_RATE / 1000)
            self._preroll.append(count)
            if sum(self._preroll) > keep:
                self._preroll.pop(0)
            return None
        if loud >= self.threshold:
            self._speech_samples += count
            self._silence_samples = 0
            return None
        self._silence_samples += count
        if self._silence_samples * 1000 / SAMPLE_RATE < self.silence_ms:
            return None
        spoken_ms = self._speech_samples * 1000 / SAMPLE_RATE
        self.end_turn()
        if spoken_ms < self.min_speech_ms:
            return None
        return "stop"
