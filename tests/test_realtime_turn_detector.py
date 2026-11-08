from __future__ import annotations

import numpy as np

from runtime.realtime.turn_detector import TurnDetector


def _loud(samples: int, amplitude: int = 3000) -> bytes:
    return np.full(samples, amplitude, dtype="<i2").tobytes()


def _quiet(samples: int) -> bytes:
    return np.zeros(samples, dtype="<i2").tobytes()


def test_loud_audio_starts_speech():
    detector = TurnDetector()
    assert detector.feed(_loud(2400)) == ["start"]
    assert detector.speaking is True


def test_silence_after_speech_stops_the_turn():
    detector = TurnDetector()
    assert detector.feed(_loud(4800)) == ["start"]
    events = detector.feed(_quiet(24_000))
    assert "stop" in events
    assert detector.speaking is False


def test_quiet_shorter_than_silence_ms_does_not_stop():
    detector = TurnDetector()
    assert detector.feed(_loud(4800)) == ["start"]
    assert detector.feed(_quiet(9_600)) == []
    assert detector.speaking is True


def test_whole_turn_in_one_buffer_returns_both_events_in_order():
    detector = TurnDetector()
    events = detector.feed(_loud(4800) + _quiet(24_000))
    assert events == ["start", "stop"]


def test_short_cough_is_not_a_turn():
    detector = TurnDetector()
    assert detector.feed(_loud(1_200)) == ["start"]
    assert detector.feed(_quiet(24_000)) == []
    assert detector.speaking is False


def test_only_silence_produces_no_events():
    detector = TurnDetector()
    assert detector.feed(_quiet(48_000)) == []
    assert detector.speaking is False


def test_empty_buffer_produces_no_events():
    detector = TurnDetector()
    assert detector.feed(b"") == []


def test_events_continue_across_turns():
    detector = TurnDetector()
    assert detector.feed(_loud(4_800) + _quiet(24_000)) == ["start", "stop"]
    assert detector.feed(_loud(4_800) + _quiet(24_000)) == ["start", "stop"]


class TestQuietSpeech:
    """A microphone at a low gain must still produce turns."""

    def test_quiet_speech_starts_and_stops_a_turn(self):
        detector = TurnDetector(silence_ms=100, min_speech_ms=20)
        amplitude = int(0.0067 * 32767)
        quiet = np.full(24_000, amplitude, dtype=np.int16).tobytes()
        events = detector.feed(quiet)
        assert events == ["start"], f"quiet speech must open a turn, got {events}"
        silence = np.zeros(24_000, dtype=np.int16).tobytes()
        assert detector.feed(silence) == ["stop"], "quiet speech must close its turn"

    def test_room_tone_opens_no_turn(self):
        detector = TurnDetector(silence_ms=100, min_speech_ms=20)
        tone = (np.random.default_rng(3).normal(0, 0.002, 24_000) * 32767).astype(np.int16).tobytes()
        assert detector.feed(tone) == [], "steady room tone is not speech"
