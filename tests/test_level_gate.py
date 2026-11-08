from __future__ import annotations

from providers.level import LevelGate


class TestLevelGate:
    """A gate that admits speech at any level and skips silence."""

    def test_digital_silence_is_skipped(self):
        gate = LevelGate(absolute_min=0.003, ratio=0.25, decay=0.9)
        gate.update(0.0)
        assert not gate.admits(0.0)

    def test_quiet_speech_is_admitted(self):
        # A microphone at a low gain: the level is a tenth of what a fixed cutoff expects.
        gate = LevelGate(absolute_min=0.003, ratio=0.25, decay=0.9)
        gate.update(0.0067)
        assert gate.admits(0.0067), "quiet speech must not be dropped"

    def test_room_tone_below_the_floor_is_skipped(self):
        gate = LevelGate(absolute_min=0.003, ratio=0.25, decay=0.9)
        gate.update(0.002)
        assert not gate.admits(0.002)

    def test_the_threshold_never_drops_below_the_floor(self):
        gate = LevelGate(absolute_min=0.003, ratio=0.25, decay=0.9)
        gate.update(0.0001)
        assert gate.threshold() >= 0.003

    def test_a_quiet_level_after_loud_audio_is_still_admitted(self):
        # The recent maximum decays, so a level that was speech a moment ago is not
        # locked out by the loud passage before it.
        gate = LevelGate(absolute_min=0.003, ratio=0.25, decay=0.5)
        gate.update(0.2)
        for _ in range(20):
            gate.update(0.0067)
        assert gate.admits(0.0067)

    def test_a_level_a_quarter_of_the_recent_maximum_is_admitted(self):
        gate = LevelGate(absolute_min=0.003, ratio=0.25, decay=0.9)
        gate.update(0.08)
        assert gate.admits(0.02)

    def test_a_level_below_a_quarter_of_the_recent_maximum_is_skipped(self):
        gate = LevelGate(absolute_min=0.003, ratio=0.25, decay=0.9)
        gate.update(0.08)
        assert not gate.admits(0.01)
