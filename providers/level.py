"""Deciding what counts as speech, whatever the input level is.

An absolute threshold cannot work across microphones. The same room gives 0.0001 from one
device and 0.09 from another — a hundredfold difference — so any fixed cutoff drops one of
them entirely, and a dropped chunk is a lost sentence with nothing to show for it.

The gate therefore sits at a fraction of the loudest recent level, with a floor under it so
that digital silence still costs nothing. The recent maximum decays, so a quiet passage
after a loud one is not locked out by the loud passage.
"""

from __future__ import annotations


class LevelGate:
    """Admits speech at any level and skips silence."""

    def __init__(
        self,
        *,
        absolute_min: float,
        ratio: float = 0.25,
        decay: float = 0.9,
    ) -> None:
        self.absolute_min = absolute_min
        self.ratio = ratio
        self.decay = decay
        self._recent_max = 0.0

    def update(self, level: float) -> None:
        """Take a level into account and move the recent maximum with it."""
        level = max(0.0, float(level))
        self._recent_max = max(level, self._recent_max * self.decay)

    def threshold(self) -> float:
        """The level below which audio is treated as silence."""
        return max(self.absolute_min, self._recent_max * self.ratio)

    def admits(self, level: float) -> bool:
        """Whether a level is worth handing to the model."""
        return float(level) >= self.threshold()
