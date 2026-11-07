"""Steadying frame-by-frame predictions so the live label doesn't flicker.

Class probabilities are averaged over recent frames (an exponential moving average). A gesture is
reported only once it has led with enough confidence for a few frames in a row, and it is dropped
when the hand disappears or nothing is confident, so a hand moving between poses reads as "none"
rather than a string of wrong guesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Smoother:
    labels: list[str]
    alpha: float = 0.35
    threshold: float = 0.7
    hold: int = 3
    _average: np.ndarray | None = field(default=None, init=False)
    _candidate: str | None = field(default=None, init=False)
    _streak: int = field(default=0, init=False)
    current: str | None = field(default=None, init=False)
    confidence: float = field(default=0.0, init=False)

    def update(self, probabilities: np.ndarray | None) -> str | None:
        """Feed one frame's probabilities (None when no hand); returns the steady gesture or None."""
        if probabilities is None:
            self.reset()
            return None
        previous = self._average
        self._average = probabilities if previous is None else self.alpha * probabilities + (1 - self.alpha) * previous
        best = int(np.argmax(self._average))
        self.confidence = float(self._average[best])
        leader = self.labels[best] if self.confidence >= self.threshold else None
        self._streak = self._streak + 1 if leader == self._candidate else 1
        self._candidate = leader
        if self._streak >= self.hold or leader is None:
            self.current = leader
        return self.current

    def reset(self) -> None:
        self._average, self._candidate, self._streak, self.current, self.confidence = None, None, 0, None, 0.0
