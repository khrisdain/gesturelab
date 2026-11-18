"""Digital zoom: crop the middle of the frame and scale it back up.

Laptop webcams are wide-angle, so a hand at arm's length fills only a small part of the picture. Cropping
before hand tracking also helps the tracker, which then sees the hand with more pixels across it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MIN, MAX, STEP = 1.0, 3.0, 1.1


def crop(frame: Any, factor: float) -> Any:
    """The centred part of `frame` that is 1/factor of its width and height."""
    if factor <= 1.0:
        return frame
    height, width = frame.shape[:2]
    h, w = max(1, round(height / factor)), max(1, round(width / factor))
    top, left = (height - h) // 2, (width - w) // 2
    return frame[top:top + h, left:left + w]


@dataclass
class Zoom:
    factor: float = 1.5

    def __post_init__(self) -> None:
        self.factor = min(MAX, max(MIN, self.factor))

    def apply(self, frame: Any) -> Any:
        """Zoomed frame at the original size, so windows and overlays don't change shape."""
        if self.factor <= 1.0:
            return frame
        import cv2

        height, width = frame.shape[:2]
        return cv2.resize(crop(frame, self.factor), (width, height), interpolation=cv2.INTER_LINEAR)

    def key(self, key: int) -> bool:
        """Handle + / - (and =, the unshifted +); True if the key was a zoom key."""
        if key in (ord("+"), ord("=")):
            self.factor = min(MAX, round(self.factor * STEP, 3))
        elif key in (ord("-"), ord("_")):
            self.factor = max(MIN, round(self.factor / STEP, 3))
        else:
            return False
        return True
