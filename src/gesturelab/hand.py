"""The 21 hand landmarks MediaPipe reports, and a detected hand."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

WRIST = 0
THUMB = (1, 2, 3, 4)  # CMC, MCP, IP, tip
INDEX = (5, 6, 7, 8)  # MCP, PIP, DIP, tip
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
FINGERS = (THUMB, INDEX, MIDDLE, RING, PINKY)
TIPS = tuple(finger[-1] for finger in FINGERS)

# Bones to draw: each finger from the wrist out, plus the knuckle line across the palm.
CONNECTIONS = tuple(
    pair
    for finger in FINGERS
    for pair in zip((WRIST, *finger[:-1]), finger, strict=True)
) + ((5, 9), (9, 13), (13, 17))

Handedness = Literal["Left", "Right"]


@dataclass(frozen=True)
class Hand:
    """One detected hand: (21, 3) landmarks in MediaPipe's image coordinates.

    x and y are fractions of the image width and height (y grows downwards); z is depth relative
    to the wrist, on roughly the same scale as x.
    """

    points: np.ndarray
    handedness: Handedness = "Right"
    score: float = 1.0

    def __post_init__(self) -> None:
        if self.points.shape != (21, 3):
            raise ValueError(f"expected (21, 3) landmarks, got {self.points.shape}")
