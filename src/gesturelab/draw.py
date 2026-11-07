"""On-screen overlays: the hand skeleton, the recognised gesture and a probability bar per class."""

from __future__ import annotations

import cv2
import numpy as np

from .hand import CONNECTIONS, TIPS, Hand

GREEN, WHITE, GREY, DARK = (80, 200, 120), (255, 255, 255), (150, 150, 150), (30, 30, 30)


def skeleton(frame: np.ndarray, hand: Hand, colour: tuple[int, int, int] = GREEN) -> None:
    h, w = frame.shape[:2]
    pixels = [(int(x * w), int(y * h)) for x, y, _ in hand.points]
    for a, b in CONNECTIONS:
        cv2.line(frame, pixels[a], pixels[b], colour, 2, cv2.LINE_AA)
    for i, point in enumerate(pixels):
        cv2.circle(frame, point, 6 if i in TIPS else 4, WHITE if i in TIPS else colour, -1, cv2.LINE_AA)


def banner(frame: np.ndarray, text: str, sub: str = "") -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 64 if sub else 44), DARK, -1)
    cv2.putText(frame, text, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
    if sub:
        cv2.putText(frame, sub, (14, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1, cv2.LINE_AA)


def probabilities(frame: np.ndarray, labels: list[str], values: np.ndarray, current: str | None) -> None:
    top = frame.shape[0] - 22 * len(labels) - 12
    for i, (label, value) in enumerate(zip(labels, values, strict=True)):
        y = top + i * 22
        cv2.rectangle(frame, (12, y), (12 + int(140 * value), y + 16), GREEN if label == current else GREY, -1)
        cv2.putText(frame, f"{label} {value:.0%}", (160, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
