"""Turning 21 landmarks into a feature vector that describes the hand's shape, not where it is.

The same gesture should look the same to the classifier wherever the hand is in the frame, how
big it appears, how it is tilted and which hand makes it. So the landmarks are:

1. translated so the wrist is at the origin,
2. mirrored if it is a left hand, so both hands share one model,
3. rotated in the image plane so the palm (wrist to middle-finger knuckle) points straight up,
4. scaled so that palm is length 1.

Then three groups of features are taken from the normalised hand: the coordinates themselves,
how bent each joint is, and the distances between fingertips (pinches) and to the wrist (reach).
"""

from __future__ import annotations

import numpy as np

from .hand import FINGERS, INDEX, MIDDLE, TIPS, WRIST, Hand

FEATURE_VERSION = 1


def normalize(hand: Hand) -> np.ndarray:
    points = hand.points.astype(np.float64) - hand.points[WRIST]
    if hand.handedness == "Left":
        points[:, 0] *= -1
    palm = points[MIDDLE[0]]
    # Rotate so the palm points to (0, -1): straight up in image coordinates.
    angle = -np.arctan2(palm[0], -palm[1])
    cos, sin = np.cos(angle), np.sin(angle)
    x, y = points[:, 0].copy(), points[:, 1].copy()
    points[:, 0], points[:, 1] = cos * x - sin * y, sin * x + cos * y
    size = np.linalg.norm(points[MIDDLE[0]])
    return points / size if size > 1e-9 else points


def _bend(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Cosine of the angle at b: 1 for a straight joint, falling as it bends."""
    u, v = a - b, c - b
    denominator = np.linalg.norm(u) * np.linalg.norm(v)
    return float(-np.dot(u, v) / denominator) if denominator > 1e-9 else 1.0


def features(hand: Hand) -> np.ndarray:
    points = normalize(hand)
    bends: list[float] = []
    for finger in FINGERS:
        chain = (WRIST, *finger)
        bends.extend(_bend(points[chain[i]], points[chain[i + 1]], points[chain[i + 2]]) for i in range(3))
    thumb_tip = points[TIPS[0]]
    pinches = [np.linalg.norm(thumb_tip - points[tip]) for tip in TIPS[1:]]
    reach = [np.linalg.norm(points[tip]) for tip in TIPS]
    spread = [np.linalg.norm(points[a] - points[b]) for a, b in zip(TIPS[1:], TIPS[2:], strict=False)]
    return np.concatenate([points.ravel(), bends, pinches, reach, spread, [points[INDEX[-1], 2]]])


def feature_names() -> list[str]:
    coords = [f"p{i}_{axis}" for i in range(21) for axis in "xyz"]
    names = ["thumb", "index", "middle", "ring", "pinky"]
    bends = [f"bend_{finger}_{joint}" for finger in names for joint in (1, 2, 3)]
    pinches = [f"pinch_thumb_{finger}" for finger in names[1:]]
    reach = [f"reach_{finger}" for finger in names]
    spread = [f"spread_{a}_{b}" for a, b in zip(names[1:], names[2:], strict=False)]
    return [*coords, *bends, *pinches, *reach, *spread, "index_tip_depth"]


def feature_matrix(hands: list[Hand]) -> np.ndarray:
    return np.stack([features(hand) for hand in hands]) if hands else np.empty((0, len(feature_names())))
