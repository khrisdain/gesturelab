"""Synthetic hands: a simple kinematic hand posed into each gesture, so the pipeline can be trained,
tested and demoed before anyone records real data.

Each finger is a chain of bones from its knuckle. A gesture is a set of curls (0 = straight,
1 = fully bent) per finger; samples jitter the curls, the bone lengths (a different "person" per
session), the spread, the orientation, the size and the position in the frame, then add
landmark noise like a real tracker's. The output uses MediaPipe's coordinate conventions.

Synthetic hands are much tidier than real ones. They are for tests and for a starter model;
record your own gestures for a model that works well on your hands and camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hand import Hand

# Curl per finger: thumb, index, middle, ring, pinky.
GESTURES: dict[str, tuple[float, float, float, float, float]] = {
    "open_palm": (0.0, 0.0, 0.0, 0.0, 0.0),
    "fist": (0.9, 1.0, 1.0, 1.0, 1.0),
    "thumbs_up": (0.0, 1.0, 1.0, 1.0, 1.0),
    "point": (0.9, 0.0, 1.0, 1.0, 1.0),
    "peace": (0.9, 0.0, 0.0, 1.0, 1.0),
    "rock": (0.9, 0.0, 1.0, 1.0, 0.0),
    "call_me": (0.0, 1.0, 1.0, 1.0, 0.0),
    "ok": (0.6, 0.6, 0.0, 0.0, 0.0),
}

# Hand space: wrist at the origin, palm facing the camera, fingers along +y, thumb towards -x.
KNUCKLES = np.array([[-0.28, 0.18, 0.0], [-0.26, 0.92, 0.0], [-0.05, 0.97, 0.0], [0.15, 0.92, 0.0], [0.33, 0.82, 0.0]])
BONES = np.array([[0.36, 0.30, 0.25], [0.45, 0.27, 0.22], [0.50, 0.30, 0.24], [0.46, 0.28, 0.22], [0.36, 0.22, 0.20]])
MAX_BEND = np.array([[0.7, 0.9, 1.0], [1.5, 1.7, 1.2], [1.5, 1.7, 1.2], [1.5, 1.7, 1.2], [1.5, 1.7, 1.2]])  # radians
SPLAY = np.array([0.95, 0.12, 0.0, -0.1, -0.22])  # each finger's resting angle away from straight up


@dataclass
class Person:
    """Per-session variation: hand proportions and how widely this person spreads their fingers."""

    bones: np.ndarray
    spread: float

    @classmethod
    def random(cls, rng: np.random.Generator) -> Person:
        return cls(bones=BONES * rng.normal(1.0, 0.06, BONES.shape), spread=float(rng.uniform(0.7, 1.3)))


def pose(curls: np.ndarray, person: Person, rng: np.random.Generator) -> np.ndarray:
    """(21, 3) landmarks in hand space for the given finger curls."""
    points = np.zeros((21, 3))
    for finger in range(5):
        base = 1 + finger * 4  # the thumb's CMC, or another finger's MCP knuckle
        position = KNUCKLES[finger].copy()
        points[base] = position
        splay = SPLAY[finger] * (1.0 if finger == 0 else person.spread) + rng.normal(0, 0.04)
        bend = 0.0
        for bone in range(3):
            bend += curls[finger] * MAX_BEND[finger, bone]
            if finger == 0:
                # The thumb folds across the palm (towards +x) and slightly towards the camera.
                direction = np.array([-np.sin(splay - bend), np.cos(splay - bend), -0.3 * np.sin(bend)])
            else:
                # Other fingers fold towards the palm: down in y and towards the camera in z.
                direction = np.array([-np.sin(splay) * np.cos(bend), np.cos(splay) * np.cos(bend), -np.sin(bend)])
            position = position + person.bones[finger, bone] * direction / np.linalg.norm(direction)
            points[base + 1 + bone] = position
    return points


def sample(gesture: str, rng: np.random.Generator, person: Person | None = None) -> Hand:
    person = person or Person.random(rng)
    curls = np.clip(np.array(GESTURES[gesture]) + rng.normal(0, 0.08, 5), 0, 1)
    points = pose(curls, person, rng)
    # Orientation: tilt out of the image plane, then roll within it.
    yaw, pitch, roll = rng.normal(0, 0.3), rng.normal(0, 0.2), rng.uniform(-0.6, 0.6)
    for axes, angle in (((0, 2), yaw), ((1, 2), pitch), ((0, 1), roll)):
        a, b = axes
        cos, sin = np.cos(angle), np.sin(angle)
        pa, pb = points[:, a].copy(), points[:, b].copy()
        points[:, a], points[:, b] = cos * pa - sin * pb, sin * pa + cos * pb
    # Into image coordinates: y grows downwards, a hand 10-30% of the frame, anywhere in it.
    points[:, 1] *= -1
    points *= rng.uniform(0.1, 0.3)
    points[:, :2] += rng.uniform(0.25, 0.75, 2)
    points += rng.normal(0, 0.003, points.shape)
    if rng.random() < 0.5:  # a left hand is the mirror image of a right one
        points[:, 0] = 1 - points[:, 0]
        return Hand(points=points, handedness="Left")
    return Hand(points=points, handedness="Right")


def dataset(per_class: int = 300, sessions: int = 6, seed: int = 0) -> tuple[list[Hand], list[str], list[str]]:
    """Hands, labels and session ids: `sessions` simulated people, each recording every gesture."""
    rng = np.random.default_rng(seed)
    hands, labels, groups = [], [], []
    for session in range(sessions):
        person = Person.random(rng)
        for gesture in GESTURES:
            for _ in range(per_class // sessions):
                hands.append(sample(gesture, rng, person))
                labels.append(gesture)
                groups.append(f"synth-{session}")
    return hands, labels, groups
