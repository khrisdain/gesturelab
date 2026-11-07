"""Recorded samples on disk: one CSV row per frame, raw landmarks plus who/when it was recorded.

Landmarks are stored rather than images: it is 63 numbers a frame, it keeps no picture of the
person recording, and features can be recomputed if the feature code changes.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hand import Hand

COLUMNS = ["label", "session", "handedness", *[f"{axis}{i}" for i in range(21) for axis in "xyz"]]


@dataclass
class Samples:
    hands: list[Hand]
    labels: list[str]
    sessions: list[str]

    def __len__(self) -> int:
        return len(self.hands)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for label in self.labels:
            out[label] = out.get(label, 0) + 1
        return dict(sorted(out.items()))


def append(path: Path, label: str, session: str, hands: list[Hand]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(COLUMNS)
        for hand in hands:
            writer.writerow([label, session, hand.handedness, *(f"{v:.5f}" for v in hand.points.ravel())])


def load(path: Path) -> Samples:
    hands, labels, sessions = [], [], []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != COLUMNS:
            raise ValueError(f"{path} is not a gesturelab samples file")
        for row in reader:
            hands.append(Hand(points=np.array(row[3:], dtype=float).reshape(21, 3), handedness=row[2]))  # type: ignore[arg-type]
            labels.append(row[0])
            sessions.append(row[1])
    return Samples(hands, labels, sessions)
