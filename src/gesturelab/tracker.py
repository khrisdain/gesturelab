"""Finding hands in camera frames with MediaPipe's hand landmarker.

The landmarker needs Google's published model file (about 7.5 MB). It is downloaded once, on
first use, to ~/.cache/gesturelab, or you can point GESTURELAB_HAND_MODEL at a copy.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

from .hand import Hand

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def model_path() -> Path:
    configured = os.environ.get("GESTURELAB_HAND_MODEL")
    if configured:
        return Path(configured)
    path = Path.home() / ".cache" / "gesturelab" / "hand_landmarker.task"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe's hand landmarker model to {path} (once) …")
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, partial)  # noqa: S310 - fixed https URL
        partial.rename(path)
    return path


class HandTracker:
    """Hand landmarks for a stream of video frames (BGR, as OpenCV delivers them)."""

    def __init__(self, max_hands: int = 1, min_confidence: float = 0.6) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path())),
            running_mode=vision.RunningMode.VIDEO, num_hands=max_hands,
            min_hand_detection_confidence=min_confidence, min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ms = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[Hand]:
        """Hands in the frame. Pass the frame already mirrored (selfie view): MediaPipe's left/right
        labels assume a mirrored image."""
        import cv2

        timestamp_ms = max(timestamp_ms, self._last_ms + 1)  # video mode needs strictly increasing timestamps
        self._last_ms = timestamp_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        hands = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness, strict=True):
            points = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=float)
            category = handedness[0]
            hands.append(Hand(points=points, handedness=category.category_name, score=float(category.score)))
        return hands

    def close(self) -> None:
        self._landmarker.close()
