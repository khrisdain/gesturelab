"""Which camera is which: probe the camera indices OpenCV can open and put names to them.

OpenCV only knows cameras by number, and on a Mac number 0 is often not the built-in webcam: a virtual
camera app, an iPhone (Continuity Camera) or an external webcam can come first. macOS reports camera
names in the same order OpenCV numbers them, and a short burst of frames tells a real sensor (which
always has some noise) from a virtual camera showing a still placeholder.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# A real sensor's frames differ by a few grey levels even when nothing moves; a still image does not.
LIVE_THRESHOLD = 0.3


@dataclass
class Camera:
    index: int
    name: str | None
    width: int = 0
    height: int = 0
    live: bool | None = None  # None when the camera opened but gave no frames

    @property
    def opened(self) -> bool:
        return self.width > 0


def macos_names(run: Callable[..., Any] = subprocess.run) -> list[str]:
    """Camera names in the order macOS lists them, or [] when that isn't available."""
    if sys.platform != "darwin":
        return []
    try:
        out = run(["system_profiler", "SPCameraDataType", "-json"], capture_output=True, text=True, timeout=20)
        return [entry.get("_name", "?") for entry in json.loads(out.stdout).get("SPCameraDataType", [])]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def frame_change(frames: list[Any]) -> float:
    """Mean absolute grey-level difference between consecutive frames."""
    import numpy as np

    grey = [np.asarray(f, dtype=np.float32).mean(axis=-1) if np.ndim(f) == 3 else np.asarray(f, np.float32)
            for f in frames]
    if len(grey) < 2:
        return 0.0
    return float(np.mean([np.abs(a - b).mean() for a, b in zip(grey, grey[1:], strict=False)]))


def probe(index: int, name: str | None, opener: Callable[[int], Any], frames: int = 8) -> Camera:
    camera = Camera(index, name)
    capture = opener(index)
    try:
        if not capture.isOpened():
            return camera
        shots = []
        for _ in range(frames + 4):  # the first frames are often black while exposure settles
            ok, frame = capture.read()
            if ok and frame is not None:
                shots.append(frame)
        if not shots:
            camera.width = camera.height = 1  # opened, but nothing came through
            return camera
        camera.height, camera.width = shots[-1].shape[:2]
        camera.live = frame_change(shots[-frames:]) > LIVE_THRESHOLD
        return camera
    finally:
        capture.release()


def find(opener: Callable[[int], Any] | None = None, names: list[str] | None = None,
         extra: int = 2) -> list[Camera]:
    """Probe every index macOS names, plus a couple more in case it missed one."""
    if opener is None:
        os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # quiet the warnings for indices that don't exist
        import cv2

        opener = cv2.VideoCapture
    names = macos_names() if names is None else names
    count = max(len(names), 1) + extra
    return [probe(i, names[i] if i < len(names) else None, opener) for i in range(count)]


def table(cameras: list[Camera]) -> str:
    rows = []
    for c in cameras:
        if not c.opened:
            if c.name:
                rows.append(f"  {c.index}  {c.name:<28} could not open (camera access off, or in use?)")
            continue
        size = f"{c.width}x{c.height}" if c.width > 1 else "no frames"
        state = {True: "live", False: "still image (a virtual camera that isn't running?)", None: ""}[c.live]
        rows.append(f"  {c.index}  {(c.name or 'unnamed'):<28} {size:<11} {state}".rstrip())
    if not rows:
        return ("No cameras found. On macOS, allow camera access for your terminal in "
                "System Settings > Privacy & Security > Camera.")
    live = [c for c in cameras if c.live]
    hint = f"\nUse one with: gesturelab live --camera {live[0].index}" if live else ""
    return "  #  name                         size        state\n" + "\n".join(rows) + hint
