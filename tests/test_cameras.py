import json
from types import SimpleNamespace

import numpy as np

from gesturelab import cameras
from gesturelab.cameras import Camera, find, macos_names, table


class FakeCapture:
    """A camera that is missing, shows a still image, or returns sensor-noisy frames."""

    def __init__(self, kind: str, seed: int = 0) -> None:
        self.kind, self.rng = kind, np.random.default_rng(seed)
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV's name
        return self.kind != "missing"

    def read(self):
        if self.kind == "dead":
            return False, None
        frame = np.full((540, 960, 3), 40, np.uint8)
        if self.kind == "live":
            frame = np.clip(frame + self.rng.normal(0, 3, frame.shape), 0, 255).astype(np.uint8)
        return True, frame

    def release(self) -> None:
        self.released = True


def opener(kinds: dict[int, str]):
    made: list[FakeCapture] = []

    def open_(index: int) -> FakeCapture:
        made.append(FakeCapture(kinds.get(index, "missing"), index))
        return made[-1]

    open_.made = made  # type: ignore[attr-defined]
    return open_


def test_a_virtual_placeholder_reads_as_still_and_the_webcam_as_live():
    open_ = opener({0: "still", 1: "live"})
    found = find(open_, names=["xpression camera", "FaceTime HD Camera"])
    assert [(c.index, c.name, c.live) for c in found if c.opened] == [(0, "xpression camera", False),
                                                                       (1, "FaceTime HD Camera", True)]
    assert (found[1].width, found[1].height) == (960, 540)
    assert all(c.released for c in open_.made)  # every probe hands the camera back


def test_the_table_points_at_the_first_live_camera():
    out = table([Camera(0, "xpression camera", 960, 540, False), Camera(1, "FaceTime HD Camera", 960, 540, True),
                 Camera(2, "iPhone Camera"), Camera(3, None)])
    assert "still image" in out and "live" in out
    assert "could not open" in out  # a named camera that failed is still listed
    rows = [line for line in out.splitlines() if line[:3].strip().isdigit()]
    assert [row.split()[0] for row in rows] == ["0", "1", "2"]  # index 3 (unnamed, didn't open) is left out
    assert out.endswith("gesturelab live --camera 1")


def test_no_cameras_explains_camera_access():
    assert "Privacy & Security" in table(find(opener({}), names=[]))


def test_a_camera_that_opens_but_sends_nothing_is_not_called_live():
    (camera, *_) = find(opener({0: "dead"}), names=["Broken"])
    assert camera.opened and camera.live is None
    assert "no frames" in table([camera])


def test_macos_names_come_from_system_profiler(monkeypatch):
    monkeypatch.setattr(cameras.sys, "platform", "darwin")
    report = {"SPCameraDataType": [{"_name": "xpression camera"}, {"_name": "FaceTime HD Camera"}]}
    assert macos_names(lambda *a, **k: SimpleNamespace(stdout=json.dumps(report))) == ["xpression camera",
                                                                                    "FaceTime HD Camera"]
    assert macos_names(lambda *a, **k: SimpleNamespace(stdout="not json")) == []
    monkeypatch.setattr(cameras.sys, "platform", "linux")
    assert macos_names(lambda *a, **k: 1 / 0) == []
