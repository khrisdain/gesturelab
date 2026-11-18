import numpy as np

from gesturelab.zoom import MAX, MIN, Zoom, crop


def test_crop_keeps_the_centre():
    frame = np.arange(100 * 200).reshape(100, 200)
    middle = crop(frame, 2.0)
    assert middle.shape == (50, 100)
    assert middle[0, 0] == frame[25, 50]
    assert crop(frame, 1.0) is frame


def test_a_zoomed_frame_keeps_its_size_and_magnifies():
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[340:380, 620:660] = 255  # a 40-pixel square in the middle
    zoomed = Zoom(2.0).apply(frame)
    assert zoomed.shape == frame.shape
    assert 70 <= (zoomed[:, :, 0] > 127).sum(axis=0).max() <= 90  # now about 80 pixels tall


def test_plus_and_minus_step_the_zoom_within_bounds():
    zoom = Zoom(1.5)
    assert zoom.key(ord("+")) and zoom.factor > 1.5
    assert zoom.key(ord("-")) and abs(zoom.factor - 1.5) < 0.01
    assert not zoom.key(ord("x"))
    for _ in range(50):
        zoom.key(ord("="))
    assert zoom.factor == MAX
    for _ in range(50):
        zoom.key(ord("-"))
    assert zoom.factor == MIN
    assert Zoom(0.2).factor == MIN and Zoom(9).factor == MAX
