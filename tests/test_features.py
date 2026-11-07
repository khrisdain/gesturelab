import numpy as np
import pytest

from gesturelab.features import FEATURE_VERSION, feature_names, features, normalize
from gesturelab.hand import MIDDLE, WRIST, Hand
from gesturelab.synth import GESTURES, sample


def hand(gesture: str = "peace", seed: int = 0) -> Hand:
    return sample(gesture, np.random.default_rng(seed))


def moved(h: Hand, dx: float = 0.0, scale: float = 1.0, angle: float = 0.0, mirror: bool = False) -> Hand:
    points = h.points.copy()
    centre = points[WRIST].copy()
    points -= centre
    c, s = np.cos(angle), np.sin(angle)
    x, y = points[:, 0].copy(), points[:, 1].copy()
    points[:, 0], points[:, 1] = c * x - s * y, s * x + c * y
    points = points * scale + centre
    points[:, 0] += dx
    handedness = h.handedness
    if mirror:
        points[:, 0] = 1 - points[:, 0]
        handedness = "Left" if h.handedness == "Right" else "Right"
    return Hand(points=points, handedness=handedness)


def test_the_palm_ends_up_pointing_straight_up_at_unit_length():
    points = normalize(hand())
    knuckle = points[MIDDLE[0]]
    assert np.allclose(points[WRIST], 0)
    assert np.isclose(knuckle[0], 0, atol=1e-9) and knuckle[1] < 0  # straight up (image y grows downwards)
    assert np.isclose(np.linalg.norm(knuckle), 1)


@pytest.mark.parametrize("change", [
    {"dx": 0.2}, {"scale": 1.7}, {"scale": 0.5}, {"angle": 0.8}, {"angle": -2.5}, {"mirror": True},
    {"dx": -0.1, "scale": 1.3, "angle": 1.1, "mirror": True},
])
def test_features_ignore_position_size_tilt_and_which_hand(change):
    original = hand()
    assert np.allclose(features(original), features(moved(original, **change)), atol=1e-6)


def test_different_gestures_give_different_features():
    vectors = {g: features(hand(g, seed=3)) for g in GESTURES}
    names = list(vectors)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert np.linalg.norm(vectors[a] - vectors[b]) > 0.3, (a, b)


def test_feature_names_match_the_vector():
    assert len(feature_names()) == len(features(hand())) and FEATURE_VERSION == 1


def test_a_hand_needs_21_landmarks():
    with pytest.raises(ValueError):
        Hand(points=np.zeros((20, 3)))
