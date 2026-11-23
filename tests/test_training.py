from pathlib import Path

import numpy as np
import pytest

from gesturelab import dataset
from gesturelab.smoothing import Smoother
from gesturelab.synth import GESTURES
from gesturelab.synth import dataset as synthetic
from gesturelab.train import Classifier, report, save, train


@pytest.fixture(scope="module")
def bundle():
    hands, labels, sessions = synthetic(per_class=120, sessions=4, seed=1)
    return train(hands, labels, sessions, models=["mlp", "forest"])


def test_training_scores_on_held_out_sessions_and_keeps_the_best(bundle):
    best = bundle["best"]
    assert best.scheme.startswith("held-out sessions")
    assert best.macro_f1 == max(r.macro_f1 for r in bundle["candidates"])
    assert best.macro_f1 > 0.9
    assert bundle["labels"] == sorted(GESTURES)
    assert "<- saved" in report(bundle)


def test_a_saved_model_classifies_fresh_hands(bundle, tmp_path: Path):
    save(bundle, tmp_path / "m.joblib")
    classifier = Classifier(tmp_path / "m.joblib")
    hands, labels, _ = synthetic(per_class=8, sessions=1, seed=99)  # a person never seen in training
    predicted = [classifier.labels[int(np.argmax(classifier.probabilities(h)))] for h in hands]
    assert np.mean(np.array(predicted) == np.array(labels)) > 0.85


def test_a_model_from_older_features_is_refused(bundle, tmp_path: Path):
    save({**bundle, "feature_version": 0}, tmp_path / "old.joblib")
    with pytest.raises(ValueError, match="retrain"):
        Classifier(tmp_path / "old.joblib")


def test_with_too_few_sessions_the_report_says_the_score_is_optimistic():
    hands, labels, sessions = synthetic(per_class=40, sessions=1, seed=2)
    assert "3+ sessions" in train(hands, labels, sessions, models=["forest"])["best"].scheme


def test_one_gesture_is_not_enough():
    hands, labels, sessions = synthetic(per_class=30, sessions=3)
    only = [i for i, label in enumerate(labels) if label == "fist"]
    with pytest.raises(ValueError):
        train([hands[i] for i in only], ["fist"] * len(only), [sessions[i] for i in only])


def test_samples_round_trip_through_csv(tmp_path: Path):
    hands, labels, _ = synthetic(per_class=16, sessions=2, seed=4)
    path = tmp_path / "samples.csv"
    dataset.append(path, "peace", "s1", hands[:5])
    dataset.append(path, "fist", "s2", hands[5:8])
    loaded = dataset.load(path)
    assert loaded.counts() == {"fist": 3, "peace": 5}
    assert loaded.sessions[0] == "s1" and loaded.hands[0].handedness == hands[0].handedness
    assert np.allclose(loaded.hands[0].points, hands[0].points, atol=1e-5)


def test_the_smoother_waits_for_a_steady_confident_gesture():
    smoother = Smoother(["fist", "peace"], alpha=1.0, threshold=0.7, hold=3)
    peace, fist, unsure = np.array([0.1, 0.9]), np.array([0.9, 0.1]), np.array([0.5, 0.5])
    assert [smoother.update(peace) for _ in range(3)] == [None, None, "peace"]
    assert smoother.update(fist) == "peace"  # one stray frame doesn't switch it
    assert smoother.update(peace) == "peace"
    assert smoother.update(unsure) is None  # nothing confident: no gesture
    assert smoother.update(None) is None and smoother.confidence == 0.0  # hand gone


def test_the_smoother_averages_out_flicker():
    smoother = Smoother(["fist", "peace"], alpha=0.35, threshold=0.7, hold=2)
    frames = [np.array([0.05, 0.95])] * 6 + [np.array([0.6, 0.4])] + [np.array([0.05, 0.95])] * 2
    assert [smoother.update(f) for f in frames][-3:] == ["peace", "peace", "peace"]


def test_a_lost_hand_clears_the_history_so_the_next_gesture_starts_fresh():
    smoother = Smoother(["fist", "peace"], alpha=0.35, threshold=0.7, hold=2)
    for _ in range(10):
        smoother.update(np.array([0.02, 0.98]))  # a long, confident peace sign
    assert smoother.current == "peace"
    smoother.update(None)  # the hand leaves the frame
    fist = np.array([0.95, 0.05])
    # Without the reset, ten frames of averaged "peace" would delay the fist for several frames.
    assert [smoother.update(fist) for _ in range(2)] == [None, "fist"]
