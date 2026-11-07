"""Training and evaluating a gesture classifier.

Frames recorded seconds apart are nearly identical, so a random train/test split would score a
model on frames it has effectively already seen. Evaluation therefore holds out whole recording
sessions (GroupKFold): the score is how well it recognises a gesture recorded at another time,
in other light, with the hand held differently.

A few model families are compared on that held-out score; the best is refitted on everything
and saved with its labels, metrics and the feature version it was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import FEATURE_VERSION, feature_matrix
from .hand import Hand

CANDIDATES: dict[str, Pipeline] = {
    "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=600,
                                                         early_stopping=True, random_state=0)),
    # SVC has no probabilities of its own; calibrate them (the smoother needs probabilities, not just labels).
    "svm": make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(C=10, gamma="scale"), ensemble=False, cv=3)),
    "forest": make_pipeline(RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1)),
}


@dataclass
class Evaluation:
    model: str
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    confusion: list[list[int]]
    scheme: str


def evaluate(name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, labels: list[str]) -> Evaluation:
    sessions = len(set(groups))
    if sessions >= 3:
        splitter: Any = GroupKFold(n_splits=min(5, sessions))
        splits = splitter.split(X, y, groups)
        scheme = f"held-out sessions ({min(5, sessions)}-fold)"
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        splits = splitter.split(X, y)
        scheme = "stratified 5-fold (record 3+ sessions per gesture for an honest score)"
    predicted = np.empty_like(y)
    for train, test in splits:
        model = clone(CANDIDATES[name]).fit(X[train], y[train])
        predicted[test] = model.predict(X[test])
    per_class = f1_score(y, predicted, labels=labels, average=None, zero_division=0)
    return Evaluation(
        model=name, accuracy=float(accuracy_score(y, predicted)), macro_f1=float(f1_score(y, predicted, average="macro")),
        per_class_f1={label: float(score) for label, score in zip(labels, per_class, strict=True)},
        confusion=confusion_matrix(y, predicted, labels=labels).tolist(), scheme=scheme,
    )


def train(hands: list[Hand], labels: list[str], sessions: list[str], models: list[str] | None = None) -> dict[str, Any]:
    if len(set(labels)) < 2:
        raise ValueError("record at least two gestures before training")
    X, y, groups = feature_matrix(hands), np.array(labels), np.array(sessions)
    classes = sorted(set(labels))
    results = [evaluate(name, X, y, groups, classes) for name in (models or list(CANDIDATES))]
    best = max(results, key=lambda r: r.macro_f1)
    pipeline = clone(CANDIDATES[best.model]).fit(X, y)
    return {
        "pipeline": pipeline, "labels": list(pipeline.classes_), "feature_version": FEATURE_VERSION,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"), "samples": len(y), "sessions": len(set(sessions)),
        "best": best, "candidates": results,
    }


def save(bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


class Classifier:
    """A saved model, ready to score hands."""

    def __init__(self, path: Path) -> None:
        bundle = joblib.load(path)
        if bundle.get("feature_version") != FEATURE_VERSION:
            raise ValueError(f"{path} was trained with feature version {bundle.get('feature_version')}, "
                             f"this code uses {FEATURE_VERSION}: retrain it")
        self.pipeline = bundle["pipeline"]
        self.labels: list[str] = bundle["labels"]
        self.bundle = bundle

    def probabilities(self, hand: Hand) -> np.ndarray:
        return self.pipeline.predict_proba(feature_matrix([hand]))[0]


def report(bundle: dict[str, Any]) -> str:
    best: Evaluation = bundle["best"]
    lines = [f"{bundle['samples']} samples, {bundle['sessions']} sessions, evaluated on {best.scheme}", "",
             f"{'model':<8} {'accuracy':>9} {'macro F1':>9}"]
    for result in bundle["candidates"]:
        mark = "  <- saved" if result.model == best.model else ""
        lines.append(f"{result.model:<8} {result.accuracy:>9.1%} {result.macro_f1:>9.3f}{mark}")
    labels = bundle["labels"]
    width = max(len(label) for label in labels)
    lines += ["", "Per gesture (F1) and confusion (rows = true, columns = predicted):",
              " " * (width + 9) + " ".join(f"{label[:6]:>6}" for label in labels)]
    for label, row in zip(labels, best.confusion, strict=True):
        lines.append(f"{label:<{width}} {best.per_class_f1[label]:>6.2f}  " + " ".join(f"{n:>6}" for n in row))
    return "\n".join(lines)
