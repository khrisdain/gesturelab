"""gesturelab: record gestures, train, and recognise them live.

    gesturelab record peace              # hold up a peace sign for ~20 s, moving your hand around
    gesturelab record thumbs_up
    gesturelab info                      # how much you have recorded
    gesturelab train                     # compare models on held-out sessions, save the best
    gesturelab live                      # recognise gestures from the webcam
    gesturelab cameras                   # which camera number is which (add --show to see each one)

    gesturelab synth                     # or start without a camera: synthetic hands for 8 gestures
    gesturelab train --data data/synthetic.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

DATA = Path("data/samples.csv")
MODEL = Path("models/gestures.joblib")


def open_camera(index: int):  # noqa: ANN201 - cv2.VideoCapture
    import cv2

    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        sys.exit(f"Could not open camera {index}. On macOS, allow camera access for your terminal in "
                 "System Settings > Privacy & Security > Camera, then try again. "
                 "`gesturelab cameras` lists the cameras you have.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    return camera


def frames(camera):  # noqa: ANN001, ANN201
    """Mirrored frames (selfie view) with millisecond timestamps, until the window is closed or q is pressed."""
    import cv2

    started = time.monotonic()
    while True:
        ok, frame = camera.read()
        if not ok:
            return
        yield cv2.flip(frame, 1), int((time.monotonic() - started) * 1000)


def show(frame) -> bool:  # noqa: ANN001
    """Show a frame; False once the user presses q or Esc."""
    import cv2

    cv2.imshow("gesturelab", frame)
    return (cv2.waitKey(1) & 0xFF) not in (ord("q"), 27)


def record(args: argparse.Namespace) -> None:
    from . import dataset, draw
    from .tracker import HandTracker

    label = args.label.strip().lower().replace(" ", "_")
    session = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    tracker, camera = HandTracker(), open_camera(args.camera)
    captured, frame_number, start_at = [], 0, None
    try:
        for frame, ms in frames(camera):
            hands = tracker.detect(frame, ms)
            for hand in hands:
                draw.skeleton(frame, hand)
            start_at = start_at if start_at is not None else ms + args.countdown * 1000
            if ms < start_at:
                draw.banner(frame, f"Get ready: {label} in {(start_at - ms) // 1000 + 1}",
                            "Move it around, turn it, go nearer and further, switch hands.")
            else:
                frame_number += 1
                if hands and frame_number % args.every == 0:
                    captured.append(hands[0])
                left = args.seconds - (ms - start_at) / 1000
                draw.banner(frame, f"Recording {label}: {len(captured)} samples", f"{max(left, 0):.0f} s left · q to stop")
                if left <= 0:
                    break
            if not show(frame):
                break
    finally:
        camera.release()
        tracker.close()
        import cv2

        cv2.destroyAllWindows()
    if not captured:
        sys.exit("No hand was seen, so nothing was saved.")
    dataset.append(args.data, label, session, captured)
    print(f"Saved {len(captured)} samples of '{label}' (session {session}) to {args.data}")


def info(args: argparse.Namespace) -> None:
    from . import dataset

    samples = dataset.load(args.data)
    per_session: dict[str, set[str]] = {}
    for label, session in zip(samples.labels, samples.sessions, strict=True):
        per_session.setdefault(label, set()).add(session)
    print(f"{len(samples)} samples in {args.data}")
    for label, count in samples.counts().items():
        sessions = len(per_session[label])
        hint = "" if sessions >= 3 else "  (record a few more sessions for a fair score)"
        print(f"  {label:<16} {count:>5} samples  {sessions} session{'s' if sessions != 1 else ''}{hint}")


def train(args: argparse.Namespace) -> None:
    from . import dataset
    from .train import report, save
    from .train import train as fit

    samples = dataset.load(args.data)
    bundle = fit(samples.hands, samples.labels, samples.sessions, args.models)
    save(bundle, args.out)
    print(report(bundle))
    print(f"\nSaved the {bundle['best'].model} model to {args.out}")


def live(args: argparse.Namespace) -> None:
    import numpy as np

    from . import draw
    from .smoothing import Smoother
    from .tracker import HandTracker
    from .train import Classifier

    classifier = Classifier(args.model)
    smoother = Smoother(classifier.labels, threshold=args.threshold)
    tracker, camera = HandTracker(), open_camera(args.camera)
    previous, fps, last = None, 0.0, time.monotonic()
    try:
        for frame, ms in frames(camera):
            hands = tracker.detect(frame, ms)
            probabilities = classifier.probabilities(hands[0]) if hands else None
            gesture = smoother.update(probabilities)
            for hand in hands:
                draw.skeleton(frame, hand)
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            title = gesture.replace("_", " ") if gesture else ("…" if hands else "show me a hand")
            draw.banner(frame, title, f"{smoother.confidence:.0%} confident · {fps:.0f} fps · q to quit")
            draw.probabilities(frame, classifier.labels, probabilities if probabilities is not None
                               else np.zeros(len(classifier.labels)), gesture)
            if gesture != previous:
                if args.events:  # one JSON line per change, for piping into other programs
                    print(json.dumps({"gesture": gesture, "confidence": round(smoother.confidence, 3), "t_ms": ms}),
                          flush=True)
                previous = gesture
            if not show(frame):
                break
    finally:
        camera.release()
        tracker.close()
        import cv2

        cv2.destroyAllWindows()


def cameras(args: argparse.Namespace) -> None:
    from .cameras import find, table

    print("Checking cameras (each one switches on for a moment)…", file=sys.stderr)
    found = find()
    print(table(found))
    if not args.show:
        return
    import cv2

    for camera in (c for c in found if c.width > 1):
        capture = open_camera(camera.index)
        try:
            print(f"Showing camera {camera.index}: any key for the next one, q to stop", file=sys.stderr)
            for frame, _ in frames(capture):
                label = f"camera {camera.index}: {camera.name or 'unnamed'}"
                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("gesturelab", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return
                if key != 255:
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()


def synth(args: argparse.Namespace) -> None:
    from . import dataset
    from .synth import dataset as generate

    hands, labels, sessions = generate(args.per_class, args.sessions, args.seed)
    if args.out.exists():
        args.out.unlink()
    for label in sorted(set(labels)):
        for session in sorted(set(sessions)):
            batch = [h for h, lab, s in zip(hands, labels, sessions, strict=True) if lab == label and s == session]
            dataset.append(args.out, label, session, batch)
    print(f"Wrote {len(hands)} synthetic samples ({len(set(labels))} gestures, {args.sessions} sessions) to {args.out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gesturelab", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("record", help="record samples of one gesture from the webcam")
    p.add_argument("label")
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--countdown", type=int, default=3)
    p.add_argument("--every", type=int, default=2, help="keep every Nth frame (neighbouring frames are near-duplicates)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--data", type=Path, default=DATA)
    p.set_defaults(run=record)

    p = sub.add_parser("info", help="what has been recorded")
    p.add_argument("--data", type=Path, default=DATA)
    p.set_defaults(run=info)

    p = sub.add_parser("train", help="train and evaluate, save the best model")
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--out", type=Path, default=MODEL)
    p.add_argument("--models", nargs="+", choices=["mlp", "svm", "forest"])
    p.set_defaults(run=train)

    p = sub.add_parser("live", help="recognise gestures from the webcam")
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--events", action="store_true", help="print a JSON line whenever the gesture changes")
    p.set_defaults(run=live)

    p = sub.add_parser("cameras", help="list cameras by number and name, to pick one for --camera")
    p.add_argument("--show", action="store_true", help="then show each camera's picture, labelled with its number")
    p.set_defaults(run=cameras)

    p = sub.add_parser("synth", help="write synthetic samples (no camera needed)")
    p.add_argument("--out", type=Path, default=Path("data/synthetic.csv"))
    p.add_argument("--per-class", type=int, default=300)
    p.add_argument("--sessions", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(run=synth)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
