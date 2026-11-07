# gesturelab

**Teach your computer your own hand gestures with a webcam.** Record a few seconds of each gesture, train a classifier on them, and gesturelab recognises them live, with a confidence bar for every gesture and an event stream other programs can listen to.

```bash
gesturelab record peace          # ~20 s of you holding up a peace sign
gesturelab record thumbs_up
gesturelab record fist
gesturelab train                 # compares three models on held-out sessions, saves the best
gesturelab live                  # recognise them from the webcam
```

## How it works

```
webcam frame ─▶ MediaPipe hand landmarker ─▶ 21 landmarks ─▶ normalise ─▶ 91 features ─▶ classifier ─▶ smoother ─▶ gesture
```

1. **Hand tracking.** MediaPipe's hand landmarker finds a hand in each frame and returns 21 3-D landmarks (wrist, knuckles, joints, fingertips) and whether it is a left or right hand.
2. **Normalisation, so a gesture looks the same wherever it is.** Landmarks are moved so the wrist is at the origin, left hands are mirrored onto right ones, the hand is rotated so the palm points straight up, and it is scaled so the palm has length 1. The tests check that moving, resizing, rotating or mirroring a hand doesn't change its features at all.
3. **Features** ([`features.py`](src/gesturelab/features.py)): the 63 normalised coordinates, plus how bent each of the 15 finger joints is, thumb-to-fingertip distances (pinches), fingertip reach from the wrist, and the spread between neighbouring fingers.
4. **Training** ([`train.py`](src/gesturelab/train.py)): an MLP, an RBF SVM (with calibrated probabilities) and a random forest are scored with **cross-validation that holds out whole recording sessions**. Frames recorded a moment apart are near-identical, so a random split would test the model on frames it has effectively seen; holding out sessions measures what matters, recognising a gesture made another time, in other light. The best model by macro-F1 is refitted on everything and saved with its labels, scores and feature version.
5. **Smoothing** ([`smoothing.py`](src/gesturelab/smoothing.py)): probabilities are averaged across frames, and a gesture is only reported once it has led confidently for a few frames in a row. A hand moving between poses reads as "none" instead of flickering through wrong guesses.

Recordings store **landmarks, not images**: 63 numbers a frame in a CSV, and no picture of whoever recorded them.

## Install

Python 3.11+ and a webcam.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

The first `record` or `live` downloads MediaPipe's hand landmarker model (about 7.5 MB, from Google's model storage) to `~/.cache/gesturelab`. On macOS, allow camera access for your terminal in **System Settings › Privacy & Security › Camera**.

## Recording good data

- Record each gesture in **3 or more short sessions** (different times, light, distance). `gesturelab info` shows what you have; `train` reports an honest held-out-session score only when there are 3+ sessions.
- Move your hand around while recording: nearer and further, turned a little, both hands. That variety is what makes it robust.
- Record a `none` gesture too (relaxed hands, half-poses, scratching your face) so the model learns when *not* to fire.

## No camera? Try it with synthetic hands

`synth.py` poses a simple kinematic hand model (bone chains with per-finger curl) into 8 gestures, varying hand proportions per simulated person plus orientation, size, position and tracker noise:

```bash
gesturelab synth                               # 2,400 samples: open_palm, fist, thumbs_up, point, peace, rock, call_me, ok
gesturelab train --data data/synthetic.csv
```

Synthetic hands are far tidier than real ones (every model scores ~100% on them), so they are for tests and for trying the pipeline, not a substitute for recording your own.

## Using gestures in other programs

`gesturelab live --events` prints a JSON line whenever the recognised gesture changes:

```json
{"gesture": "thumbs_up", "confidence": 0.94, "t_ms": 18233}
```

Pipe it into anything: a script that skips a song on `peace`, mutes a call on `fist`, advances slides on `point`.

## Project layout

| File | |
|---|---|
| [`tracker.py`](src/gesturelab/tracker.py) | MediaPipe hand landmarker in video mode |
| [`features.py`](src/gesturelab/features.py) | normalisation and the 91 features |
| [`train.py`](src/gesturelab/train.py) | model comparison on held-out sessions, saving and loading |
| [`smoothing.py`](src/gesturelab/smoothing.py) | steadying live predictions |
| [`dataset.py`](src/gesturelab/dataset.py) | the samples CSV |
| [`synth.py`](src/gesturelab/synth.py) | synthetic hands |
| [`draw.py`](src/gesturelab/draw.py), [`cli.py`](src/gesturelab/cli.py) | overlays and commands |

## Tests

```bash
pytest
```

Feature invariance (translation, scale, rotation, mirroring), training and held-out-session evaluation, classifying hands of an unseen simulated person, refusing models built on older features, CSV round trips, and the smoother's behaviour on steady, flickering and uncertain input. CI runs lint (ruff), types (mypy) and the tests.
