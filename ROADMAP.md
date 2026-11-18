# Roadmap

gesturelab today recognises still hand poses, is trained on your own recordings, and runs on your machine. Next, in order:

1. **A public benchmark.** Train and evaluate on HaGRID (18 gestures, thousands of people), scored on people the model never saw. Use the dataset's landmark annotations if it ships them, so no images need downloading.
2. **Dynamic gestures.** Swipe, wave, pinch-and-drag: a small PyTorch temporal model (1D CNN or GRU) over about a second of landmarks, with gesture start/end detection and a latency budget for live use.
3. **A browser demo.** Export to ONNX and run it with MediaPipe's JavaScript hand tracker, hosted statically (GitHub Pages). Video never leaves the visitor's device.
4. **Few-shot gestures.** Learn a landmark embedding so a new gesture can be added from about five examples, classified against the mean of its examples, without retraining.
