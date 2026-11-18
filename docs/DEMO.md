# Demo setup (Samsung laptop, Intel Core Ultra 5 with Intel Arc)

These steps run the website for the CIE-3 demo. Do them once at home, not in the exam room.

## 1. Install (once, needs internet)

1. Install Git and [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 by itself).
2. Open a terminal and run:

```
git clone https://github.com/YajatGhule/rvu-id-detector
cd rvu-id-detector
git checkout web-refactor      # only if the new pages are not on main yet
uv sync
```

The model file `models/yolo11n-pose.onnx` is already in the repository.

## 2. Check that it works

```
uv run idcard doctor
```

It prints the backend that loaded and the speed. If it says CPU, that is fine for the demo: the
Core Ultra 5 processor is fast enough for one webcam. The Intel Arc graphics are only used if the
OpenVINO build of ONNX Runtime is installed. This is optional and has NOT been tested on this
project, so do not depend on it:

```
uv pip uninstall onnxruntime
uv pip install onnxruntime-openvino
uv run idcard doctor      # should now list OpenVINOExecutionProvider
```

If anything goes wrong, undo it with `uv sync --reinstall-package onnxruntime`.

## 3. Run the demo

```
uv run idcard serve
```

Open http://127.0.0.1:8000 in Chrome or Edge. Allow nothing else: the website only needs the
webcam through the Python program, not through the browser. Camera number 0 is the first webcam.
If the picture stays dark, open Settings and try camera 1.

Demo order: Home, Upload (use a photo), Settings (try wrong values), History, Gallery.
Press keys 1 to 6 on any page to jump between pages.

## 4. Backup if the camera fails

Run the website on a video file instead:

```
uv run idcard serve --source path\to\video.mp4
```

Or open the public pages at https://yajatghule.github.io/rvu-id-detector/ . They show a sample
result instead of the live camera, because the detector is not running there.
