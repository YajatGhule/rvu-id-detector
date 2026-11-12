# RVU ID Card Detector - plan

## Context

You want a locally run website that watches a webcam (or an uploaded photo/video), finds every
person, and marks each one as wearing or not wearing the RVU ID card, using the lanyard band
colour as a key cue. It doubles as your Web Technology course project and a public portfolio
repo, so the frontend stays at coursework level (like `Html/module2/index.html` + `style.css`:
plain tags, element/class selectors, no frameworks) while the backend is where the real work goes.

Your answers: course project + portfolio, inputs = webcam and uploads, you can collect 50+
photos/clips, output = live overlay plus a violation log with snapshots. No face recognition.

Assumption on "HTML/CSS should run on their own": the pages are static files that open and
look right by double-clicking them (showing "detector offline" where the video goes), and the
detector runs as a separate background process that the pages talk to. Tell me if you meant
something else.

## Shape of the system

```
browser (plain HTML/CSS)            Python backend on 127.0.0.1:8000
  index.html    <img src=/stream>   <- MJPEG, boxes already drawn by the server
  settings.html <form method=post>  -> /api/settings (band colour, tolerance, device, camera)
  upload.html   <form multipart>    -> /api/upload -> annotated result page
  history.html  <iframe>            <- /fragments/history (violations table + snapshots)
```

Boxes are drawn server side and streamed as MJPEG into an `<img>`, and every control is an
HTML form. That means the site needs **zero JavaScript**, which is easy to defend in a viva.
One optional ~30-line `script.js` refreshes the live counters without a page reload.

## Detection pipeline (per frame)

1. **People + pose**: YOLO11n-pose gives a box and 17 keypoints for every person.
2. **Tracking**: ByteTrack gives each person a stable ID across frames.
3. **Torso region**: from shoulders and hips, crop neck-to-waist. If the face keypoints are
   missing (back turned) or the torso is hidden, the person is `UNSURE`, not a violation.
4. **Card/lanyard detector**: a small YOLO11n fine-tuned on your footage, classes
   `lanyard` and `id_card`, run on the torso crop.
5. **Band colour check**: fraction of pixels inside the lanyard box (and along the
   neck-to-chest V) that match the configured band colour in CIELAB/HSV within tolerance.
   Rejects other colleges' lanyards and catches thin straps the detector misses.
6. **Fusion + smoothing**: combine detector confidence and colour score, then vote over the
   last ~15 frames per track with hysteresis so boxes do not flicker.
   Green = wearing, red = not wearing, grey = unsure. Counts shown on the page.
7. **Logging**: when a track stays `NOT WEARING` for N seconds, save one cropped snapshot and a
   row in SQLite (time, track ID, scores). History page lists them, with CSV export and a
   "delete all" button. Nothing is uploaded anywhere.

## Borrowed components

| Piece | Project | Licence |
|---|---|---|
| Pose model, training, export | ultralytics/ultralytics | AGPL-3.0 |
| ByteTrack, box/label drawing | roboflow/supervision | MIT |
| Inference runtime + GPU providers | microsoft/onnxruntime | MIT |
| Vulkan inference | Tencent/ncnn | BSD-3 |
| Labelling with auto-prelabels | CVHub520/X-AnyLabeling (local) | GPL-3.0, tool only |
| Web server | FastAPI + uvicorn | MIT/BSD |

Using Ultralytics means the repo is published under **AGPL-3.0**. Fine for a course/portfolio
project; say so if you would rather avoid it and I will swap in an Apache-licensed detector.

## GPU support

Models are exported once to ONNX and NCNN. A `device.py` module probes the machine at startup,
picks the best backend, and the Settings page lets you override it.

| Hardware | Backend |
|---|---|
| NVIDIA (Windows/Linux) | ONNX Runtime CUDA |
| AMD or Intel integrated, Windows | ONNX Runtime DirectML |
| AMD integrated, Linux | ncnn Vulkan |
| Intel integrated, Linux (your T490) | OpenVINO, ncnn Vulkan as fallback |
| Apple Silicon Mac | ONNX Runtime CoreML |
| Anything else | CPU |

I can only run it on your T490 (Intel UHD 620, Linux). NVIDIA, AMD, Windows and Mac paths will
be written and covered by a CPU smoke test on all three OSes in GitHub Actions, but their GPU
paths stay **untested until someone runs them on that hardware**. `idcard doctor` prints which
backend loaded and the measured FPS so that is quick to check.

## Repo layout: `~/Projects/rvu-id-detector` (new public repo, you as sole author)

```
web/            index.html settings.html upload.html history.html about.html style.css
idcard/
  server.py     FastAPI routes, MJPEG stream, form handlers, static files
  camera.py     webcam thread (OpenCV), latest-frame buffer
  pipeline.py   stages 1-6 glued together
  backends/     onnx.py, ncnn.py, common interface
  device.py     hardware probe + backend choice
  color.py      band colour matching, presets
  fusion.py     scoring, per-track smoothing, states
  store.py      SQLite log + snapshots
  config.py     settings.toml load/save
train/          collect.py, prelabel.py, train.ipynb (Colab), export.py, eval.py
tests/          colour, fusion, device selection, API smoke tests
install.sh / install.ps1, pyproject.toml (uv, Python 3.12), README.md
```

Reuse from `~/Projects/meme-cam`: the `install.sh`/`install.ps1` pattern, uv + Python 3.12 pin,
and the camera loop structure in `memecam/tracking.py`.

## Build order

1. **Skeleton**: repo, uv project, FastAPI serving `web/`, webcam MJPEG stream, static pages.
2. **Baseline, no training**: pose + tracking + colour check only. Already demoable.
3. **Settings menu**: colour picker, tolerance, thresholds, camera, device; saved to
   `settings.toml`.
4. **Data**: you record clips with `train/collect.py` (varied people, lighting, distances,
   card flipped, card under jacket, no card, similar-coloured clothes). `prelabel.py` drafts
   boxes, you correct them in X-AnyLabeling. Target ~300-500 labelled frames.
5. **Train**: `train.ipynb` on a free Colab T4 (the T490 has no usable training GPU), export
   ONNX + NCNN, `eval.py` reports precision/recall on held-out people.
6. **Fusion + log**: full pipeline, history page, snapshots, CSV.
7. **Uploads**: photo and video in, annotated file out.
8. **GPU backends + installers**, `idcard doctor`, CI on Linux/Windows/macOS.
9. **Polish**: README with screenshots, about page explaining the pipeline for the viva.

## What I need from you

- The band colour (photo of the lanyard in daylight is best, or a hex code).
- A clear photo of the card front and back on the lanyard.
- Footage for step 4, after step 2 is working, from at least 5-6 different people who agree
  to be filmed.

## Verification

- `uv run pytest`: colour matching, fusion state machine, device selection, API routes.
- `uv run idcard doctor`: backend chosen and FPS on this machine (aim: 10+ FPS on the T490).
- `uv run idcard serve`, open `http://127.0.0.1:8000`: boxes on several people at once, colour
  change in Settings takes effect live, a no-card person appears in History with a snapshot.
- Open `web/index.html` directly from disk with the server stopped: page renders with the
  offline placeholder.
- `train/eval.py` on people who were not in the training set, reported honestly in the README.
