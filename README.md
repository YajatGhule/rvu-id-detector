# RVU ID Card Detector

A website that runs on your own computer and watches a webcam, or a photo or video you upload.
It finds every person in the picture and marks each one as wearing or not wearing the college
ID card on its lanyard.

This is a first year Web Technology course project at RV University. The pages are plain HTML
and CSS with no JavaScript. The detection work happens in a Python backend.

## Status

Work in progress.

- Working: person detection, tracking of several people at once, lanyard colour check,
  live stream with boxes, settings page, photo and video upload, violation log with snapshots.
- In progress: a trained detector for "wearing" and "not wearing" (see `train/train.ipynb`).
- Only tested on one laptop (Linux, Intel CPU). The GPU options are written but not tested.

Measured so far: with colour only, 164 of 200 labelled test photos were judged correctly.
Nobody without a card was marked as wearing one, and 36 of 100 wearers were missed.

## Run it

You need [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run idcard serve
```

Then open http://127.0.0.1:8000 in a browser. `uv run idcard doctor` shows which backend
loaded and the speed on your machine. `uv run pytest` runs the tests.

## How it works

1. A YOLO11 pose model finds each person and their shoulders and hips.
2. A tracker gives each person a number that stays the same between frames.
3. The chest area is cut out. People who are turned away or too far off are marked UNSURE.
4. The chest is checked for a thin strap in the band colour chosen on the settings page.
5. The result is smoothed over about 15 frames so the label does not flicker.
6. Someone who stays NOT WEARING for a few seconds is saved to the log with a snapshot.

Boxes are drawn by the server and sent to the page as a stream of JPEG images, and every
control is an HTML form. That is why the pages need no JavaScript.

## Privacy

Everything stays on your computer. There is no face recognition and nothing is uploaded.
Snapshots and the log live in the `data/` folder, which is not part of this repository.

## Credits

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) for the pose model and training (AGPL-3.0)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) to run the model (MIT)
- [FastAPI](https://github.com/fastapi/fastapi) and [uvicorn](https://github.com/encode/uvicorn) for the web server
- Tracker idea from [ByteTrack](https://github.com/FoundationVision/ByteTrack)
- Training data: [id-detectionn-2](https://universe.roboflow.com/id-card-detection-ozsif/id-detectionn-2) on Roboflow Universe (CC BY 4.0)
- Test photos: [Aravinth1525/ID_Card_Detection_Final](https://github.com/Aravinth1525/ID_Card_Detection_Final) (MIT)

## Licence

AGPL-3.0, because the YOLO11 model and code from Ultralytics use that licence. See `LICENSE`.
