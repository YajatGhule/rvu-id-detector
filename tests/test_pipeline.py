from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from idcard.backends.onnx import OnnxPoseModel, letterbox
from idcard.pipeline import MODELS_DIR, POSE_MODEL, Person, Pipeline


def settings(**changes):
    s = SimpleNamespace(band_color="#1e40af", tolerance=25.0, on_threshold=0.6,
                        off_threshold=0.35, violation_seconds=3.0, person_conf=0.4,
                        camera="0", device="auto")
    for k, v in changes.items():
        setattr(s, k, v)
    return s


needs_model = pytest.mark.skipif(not (MODELS_DIR / POSE_MODEL).exists(),
                                 reason="models/yolo11n-pose.onnx not exported")


def test_letterbox_keeps_aspect_and_centres():
    img = np.zeros((360, 1280, 3), np.uint8)
    padded, scale, pad_x, pad_y = letterbox(img, 640)
    assert padded.shape == (640, 640, 3)
    assert scale == 0.5 and pad_x == 0 and pad_y == 230


def test_decode_scales_back_to_frame_and_applies_nms():
    output = np.zeros((1, 56, 8400), np.float32)
    # two overlapping candidates for one person + one weak candidate, in 640x640 letterbox
    for i, (cx, conf) in enumerate([(320, 0.9), (322, 0.8), (100, 0.1)]):
        output[0, :5, i] = (cx, 320, 100, 200, conf)
        output[0, 5::3, i], output[0, 6::3, i], output[0, 7::3, i] = cx, 320, 0.9
    det = OnnxPoseModel.decode(output, conf=0.4, iou=0.45, max_people=100, scale=0.5,
                               pad_x=0, pad_y=140, frame_hw=(720, 1280))
    assert len(det) == 1
    assert np.allclose(det.boxes[0], [540, 160, 740, 560])
    assert np.allclose(det.keypoints[0, 0], [640, 360, 0.9])


@needs_model
def test_blank_frame_has_no_people():
    pipe = Pipeline(settings())
    assert pipe.backend_name.endswith("ExecutionProvider")
    frame = np.full((480, 640, 3), 127, np.uint8)
    for i in range(3):
        assert pipe.process(frame, now=i * 0.1) == []


@needs_model
def test_reset_and_reload():
    pipe = Pipeline(settings(device="cpu"))
    pipe.reset()
    pipe.settings.device = "cuda"  # not on this machine -> falls back without error
    pipe.reload()
    assert pipe.backend_name


@needs_model
def test_annotate_returns_new_image():
    pipe = Pipeline(settings())
    frame = np.zeros((300, 400, 3), np.uint8)
    people = [Person(1, (10, 10, 100, 200), "WEARING", 0.9, False),
              Person(2, (150, 0, 300, 290), "NOT_WEARING", 0.1, True),
              Person(3, (310, 50, 390, 250), "UNSURE", 0.0, False)]
    out = pipe.annotate(frame, people)
    assert out is not frame and out.shape == frame.shape
    assert frame.sum() == 0 and out.sum() > 0
    assert tuple(out[100, 10]) == (60, 180, 60)    # green WEARING box edge
    assert tuple(out[150, 150]) == (40, 40, 220)   # red NOT_WEARING box edge


def test_missing_model_raises(tmp_path: Path):
    with pytest.raises(Exception):
        Pipeline(settings(), models_dir=tmp_path)
