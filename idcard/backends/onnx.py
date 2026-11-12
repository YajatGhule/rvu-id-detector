"""YOLO11-pose inference with ONNX Runtime (numpy + OpenCV only, no torch).

The exported model takes a 1x3x640x640 RGB image scaled to 0..1 and returns (1, 56, 8400):
for each of 8400 candidate boxes: cx, cy, w, h, person score, then 17 keypoints * (x, y, conf).
Scores and keypoint confidences are already passed through a sigmoid by the export.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class PoseDetections:
    boxes: np.ndarray      # (N, 4) float32 x1, y1, x2, y2 in frame pixels
    scores: np.ndarray     # (N,) float32 person confidence
    keypoints: np.ndarray  # (N, 17, 3) float32 x, y, conf in frame pixels

    def __len__(self) -> int:
        return len(self.scores)

    @staticmethod
    def empty() -> "PoseDetections":
        return PoseDetections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                              np.zeros((0, 17, 3), np.float32))


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    """Resize keeping aspect ratio and pad to size x size with grey (like Ultralytics).

    Returns the padded image, the scale factor and the left/top padding.
    """
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_w) / 2, (size - new_h) / 2
    top, left = round(pad_y - 0.1), round(pad_x - 0.1)
    bottom, right = size - new_h - top, size - new_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=(114, 114, 114))
    return padded, scale, left, top


class OnnxPoseModel:
    def __init__(self, model_path: Path, providers: list[str]):
        options = ort.SessionOptions()
        options.log_severity_level = 3  # only errors
        try:
            self.session = ort.InferenceSession(str(model_path), options, providers=providers)
        except Exception:
            # a GPU provider that is listed but broken (missing drivers etc.): use the CPU
            self.session = ort.InferenceSession(str(model_path), options,
                                                providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.size = int(model_input.shape[2]) if isinstance(model_input.shape[2], int) else 640

    @property
    def provider(self) -> str:
        """The provider that actually runs the model, e.g. "CPUExecutionProvider"."""
        return self.session.get_providers()[0]

    def __call__(self, frame_bgr: np.ndarray, conf: float = 0.25,
                 iou: float = 0.45, max_people: int = 100) -> PoseDetections:
        image, scale, pad_x, pad_y = letterbox(frame_bgr, self.size)
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, swapRB=True)  # (1,3,H,W) RGB float32
        output = self.session.run(None, {self.input_name: blob})[0]
        return self.decode(output, conf, iou, max_people, scale, pad_x, pad_y,
                           frame_bgr.shape[:2])

    @staticmethod
    def decode(output: np.ndarray, conf: float, iou: float, max_people: int,
               scale: float, pad_x: float, pad_y: float, frame_hw) -> PoseDetections:
        rows = output[0].T                      # (8400, 56)
        rows = rows[rows[:, 4] >= conf]
        if len(rows) == 0:
            return PoseDetections.empty()

        cx, cy, w, h = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
        scores = rows[:, 4]
        xywh = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)  # top-left form for NMS
        keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), conf, iou, top_k=max_people)
        keep = np.array(keep, dtype=int).reshape(-1)
        if len(keep) == 0:
            return PoseDetections.empty()

        rows, xywh, scores = rows[keep], xywh[keep], scores[keep]
        boxes = np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], axis=1)
        keypoints = rows[:, 5:].reshape(-1, 17, 3).copy()

        # undo the letterbox: remove padding, then undo the resize
        frame_h, frame_w = frame_hw
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - pad_x) / scale).clip(0, frame_w - 1)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - pad_y) / scale).clip(0, frame_h - 1)
        keypoints[:, :, 0] = (keypoints[:, :, 0] - pad_x) / scale
        keypoints[:, :, 1] = (keypoints[:, :, 1] - pad_y) / scale
        return PoseDetections(boxes.astype(np.float32), scores.astype(np.float32),
                              keypoints.astype(np.float32))
