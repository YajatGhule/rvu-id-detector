"""The whole per-frame pipeline: people + pose -> tracking -> torso -> colour -> smoothing.

    detector (YOLO11n-pose, ONNX)   boxes + 17 keypoints for every person
    tracker (ByteTrack style)       a stable id per person across frames
    geometry                        torso region, or UNSURE if facing away / hidden / tiny
    color                           how strap-like the band-coloured pixels on the torso are
    fusion                          per-person smoothing + WEARING / NOT_WEARING / UNSURE

Settings are read from the live settings object on every frame, so changes apply at once.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import color, geometry
from .backends.onnx import OnnxPoseModel
from .device import choose_providers
from .fusion import NOT_WEARING, WEARING, Fusion, combine_scores
from .tracker import ByteTracker

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
POSE_MODEL = "yolo11n-pose.onnx"
LOW_CONF = 0.1  # the tracker also looks at weak detections to keep ids (ByteTrack idea)

COLORS_BGR = {WEARING: (60, 180, 60), NOT_WEARING: (40, 40, 220), "UNSURE": (150, 150, 150)}


@dataclass
class Person:
    track_id: int
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in frame pixels
    state: str                      # "WEARING" | "NOT_WEARING" | "UNSURE"
    score: float                    # smoothed 0..1
    new_violation: bool             # True once per track, after violation_seconds


class Pipeline:
    def __init__(self, settings, models_dir: Path | None = None):
        self.settings = settings
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.card_detector = None  # phase 2: trained lanyard / ID card detector goes here
        self.reload()
        self.reset()

    def reload(self) -> None:
        """(Re)build the inference session, e.g. after the device setting changed."""
        providers = choose_providers(self.settings.device)
        self.pose = OnnxPoseModel(self.models_dir / POSE_MODEL, providers)
        self.backend_name = self.pose.provider

    def reset(self) -> None:
        """Forget all people (new camera, new video or new upload)."""
        self.tracker = ByteTracker()
        self.fusion = Fusion()

    def process(self, frame_bgr: np.ndarray, now: float) -> list[Person]:
        s = self.settings
        detections = self.pose(frame_bgr, conf=min(LOW_CONF, s.person_conf))
        self.tracker.high_conf = s.person_conf
        people = []
        for track_id, i in self.tracker.update(detections.boxes, detections.scores):
            score = self._frame_score(frame_bgr, detections.keypoints[i])
            state, smoothed, new_violation = self.fusion.update(track_id, score, now, s)
            box = tuple(int(v) for v in detections.boxes[i])
            people.append(Person(track_id, box, state, round(smoothed, 3), new_violation))
        self.fusion.forget_old(now)
        return people

    def _frame_score(self, frame_bgr: np.ndarray, keypoints: np.ndarray) -> float | None:
        """Evidence (0..1) that this person wears the lanyard in this frame, None = unsure."""
        view = geometry.torso_view(keypoints, frame_bgr.shape)
        if not view.ok:
            return None
        x1, y1, x2, y2 = view.box
        torso = frame_bgr[y1:y2, x1:x2]
        colour = color.lanyard_score(torso, self.settings.band_color, self.settings.tolerance)
        card_score = None
        # HOOK (phase 2): run self.card_detector on `torso` and pass its best lanyard/id_card
        # confidence here; combine_scores() in fusion.py says how the two are weighted.
        return combine_scores(colour.score, card_score)

    def annotate(self, frame_bgr: np.ndarray, people: list[Person]) -> np.ndarray:
        """A copy of the frame with a coloured box and '#id STATE score' label per person."""
        image = frame_bgr.copy()
        thickness = max(2, round(min(image.shape[:2]) / 300))
        font_scale = max(0.5, min(image.shape[:2]) / 900)
        for p in people:
            colour = COLORS_BGR.get(p.state, COLORS_BGR["UNSURE"])
            x1, y1, x2, y2 = p.box
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)
            label = f"#{p.track_id} {p.state} {p.score:.2f}"
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            top = y1 - th - base - 4 if y1 - th - base - 4 >= 0 else y1  # inside if no room
            cv2.rectangle(image, (x1, top), (x1 + tw + 6, top + th + base + 4), colour, -1)
            cv2.putText(image, label, (x1 + 3, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        return image
