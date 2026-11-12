"""Where is the torso, and can we judge it? Uses the 17 COCO pose keypoints.

Keypoint order (COCO): 0 nose, 1 left eye, 2 right eye, 3 left ear, 4 right ear,
5 left shoulder, 6 right shoulder, 7/8 elbows, 9/10 wrists, 11 left hip, 12 right hip, ...
"Left" means the person's own left, which appears on the RIGHT of the image when they face us.
"""
from dataclasses import dataclass

import numpy as np

NOSE, L_EYE, R_EYE = 0, 1, 2
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12

KEYPOINT_CONF = 0.5       # a keypoint counts as visible at or above this confidence
FACE_CONF = 0.5           # nose or an eye at least this sure -> the face is towards us
MIN_TORSO_W = 32          # pixels; smaller than this is too far away to see a lanyard
MIN_TORSO_H = 40
TORSO_PER_SHOULDER = 1.4  # shoulder-to-hip distance is about 1.4 x the shoulder width
MIN_FRONTAL_RATIO = 0.35  # shoulder width / torso height below this = standing side-on


@dataclass
class TorsoView:
    box: tuple[int, int, int, int] | None  # x1, y1, x2, y2 of the region to check, in pixels
    ok: bool                               # True if we can judge wearing / not wearing
    reason: str                            # why not, e.g. "facing away" ("" when ok)


def _visible(kpts: np.ndarray, index: int) -> bool:
    return kpts[index, 2] >= KEYPOINT_CONF


def is_facing_camera(kpts: np.ndarray) -> bool:
    """Face keypoints are confident AND the shoulders are in mirror order.

    When someone faces the camera their left shoulder is on the right of the image, so
    x(left shoulder) > x(right shoulder). From behind the order flips and the face is hidden.
    """
    face_conf = max(kpts[NOSE, 2], kpts[L_EYE, 2], kpts[R_EYE, 2])
    shoulders_mirrored = kpts[L_SHOULDER, 0] > kpts[R_SHOULDER, 0]
    return face_conf >= FACE_CONF and shoulders_mirrored


def torso_view(kpts: np.ndarray, frame_shape) -> TorsoView:
    """Region from just below the neck to roughly the belly, between the shoulders.

    That is where a lanyard hangs. Hips are used when visible; otherwise they are estimated
    from the shoulder width (people are often cut off by the bottom of the frame or a desk).
    """
    if not (_visible(kpts, L_SHOULDER) and _visible(kpts, R_SHOULDER)):
        return TorsoView(None, False, "shoulders not visible")
    if not is_facing_camera(kpts):
        return TorsoView(None, False, "facing away")

    left, right = kpts[L_SHOULDER, :2], kpts[R_SHOULDER, :2]
    shoulder_mid = (left + right) / 2
    shoulder_w = float(np.linalg.norm(left - right))

    if _visible(kpts, L_HIP) and _visible(kpts, R_HIP):
        hip_mid = (kpts[L_HIP, :2] + kpts[R_HIP, :2]) / 2
        torso_h = float(hip_mid[1] - shoulder_mid[1])
    else:
        torso_h = TORSO_PER_SHOULDER * shoulder_w
    if torso_h <= 0:
        return TorsoView(None, False, "odd pose")
    if shoulder_w < MIN_FRONTAL_RATIO * torso_h:
        return TorsoView(None, False, "side-on")

    # Straps start at the neck (a little above the shoulder line) and the card usually hangs
    # around mid-chest to the belly, so stop at ~85% of the way to the hips. Horizontally stay
    # just inside the shoulders so arms and background are mostly excluded.
    x1 = min(left[0], right[0]) + 0.05 * shoulder_w
    x2 = max(left[0], right[0]) - 0.05 * shoulder_w
    y1 = shoulder_mid[1] - 0.10 * torso_h
    y2 = shoulder_mid[1] + 0.85 * torso_h

    frame_h, frame_w = frame_shape[:2]
    box = (int(max(0, x1)), int(max(0, y1)), int(min(frame_w, x2)), int(min(frame_h, y2)))
    if box[2] - box[0] < MIN_TORSO_W or box[3] - box[1] < MIN_TORSO_H:
        return TorsoView(box, False, "too small")
    return TorsoView(box, True, "")
