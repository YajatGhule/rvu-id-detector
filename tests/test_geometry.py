import numpy as np

from idcard.geometry import is_facing_camera, torso_view

FRAME = (720, 1280, 3)


def person(facing=True, hips=True, face_conf=0.9, shoulder_conf=0.9, scale=1.0, cx=600):
    """Keypoints of an upright person; their left shoulder is image-right when facing us."""
    k = np.zeros((17, 3), np.float32)
    half = 60 * scale
    sign = 1 if facing else -1
    k[0] = (cx, 150 * scale, face_conf)                     # nose
    k[1] = (cx + 15 * sign * scale, 140 * scale, face_conf)  # left eye
    k[2] = (cx - 15 * sign * scale, 140 * scale, face_conf)  # right eye
    k[5] = (cx + sign * half, 250 * scale, shoulder_conf)   # left shoulder
    k[6] = (cx - sign * half, 250 * scale, shoulder_conf)   # right shoulder
    if hips:
        k[11] = (cx + sign * 45 * scale, 420 * scale, 0.9)
        k[12] = (cx - sign * 45 * scale, 420 * scale, 0.9)
    return k


def test_frontal_person_has_torso_between_shoulders():
    view = torso_view(person(), FRAME)
    assert view.ok, view.reason
    x1, y1, x2, y2 = view.box
    assert 540 <= x1 < x2 <= 660
    assert y1 < 250 < y2 <= 420


def test_hips_estimated_when_hidden():
    view = torso_view(person(hips=False), FRAME)
    assert view.ok
    # estimated torso height = 1.4 * shoulder width (120) = 168
    assert abs(view.box[3] - (250 + 0.85 * 168)) <= 1


def test_back_turned_is_not_judged():
    assert not is_facing_camera(person(facing=False))
    assert torso_view(person(facing=False), FRAME).reason == "facing away"
    assert torso_view(person(face_conf=0.1), FRAME).reason == "facing away"


def test_hidden_shoulders_not_judged():
    assert torso_view(person(shoulder_conf=0.2), FRAME).reason == "shoulders not visible"


def test_far_away_person_too_small():
    view = torso_view(person(scale=0.2, cx=200), FRAME)
    assert not view.ok and view.reason == "too small"


def test_side_on_not_judged():
    k = person()
    k[5, 0], k[6, 0] = 610, 590  # shoulders almost on top of each other
    assert torso_view(k, FRAME).reason == "side-on"


def test_box_clipped_to_frame():
    view = torso_view(person(cx=1260), FRAME)
    assert view.box[2] <= 1280
