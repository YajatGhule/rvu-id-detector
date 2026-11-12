import numpy as np

from idcard.tracker import ByteTracker, iou_matrix


def test_iou():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
    assert np.allclose(iou_matrix(a, b), [[1.0, 1 / 3, 0.0]])


def test_ids_stay_with_moving_people():
    t = ByteTracker(high_conf=0.4)
    ids = {}
    for step in range(20):
        a = [100 + 5 * step, 100, 200 + 5 * step, 400]   # walking right
        b = [600 - 5 * step, 100, 700 - 5 * step, 400]   # walking left
        boxes = np.array([b, a] if step % 2 else [a, b], np.float32)  # order changes
        for track_id, i in t.update(boxes, np.array([0.9, 0.9])):
            side = "a" if boxes[i][0] < 400 and step < 10 or boxes[i][0] < 300 else "b"
            ids.setdefault(side, set()).add(track_id)
    assert ids == {"a": {1}, "b": {2}}


def test_low_confidence_keeps_existing_track_but_does_not_start_one():
    t = ByteTracker(high_conf=0.5)
    box = np.array([[100, 100, 200, 400]], np.float32)
    assert t.update(box, [0.9]) == [(1, 0)]
    assert t.update(box, [0.2]) == [(1, 0)]          # blurry frame: same id
    other = np.array([[600, 100, 700, 400]], np.float32)
    assert t.update(other, [0.2]) == []               # weak and new: ignored


def test_lost_track_returns_and_old_tracks_expire():
    t = ByteTracker(high_conf=0.5, max_lost=5)
    box = np.array([[100, 100, 200, 400]], np.float32)
    t.update(box, [0.9])
    for _ in range(3):
        t.update(np.zeros((0, 4)), np.zeros(0))
    assert t.update(box, [0.9]) == [(1, 0)]
    for _ in range(6):
        t.update(np.zeros((0, 4)), np.zeros(0))
    assert t.update(box, [0.9]) == [(2, 0)]
