"""Small ByteTrack-style tracker: gives each person a stable id across frames.

supervision 0.31 (what this project installs) no longer ships ByteTrack, so this is a compact
version of the same idea (Zhang et al., "ByteTrack", 2022), with greedy IoU matching instead
of a Kalman filter + Hungarian algorithm:
  1. match confident detections to existing tracks by box overlap (IoU),
  2. match the leftover LOW-confidence detections to the leftover tracks, so a person who is
     briefly blurred or half hidden keeps their id instead of getting a new one,
  3. confident detections that matched nothing start new tracks,
  4. tracks unseen for `max_lost` frames are deleted.
Each track remembers its last box and how fast it was moving, to predict the next box.
"""
from dataclasses import dataclass

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and every box in b (M,4), boxes as x1,y1,x2,y2."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def greedy_match(iou: np.ndarray, min_iou: float) -> list[tuple[int, int]]:
    """Pairs (row, col) taking the highest IoU first; each row and col used at most once."""
    pairs, used_rows, used_cols = [], set(), set()
    for flat in np.argsort(-iou, axis=None):
        r, c = np.unravel_index(flat, iou.shape)
        if iou[r, c] < min_iou:
            break
        if r in used_rows or c in used_cols:
            continue
        pairs.append((int(r), int(c)))
        used_rows.add(r)
        used_cols.add(c)
    return pairs


@dataclass
class _Track:
    track_id: int
    box: np.ndarray
    velocity: np.ndarray
    lost: int = 0

    def predicted(self) -> np.ndarray:
        return self.box + self.velocity * (self.lost + 1)


class ByteTracker:
    def __init__(self, high_conf: float = 0.4, low_conf: float = 0.1,
                 match_iou: float = 0.3, max_lost: int = 30) -> None:
        self.high_conf, self.low_conf = high_conf, low_conf
        self.match_iou, self.max_lost = match_iou, max_lost
        self.tracks: list[_Track] = []
        self.next_id = 1

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> list[tuple[int, int]]:
        """Feed one frame of detections. Returns (track_id, detection_index) for every
        detection that belongs to a track this frame. Unmatched low-confidence detections
        are not returned (they are probably not people)."""
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        scores = np.asarray(scores, np.float32).reshape(-1)
        high = np.flatnonzero(scores >= self.high_conf)
        low = np.flatnonzero((scores >= self.low_conf) & (scores < self.high_conf))

        # stage 1: confident detections, stage 2: low-confidence ones on the leftover tracks
        pairs, lost_tracks, new_dets = self._associate(list(range(len(self.tracks))), high,
                                                       boxes, self.match_iou)
        pairs2, lost_tracks, _ = self._associate(lost_tracks, low, boxes, 0.5)

        results = []
        for t, d in pairs + pairs2:
            self._move(self.tracks[t], boxes[d])
            results.append((self.tracks[t].track_id, d))
        for t in lost_tracks:
            self.tracks[t].lost += 1
        for d in new_dets:
            track = _Track(self.next_id, boxes[d].copy(), np.zeros(4, np.float32))
            self.next_id += 1
            self.tracks.append(track)
            results.append((track.track_id, d))
        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]
        return results

    def _associate(self, track_ids: list[int], det_ids: np.ndarray, boxes: np.ndarray,
                   min_iou: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Match tracks to detections. Returns (pairs, unmatched tracks, unmatched dets)."""
        det_ids = [int(d) for d in det_ids]
        if not track_ids or not det_ids:
            return [], track_ids, det_ids
        predicted = np.array([self.tracks[t].predicted() for t in track_ids])
        matches = greedy_match(iou_matrix(predicted, boxes[det_ids]), min_iou)
        pairs = [(track_ids[r], det_ids[c]) for r, c in matches]
        used_t, used_d = {t for t, _ in pairs}, {d for _, d in pairs}
        return (pairs, [t for t in track_ids if t not in used_t],
                [d for d in det_ids if d not in used_d])

    @staticmethod
    def _move(track: _Track, box: np.ndarray) -> None:
        step = (box - track.box) / (track.lost + 1)
        track.velocity = 0.7 * track.velocity + 0.3 * step  # smoothed per-frame motion
        track.box = box.copy()
        track.lost = 0
