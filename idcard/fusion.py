"""Turn noisy per-frame scores into a steady state per person.

Each tracked person keeps the last WINDOW observations. An observation is a score 0..1, or
None when that frame could not be judged (facing away, too small, ...). The average of the
valid scores is compared with two thresholds (hysteresis), so a score hovering around one
value does not make the box flicker:
    average >= on_threshold   -> WEARING
    average <= off_threshold  -> NOT_WEARING
    in between                -> keep the previous state
With fewer than MIN_VALID valid scores in the window the person is UNSURE.
"""
from collections import deque
from dataclasses import dataclass, field

WEARING, NOT_WEARING, UNSURE = "WEARING", "NOT_WEARING", "UNSURE"
WINDOW = 15
MIN_VALID = 3
FORGET_AFTER = 5.0  # seconds without seeing a track before its state is dropped


def combine_scores(color_score: float | None, card_score: float | None = None) -> float | None:
    """Fuse the evidence for one frame into one score (None = cannot judge).

    Phase 1 has only the colour score. HOOK: once the trained lanyard/ID-card detector exists,
    pass its best confidence on the torso as `card_score`. Plan: a confident card detection
    counts on its own, the colour check confirms it is OUR college's band:
        if card_score is None: return color_score
        if color_score is None: return card_score
        return 0.6 * card_score + 0.4 * color_score
    """
    if card_score is None:
        return color_score
    if color_score is None:
        return card_score
    return 0.6 * card_score + 0.4 * color_score


@dataclass
class TrackState:
    scores: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    state: str = UNSURE
    smoothed: float = 0.0
    not_wearing_since: float | None = None
    violation_reported: bool = False
    last_seen: float = 0.0


class Fusion:
    def __init__(self) -> None:
        self.tracks: dict[int, TrackState] = {}

    def reset(self) -> None:
        self.tracks.clear()

    def update(self, track_id: int, score: float | None, now: float,
               settings) -> tuple[str, float, bool]:
        """Add one observation. Returns (state, smoothed score, new_violation)."""
        track = self.tracks.setdefault(track_id, TrackState())
        track.last_seen = now
        track.scores.append(score)

        valid = [s for s in track.scores if s is not None]
        if len(valid) < MIN_VALID:
            track.state = UNSURE
        else:
            track.smoothed = sum(valid) / len(valid)
            if track.smoothed >= settings.on_threshold:
                track.state = WEARING
            elif track.smoothed <= settings.off_threshold:
                track.state = NOT_WEARING
            elif track.state == UNSURE:
                pass  # in the dead band with no earlier decision: stay UNSURE

        new_violation = False
        if track.state == NOT_WEARING:
            if track.not_wearing_since is None:
                track.not_wearing_since = now
            lasted = now - track.not_wearing_since
            if lasted >= settings.violation_seconds and not track.violation_reported:
                track.violation_reported = True  # only once per track
                new_violation = True
        else:
            track.not_wearing_since = None
        return track.state, track.smoothed, new_violation

    def forget_old(self, now: float, after: float = FORGET_AFTER) -> None:
        """Drop tracks that have not been seen for `after` seconds."""
        for track_id in [t for t, s in self.tracks.items() if now - s.last_seen > after]:
            del self.tracks[track_id]
