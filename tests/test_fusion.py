from types import SimpleNamespace

from idcard.fusion import MIN_VALID, NOT_WEARING, UNSURE, WEARING, Fusion, combine_scores

S = SimpleNamespace(on_threshold=0.6, off_threshold=0.35, violation_seconds=3.0)


def feed(fusion, scores, start=0.0, step=0.1, track=1):
    out = None
    for i, s in enumerate(scores):
        out = fusion.update(track, s, start + i * step, S)
    return out


def test_unsure_until_enough_valid_scores():
    f = Fusion()
    assert feed(f, [0.9] * (MIN_VALID - 1))[0] == UNSURE
    assert feed(f, [0.9])[0] == WEARING


def test_unjudgeable_frames_give_unsure():
    f = Fusion()
    assert feed(f, [None] * 20)[0] == UNSURE


def test_states_and_hysteresis():
    f = Fusion()
    assert feed(f, [0.9] * 15)[0] == WEARING
    # average drifts into the dead band (0.35..0.6): stays WEARING
    state, smoothed, _ = feed(f, [0.3] * 8)
    assert 0.35 < smoothed < 0.6 and state == WEARING
    assert feed(f, [0.0] * 15)[0] == NOT_WEARING
    state, smoothed, _ = feed(f, [0.9] * 6)
    assert 0.35 < smoothed < 0.6 and state == NOT_WEARING


def test_single_outlier_does_not_flip():
    f = Fusion()
    feed(f, [0.9] * 15)
    assert feed(f, [0.0])[0] == WEARING


def test_violation_once_after_violation_seconds():
    f = Fusion()
    flags = []
    for i in range(100):  # 10 s at 10 fps of "no lanyard"
        state, _, new = f.update(7, 0.0, i * 0.1, S)
        flags.append(new)
    assert flags.count(True) == 1
    first_bad = MIN_VALID - 1  # frame where the state became NOT_WEARING
    fired = flags.index(True)
    assert abs((fired - first_bad) * 0.1 - S.violation_seconds) < 0.11


def test_violation_needs_continuous_not_wearing():
    f = Fusion()
    feed(f, [0.0] * 20, start=0.0)          # 2 s not wearing
    feed(f, [None] * 15, start=2.0)          # turns away: valid scores remain -> still decided
    f2 = Fusion()
    t = 0.0
    for _ in range(3):                       # 2 s bad, then 2 s good, repeated
        for s in [0.0] * 20 + [1.0] * 40:
            assert not f2.update(1, s, t, S)[2]
            t += 0.1


def test_settings_change_applies_immediately():
    f = Fusion()
    feed(f, [0.5] * 15)
    assert f.tracks[1].state == UNSURE
    strict = SimpleNamespace(on_threshold=0.45, off_threshold=0.3, violation_seconds=3.0)
    assert f.update(1, 0.5, 2.0, strict)[0] == WEARING


def test_tracks_are_independent_and_forgotten():
    f = Fusion()
    feed(f, [0.9] * 5, track=1)
    feed(f, [0.0] * 5, track=2)
    assert f.tracks[1].state == WEARING and f.tracks[2].state == NOT_WEARING
    f.update(2, 0.0, 20.0, S)
    f.forget_old(now=20.0)
    assert list(f.tracks) == [2]
    f.reset()
    assert f.tracks == {}


def test_combine_scores_hook():
    assert combine_scores(0.7) == 0.7
    assert combine_scores(None) is None
    assert combine_scores(None, 0.9) == 0.9
    assert abs(combine_scores(0.5, 1.0) - 0.8) < 1e-9
