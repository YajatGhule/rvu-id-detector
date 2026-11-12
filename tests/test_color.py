import cv2
import numpy as np
import pytest

from idcard.color import hex_to_bgr, lanyard_score, match_mask

BLUE = "#1e40af"
SHIRT = (200, 200, 200)  # light grey, BGR


def torso(shirt=SHIRT, w=120, h=150):
    return np.full((h, w, 3), shirt, np.uint8)


def draw_v_lanyard(img, colour_bgr, thickness=5):
    """Two straps from the neck corners down to a card in the middle of the chest."""
    h, w = img.shape[:2]
    cv2.line(img, (int(w * 0.3), 0), (w // 2, int(h * 0.7)), colour_bgr, thickness)
    cv2.line(img, (int(w * 0.7), 0), (w // 2, int(h * 0.7)), colour_bgr, thickness)
    return img


def test_hex_to_bgr():
    assert hex_to_bgr("#1e40af") == (0xAF, 0x40, 0x1E)
    assert hex_to_bgr("fff") == (255, 255, 255)
    with pytest.raises(ValueError):
        hex_to_bgr("#12345")


def test_match_mask_exact_and_far():
    img = np.zeros((2, 2, 3), np.uint8)
    img[0, 0] = hex_to_bgr(BLUE)
    img[0, 1] = (0, 0, 255)       # red
    img[1, 0] = (200, 200, 200)   # grey
    img[1, 1] = (0, 255, 0)       # green
    mask = match_mask(img, BLUE, 25)
    assert mask[0, 0] == 255
    assert mask[0, 1] == mask[1, 0] == mask[1, 1] == 0


def test_match_mask_tolerates_shadow_and_light():
    b, g, r = hex_to_bgr(BLUE)
    shades = [np.clip(np.array([b, g, r]) * f, 0, 255) for f in (0.55, 0.75, 1.25)]
    img = np.array([shades], np.uint8)
    assert (match_mask(img, BLUE, 25) == 255).all()


def test_match_mask_bigger_tolerance_matches_more():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (50, 50, 3), dtype=np.uint8)
    counts = [np.count_nonzero(match_mask(img, BLUE, t)) for t in (10, 25, 50, 80)]
    assert counts == sorted(counts) and counts[-1] > counts[0]


def test_grey_target_uses_lightness():
    img = np.array([[(20, 20, 20), (128, 128, 128), (235, 235, 235)]], np.uint8)
    mask = match_mask(img, "#141414", 25)  # near-black band
    assert list(mask[0]) == [255, 0, 0]


def test_thin_strap_scores_high():
    result = lanyard_score(draw_v_lanyard(torso(), hex_to_bgr(BLUE)), BLUE, 25)
    assert result.score is not None and result.score >= 0.8


def test_strap_in_shadow_still_scores():
    img = draw_v_lanyard(torso(), hex_to_bgr(BLUE))
    img[:, : img.shape[1] // 2] = (img[:, : img.shape[1] // 2] * 0.6).astype(np.uint8)
    assert lanyard_score(img, BLUE, 25).score >= 0.6


def test_no_strap_scores_zero():
    result = lanyard_score(torso(), BLUE, 25)
    assert result.score == 0.0


def test_shirt_in_band_colour_is_unsure_not_low():
    result = lanyard_score(torso(shirt=hex_to_bgr(BLUE)), BLUE, 25)
    assert result.score is None
    assert result.match_fraction > 0.9


def test_wide_blob_and_horizontal_stripes_score_low():
    img = torso()
    cv2.rectangle(img, (30, 40), (90, 90), hex_to_bgr(BLUE), -1)        # a big logo
    for y in (20, 60, 100, 140):                                          # striped shirt
        cv2.line(img, (0, y), (119, y), hex_to_bgr(BLUE), 5)
    assert lanyard_score(img, BLUE, 25).score <= 0.2


def test_strap_must_come_from_the_neck():
    img = torso()
    cv2.line(img, (60, 90), (60, 149), hex_to_bgr(BLUE), 5)  # only low on the torso
    assert lanyard_score(img, BLUE, 25).score == 0.0


def test_black_strap_on_light_shirt_and_on_black_shirt():
    black = "#151515"
    on_light = draw_v_lanyard(torso(), hex_to_bgr(black))
    assert lanyard_score(on_light, black, 25).score >= 0.8
    on_black = draw_v_lanyard(torso(shirt=(25, 25, 25)), hex_to_bgr(black))
    assert lanyard_score(on_black, black, 25).score is None


def test_small_torso_crop_still_works():
    small = cv2.resize(draw_v_lanyard(torso(), hex_to_bgr(BLUE), 6), (40, 50),
                       interpolation=cv2.INTER_AREA)
    assert lanyard_score(small, BLUE, 25).score >= 0.6


def test_empty_crop():
    assert lanyard_score(np.zeros((0, 0, 3), np.uint8), BLUE, 25).score is None


def test_vertical_striped_shirt_is_unsure():
    img = torso()
    for x in range(10, 120, 20):
        cv2.line(img, (x, 0), (x, 149), hex_to_bgr(BLUE), 5)
    assert lanyard_score(img, BLUE, 25).score is None
