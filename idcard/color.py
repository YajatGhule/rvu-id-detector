"""Lanyard band colour: per-pixel colour matching and a "does this look like a strap" score.

Colours are compared in CIELAB, where L is lightness (0..100) and a/b are the colour
(green-red and blue-yellow axes). Straight Euclidean distance in Lab ("Delta E") breaks under
shadows: a blue strap in shadow gets darker AND less saturated, so it drifts far from the
target. Two fixes, both strongest for colourful targets and fading out for grey ones:
  1. lightness differences count only 25% (weight 0.25 instead of 1),
  2. the pixel's a/b are scaled by (target L / pixel L) before comparing, which roughly undoes
     the loss of saturation in shadow (and the gain in bright light).
For grey, black or white bands the colour axes carry no information, so lightness is all we
can compare and both fixes are switched off. That makes dark bands much less reliable.
"""
from dataclasses import dataclass

import cv2
import numpy as np

WORK_HEIGHT = 128          # torso crops are resized to this height so rules are size independent
SHIRT_FRACTION = 0.45      # this much of the torso in band colour -> cannot judge (see below)
FULL_STRAP_LENGTH = 0.8    # total strap length (in torso heights) that gives score 1.0
MAX_STRAPS = 4             # more strap-like pieces than this = striped clothing, not a lanyard


@dataclass
class ColorResult:
    score: float | None    # 0..1 lanyard evidence, None = cannot judge from colour
    match_fraction: float  # share of torso pixels that match the band colour
    strap_length: float    # total length of strap-like regions, in torso heights
    reason: str = ""


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """'#1e40af' or '1e40af' -> (175, 64, 30) in OpenCV's BGR order."""
    value = hex_color.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        raise ValueError(f"not a hex colour: {hex_color!r}")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r


def to_lab(image_bgr: np.ndarray) -> np.ndarray:
    """uint8 BGR image -> float32 CIELAB (L 0..100, a/b about -128..127)."""
    return cv2.cvtColor(image_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def color_distance(image_bgr: np.ndarray, hex_color: str) -> np.ndarray:
    """Per-pixel perceptual distance to the target colour (shadow tolerant, see top)."""
    target = to_lab(np.array([[hex_to_bgr(hex_color)]], np.uint8))[0, 0]
    target_l, target_ab = target[0], target[1:]
    # 0 for grey/black/white targets, 1 for clearly coloured ones
    colourfulness = float(np.clip(np.hypot(*target_ab) / 30.0, 0.0, 1.0))
    light_weight = 1.0 - 0.75 * colourfulness

    lab = to_lab(image_bgr)
    pixel_l, pixel_ab = lab[..., 0], lab[..., 1:]
    shade = np.clip(target_l / np.maximum(pixel_l, 1.0), 0.5, 2.5)
    shade = 1.0 + colourfulness * (shade - 1.0)
    d_ab = pixel_ab * shade[..., None] - target_ab
    d_l = light_weight * (pixel_l - target_l)
    return np.sqrt(d_l ** 2 + (d_ab ** 2).sum(axis=-1))


def match_mask(image_bgr: np.ndarray, hex_color: str, tolerance: float) -> np.ndarray:
    """uint8 mask, 255 where the pixel is within `tolerance` of the band colour."""
    return np.where(color_distance(image_bgr, hex_color) <= tolerance, 255, 0).astype(np.uint8)


def lanyard_score(torso_bgr: np.ndarray, hex_color: str, tolerance: float) -> ColorResult:
    """How much the torso crop looks like it has a band-coloured strap hanging on it.

    We reward thin, long, roughly vertical regions of band colour that start near the neck,
    and ignore wide blobs. A shirt or jacket in the band colour is one big blob, so it adds
    nothing. If most of the torso is band coloured we return score=None (UNSURE) rather than
    a low score: the lanyard would be invisible against such a shirt, so a low score would
    log innocent people as violations. A bit of missed enforcement is the lesser evil.
    """
    if torso_bgr.size == 0:
        return ColorResult(None, 0.0, 0.0, "empty crop")
    h, w = torso_bgr.shape[:2]
    work_w = max(8, round(w * WORK_HEIGHT / h))
    crop = cv2.resize(torso_bgr, (work_w, WORK_HEIGHT), interpolation=cv2.INTER_AREA)
    crop = cv2.GaussianBlur(crop, (3, 3), 0)

    mask = match_mask(crop, hex_color, tolerance)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # drop speckles
    fraction = float(np.count_nonzero(mask)) / mask.size
    if fraction >= SHIRT_FRACTION:
        return ColorResult(None, fraction, 0.0, "torso mostly band colour")

    straps = _strap_lengths(mask)  # lengths in pixels of the working image
    strap_length = float(sum(straps)) / WORK_HEIGHT
    if len(straps) > MAX_STRAPS:
        # A lanyard shows as one or two straps (maybe split by a hand or hair). Many parallel
        # straps mean a striped shirt in the band colour: as with a plain one, cannot judge.
        return ColorResult(None, fraction, strap_length, "striped clothing")
    score = float(np.clip(strap_length / FULL_STRAP_LENGTH, 0.0, 1.0))
    return ColorResult(score, fraction, strap_length)


def _strap_lengths(mask: np.ndarray) -> list[float]:
    """Length (pixels) of each connected region of the mask that is shaped like a strap."""
    height, width = mask.shape
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)  # distance to nearest non-match
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    lengths = []
    for i in range(1, count):  # label 0 is the background
        x, y, bw, bh, area = stats[i]
        if area < 15:
            continue
        pixels = labels == i
        # For a band of width t, distances run evenly from ~0 to t/2, so their mean is ~t/4.
        thickness = max(1.0, 4.0 * float(distance[pixels].mean()))
        length = area / thickness
        if thickness > 0.15 * width:          # too wide: shirt, jacket, bag, ...
            continue
        if length < 4.0 * thickness:          # not elongated: a blob or a logo
            continue
        if length < 0.15 * height:            # too short to be a strap
            continue
        if y > 0.35 * height:                 # straps come down from the neck
            continue
        if _angle_from_vertical(pixels) > 60:  # horizontal stripes on clothing
            continue
        lengths.append(length)
    return lengths


def _angle_from_vertical(pixels: np.ndarray) -> float:
    """Angle in degrees between the region's main axis and the vertical (0 = upright)."""
    m = cv2.moments(pixels.astype(np.uint8), binaryImage=True)
    angle = 0.5 * np.degrees(np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"]))  # from x axis
    return abs(90.0 - abs(angle))
