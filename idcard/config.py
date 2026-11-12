"""Settings: a small dataclass saved as JSON, with validation for web form input."""

import json
import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

DEVICES = ("auto", "cpu", "cuda", "directml", "openvino", "coreml")
_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


@dataclass
class Settings:
    band_color: str = "#1e40af"  # hex, lanyard band colour
    tolerance: float = 25.0  # colour distance tolerance, 5..80
    on_threshold: float = 0.6  # smoothed score >= this -> WEARING
    off_threshold: float = 0.35  # smoothed score <= this -> NOT_WEARING
    violation_seconds: float = 3.0  # continuous NOT_WEARING before logging
    person_conf: float = 0.4  # person detection confidence
    camera: str = "0"  # camera index, or a video file path
    device: str = "auto"  # auto|cpu|cuda|directml|openvino|coreml


def _number(value, low, high):
    """Parse value as a finite float clamped to [low, high]; None if it is not a number."""
    try:
        number = float(str(value).strip())
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return min(max(number, low), high)


def load(path) -> Settings:
    """Read settings from a JSON file. Missing or corrupt file gives the defaults."""
    settings = Settings()
    try:
        data = json.loads(Path(path).read_text())
        if isinstance(data, dict):
            # Reuse the form validation so a hand-edited file cannot hold bad values.
            update_from_form(settings, {k: str(v) for k, v in data.items()})
    except (OSError, ValueError):
        pass
    return settings


def save(settings: Settings, path) -> None:
    Path(path).write_text(json.dumps(asdict(settings), indent=2) + "\n")


def update_from_form(settings: Settings, form: dict) -> None:
    """Apply form fields to settings. Invalid values keep the old value; unknown keys are ignored."""
    known = {f.name for f in fields(Settings)}
    form = {k: v for k, v in form.items() if k in known}

    color = str(form.get("band_color", "")).strip()
    if _HEX_COLOR.match(color):
        settings.band_color = color.lower()

    for name, low, high in [
        ("tolerance", 5, 80),
        ("violation_seconds", 0, 600),
        ("person_conf", 0.05, 0.95),
    ]:
        if name in form:
            number = _number(form[name], low, high)
            if number is not None:
                setattr(settings, name, number)

    # on_threshold must stay above off_threshold, otherwise keep both old values.
    new_on = _number(form["on_threshold"], 0, 1) if "on_threshold" in form else None
    new_off = _number(form["off_threshold"], 0, 1) if "off_threshold" in form else None
    on = settings.on_threshold if new_on is None else new_on
    off = settings.off_threshold if new_off is None else new_off
    if on > off:
        settings.on_threshold, settings.off_threshold = on, off

    camera = str(form.get("camera", "")).strip()
    if camera and len(camera) <= 500:
        settings.camera = camera

    device = str(form.get("device", "")).strip().lower()
    if device in DEVICES:
        settings.device = device
