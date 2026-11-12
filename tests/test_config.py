from idcard import config
from idcard.config import Settings


def test_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    s = Settings(band_color="#112233", tolerance=40.0, camera="clip.avi", device="cpu")
    config.save(s, path)
    assert config.load(path) == s


def test_missing_and_corrupt_file_give_defaults(tmp_path):
    assert config.load(tmp_path / "nope.json") == Settings()
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert config.load(bad) == Settings()


def test_form_updates_and_clamps():
    s = Settings()
    config.update_from_form(s, {"band_color": "#ABCDEF", "tolerance": "500",
                                "person_conf": "0", "device": "cuda", "camera": " 1 ",
                                "unknown": "x"})
    assert s.band_color == "#abcdef"
    assert s.tolerance == 80
    assert s.person_conf == 0.05
    assert s.device == "cuda"
    assert s.camera == "1"


def test_bad_values_keep_old():
    s = Settings()
    config.update_from_form(s, {"band_color": "blue", "tolerance": "abc",
                                "violation_seconds": "nan", "device": "tpu", "camera": "  "})
    assert s == Settings()


def test_on_threshold_stays_above_off():
    s = Settings()
    config.update_from_form(s, {"on_threshold": "0.2", "off_threshold": "0.5"})
    assert (s.on_threshold, s.off_threshold) == (0.6, 0.35)
    config.update_from_form(s, {"on_threshold": "0.8", "off_threshold": "0.3"})
    assert (s.on_threshold, s.off_threshold) == (0.8, 0.3)
    config.update_from_form(s, {"on_threshold": "0.1"})  # would drop below off_threshold
    assert s.on_threshold == 0.8
