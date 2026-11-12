import socket
import threading
import time
from dataclasses import dataclass

import cv2
import httpx
import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient

from idcard import config
from idcard.server import create_app

SETTINGS_HTML = """<html><body><form method="post" action="/api/settings">
<input type="color" name="band_color" value="#1e40af">
<input type="range" name="tolerance" min="5" max="80" value="25">
<input type="number" name="on_threshold" step="0.05" value="0.6">
<input type="number" name="off_threshold" step="0.05" value="0.35">
<input type="number" name="violation_seconds" value="3">
<input type="number" name="person_conf" value="0.4">
<input type="text" name="camera" value="0">
<select name="device"><option value="auto" selected>auto</option><option value="cpu">cpu</option>
<option value="cuda">cuda</option></select></form></body></html>"""


@dataclass
class Person:
    track_id: int
    box: tuple
    state: str
    score: float
    new_violation: bool


class FakePipeline:
    backend_name = "FakeProvider"

    def __init__(self, settings):
        self.settings = settings
        self.calls = 0
        self.reloads = 0

    def process(self, frame, now):
        self.calls += 1
        return [Person(1, (2, 2, 30, 40), "WEARING", 0.9, False),
                Person(2, (40, 5, 90, 60), "NOT_WEARING", 0.1, self.calls == 1)]

    def annotate(self, frame, people):
        out = frame.copy()
        for p in people:
            x1, y1, x2, y2 = p.box
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return out

    def reset(self):
        pass

    def reload(self):
        self.reloads += 1


def make_video(path, frames=20):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (160, 120))
    for i in range(frames):
        w.write(np.full((120, 160, 3), i * 10 % 255, np.uint8))
    w.release()


@pytest.fixture
def root(tmp_path):
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "settings.html").write_text(SETTINGS_HTML)
    (tmp_path / "web" / "index.html").write_text("<html>home</html>")
    (tmp_path / "web" / "style.css").write_text("body{}")
    return tmp_path


@pytest.fixture
def app(root):
    app = create_app(root, FakePipeline)
    yield app
    app.state.camera.stop()


@pytest.fixture
def client(app):
    return TestClient(app)


def test_static_and_index(client):
    assert "home" in client.get("/").text
    assert client.get("/style.css").status_code == 200


def test_settings_form_then_page(client, root):
    r = client.post("/api/settings", data={"band_color": "#ff0000", "tolerance": "40",
                                           "camera": "my<clip>.mp4", "device": "cpu"},
                    follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/settings.html"
    assert config.load(root / "settings.json").band_color == "#ff0000"
    text = client.get("/settings.html").text
    assert 'name="band_color" value="#ff0000"' in text
    assert 'name="tolerance" min="5" max="80" value="40.0"' in text
    assert 'value="my&lt;clip&gt;.mp4"' in text and "my<clip>" not in text
    assert '<option value="cpu" selected>' in text
    assert text.count("selected") == 1


def test_device_change_reloads_pipeline(app, client):
    with app.state.camera.pipeline_lock:
        pipe = app.state.camera.get_pipeline()
    client.post("/api/settings", data={"device": "cuda"})
    assert pipe.reloads == 1
    client.post("/api/settings", data={"device": "cuda"})
    assert pipe.reloads == 1


def test_violation_snapshot_and_history(app, client, root):
    cam = app.state.camera
    frame = np.full((100, 100, 3), 120, np.uint8)
    cam.settings.camera = "0"
    annotated, counts = cam._process(frame)
    assert counts == {"wearing": 1, "not_wearing": 1, "unsure": 0}
    files = list((root / "data" / "snapshots").glob("*.jpg"))
    assert len(files) == 1
    rows = app.state.store.list()
    assert len(rows) == 1 and rows[0]["track_id"] == 2 and rows[0]["snapshot"] == files[0].name

    frag = client.get("/fragments/history").text
    assert files[0].name in frag and "/style.css" in frag
    assert client.get("/snapshots/" + files[0].name).status_code == 200
    assert files[0].name in client.get("/history.csv").text
    assert "refresh" in client.get("/fragments/stats").text

    r = client.post("/api/history/clear", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/history.html"
    assert app.state.store.list() == [] and not list((root / "data" / "snapshots").glob("*"))
    assert "No violations" in client.get("/fragments/history").text


def test_image_upload(client, root):
    ok, buf = cv2.imencode(".png", np.full((60, 80, 3), 90, np.uint8))
    r = client.post("/api/upload", files={"file": ("a.png", buf.tobytes(), "image/png")})
    assert r.status_code == 200 and "<img" in r.text and "/upload.html" in r.text
    results = list((root / "data" / "uploads").glob("*-result.jpg"))
    assert len(results) == 1
    assert client.get("/uploads/" + results[0].name).status_code == 200


def test_upload_rejects_bad_files(client):
    r = client.post("/api/upload", files={"file": ("a.exe", b"x", "application/octet-stream")})
    assert "Unsupported" in r.text
    r = client.post("/api/upload", files={"file": ("a.png", b"not an image", "image/png")})
    assert "Could not read" in r.text


def test_video_upload(client, root, tmp_path):
    make_video(tmp_path / "v.avi")
    data = (tmp_path / "v.avi").read_bytes()
    r = client.post("/api/upload", files={"file": ("v.avi", data, "video/x-msvideo")})
    assert "download" in r.text
    assert len(list((root / "data" / "uploads").glob("*-result.mp4"))) == 1


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_stream_from_video_file(root, tmp_path):
    make_video(root / "clip.avi")
    app = create_app(root, FakePipeline, source="clip.avi")
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        jpegs = 0
        buf = b""
        with httpx.stream("GET", f"http://127.0.0.1:{port}/stream", timeout=10) as r:
            assert r.headers["content-type"].startswith("multipart/x-mixed-replace")
            for chunk in r.iter_bytes():
                buf += chunk
                jpegs = buf.count(b"Content-Type: image/jpeg")
                if jpegs >= 5:
                    break
        assert jpegs >= 5
        assert app.state.camera.stats()["wearing"] == 1  # video decoded and processed
        assert app.state.camera.stats()["backend"] == "FakeProvider"
        # camera is released a moment after the viewer leaves
        app.state.camera.grace = 0.2
        for _ in range(60):
            if app.state.camera.viewers == 0:
                break
            time.sleep(0.1)
        assert app.state.camera.viewers == 0
    finally:
        server.should_exit = True
        thread.join(5)
        app.state.camera.stop()


def test_unavailable_source_streams_placeholder(root):
    app = create_app(root, FakePipeline, source="missing.avi")
    cam = app.state.camera
    try:
        cam.add_viewer()
        seq, jpeg = cam.wait_frame(-1, 1)
        for _ in range(30):
            if cam.stats()["fps"] == 0.0 and cam._seq > 1:
                break
            time.sleep(0.1)
        seq, jpeg = cam.wait_frame(seq, 3)
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert img is not None and img.shape[:2] == (480, 640)
        assert cam.pipeline is None  # never ran the pipeline
    finally:
        cam.remove_viewer()
        cam.stop()
