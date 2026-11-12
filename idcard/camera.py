"""Background worker: owns the video source and the pipeline, keeps the latest JPEG frame.

The source is only open while someone is watching /stream (plus a short grace period).
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("idcard.camera")

GRACE_SECONDS = 5.0  # keep the camera open this long after the last viewer leaves
RETRY_SECONDS = 3.0  # wait this long before trying to open a failed source again


def unavailable_frame(text="Camera unavailable") -> np.ndarray:
    """A plain image shown in the stream when the source cannot be opened."""
    img = np.full((480, 640, 3), 40, np.uint8)
    cv2.putText(img, text, (130, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (230, 230, 230), 2)
    cv2.putText(img, "Check Settings > Camera. Retrying...", (130, 275),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1)
    return img


def encode_jpeg(img) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b""


class CameraWorker(threading.Thread):
    def __init__(self, settings, pipeline_factory, store, snapshots_dir, root=None,
                 grace=GRACE_SECONDS, retry=RETRY_SECONDS):
        super().__init__(daemon=True, name="camera-worker")
        self.settings = settings
        self._factory = pipeline_factory
        self.store = store
        self.snapshots_dir = Path(snapshots_dir)
        self.root = Path(root) if root else Path.cwd()
        self.grace = grace
        self.retry = retry

        self.pipeline = None
        self.pipeline_lock = threading.RLock()  # guards every use of self.pipeline

        self._cond = threading.Condition()  # guards viewers, the frame and the stats below
        self._viewers = 0
        self._stop_flag = False
        self._jpeg = encode_jpeg(unavailable_frame("Starting..."))
        self._seq = 0
        self._counts = {"wearing": 0, "not_wearing": 0, "unsure": 0}
        self._fps = 0.0

    # ---- used by request handlers -------------------------------------------------

    def add_viewer(self):
        with self._cond:
            self._viewers += 1
            self._cond.notify_all()

    def remove_viewer(self):
        with self._cond:
            self._viewers = max(0, self._viewers - 1)
            self._cond.notify_all()

    @property
    def viewers(self) -> int:
        with self._cond:
            return self._viewers

    def wait_frame(self, last_seq, timeout=1.0):
        """Block until a frame newer than last_seq exists (or timeout). Returns (seq, jpeg)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq or self._stop_flag, timeout)
            return self._seq, self._jpeg

    def stats(self) -> dict:
        with self._cond:
            backend = getattr(self.pipeline, "backend_name", "not loaded yet")
            return {**self._counts, "fps": self._fps, "backend": backend}

    def get_pipeline(self):
        """Create the pipeline on first use. Call with pipeline_lock held."""
        if self.pipeline is None:
            self.pipeline = self._factory(self.settings)
        return self.pipeline

    def stop(self):
        with self._cond:
            self._stop_flag = True
            self._cond.notify_all()

    # ---- internals ----------------------------------------------------------------

    def _publish(self, img, counts=None, fps=None):
        jpeg = encode_jpeg(img)
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            if counts is not None:
                self._counts = counts
            if fps is not None:
                self._fps = fps
            self._cond.notify_all()

    def _open(self, source):
        """Returns (capture, is_file, frame_period) or None."""
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
            is_file = False
        else:
            path = Path(source).expanduser()
            if not path.is_absolute():
                path = self.root / path
            cap = cv2.VideoCapture(str(path))
            is_file = True
        if not cap.isOpened():
            cap.release()
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 0
        period = 1.0 / fps if fps and 1 < fps < 240 else (1 / 25 if is_file else 0)
        return cap, is_file, period

    def _save_violation(self, frame, person):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in person.box)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame
        name = f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-id{int(person.track_id)}.jpg"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.snapshots_dir / name), crop)
        self.store.add(person.track_id, person.score, name)

    def _process(self, frame):
        with self.pipeline_lock:
            pipeline = self.get_pipeline()
            people = pipeline.process(frame, time.monotonic())
            annotated = pipeline.annotate(frame, people)
        for person in people:
            if person.new_violation:
                self._save_violation(frame, person)
        counts = {"wearing": 0, "not_wearing": 0, "unsure": 0}
        key = {"WEARING": "wearing", "NOT_WEARING": "not_wearing"}
        for person in people:
            counts[key.get(person.state, "unsure")] += 1
        return annotated, counts

    def run(self):
        cap = None
        opened = None
        is_file = False
        period = 0.0
        last_viewer_time = -1e9
        next_try = 0.0
        frames, window_start = 0, time.monotonic()
        fps = 0.0

        while True:
            with self._cond:
                if self._stop_flag:
                    break
                viewers = self._viewers
            now = time.monotonic()
            if viewers:
                last_viewer_time = now

            if not viewers and now - last_viewer_time >= self.grace:
                # Nobody is watching: release the source and sleep until someone shows up.
                if cap is not None:
                    cap.release()
                    cap = opened = None
                    self._publish(unavailable_frame("Idle"), {"wearing": 0, "not_wearing": 0,
                                                              "unsure": 0}, 0.0)
                with self._cond:
                    self._cond.wait_for(lambda: self._viewers or self._stop_flag, 0.5)
                continue

            source = str(self.settings.camera).strip()
            if cap is not None and source != opened:
                cap.release()  # the camera setting changed
                cap = opened = None
                with self.pipeline_lock:
                    if self.pipeline is not None:
                        self.pipeline.reset()

            if cap is None:
                if now < next_try:
                    time.sleep(0.1)
                    continue
                try:
                    result = self._open(source)
                except Exception:
                    log.exception("opening %r failed", source)
                    result = None
                if result is None:
                    next_try = now + self.retry
                    self._publish(unavailable_frame(), {"wearing": 0, "not_wearing": 0,
                                                        "unsure": 0}, 0.0)
                    continue
                cap, is_file, period = result
                opened = source
                with self.pipeline_lock:
                    if self.pipeline is not None:
                        self.pipeline.reset()

            started = time.monotonic()
            ok, frame = cap.read()
            if not ok and is_file:  # end of the video: loop it
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = opened = None
                next_try = time.monotonic() + self.retry
                self._publish(unavailable_frame("Camera lost"))
                continue

            try:
                annotated, counts = self._process(frame)
            except Exception:
                log.exception("pipeline failed on a frame")
                annotated, counts = frame, None

            frames += 1
            if time.monotonic() - window_start >= 1.0:
                fps = frames / (time.monotonic() - window_start)
                frames, window_start = 0, time.monotonic()
            self._publish(annotated, counts, fps)

            if period:  # pace video files at their own frame rate
                time.sleep(max(0.0, period - (time.monotonic() - started)))

        if cap is not None:
            cap.release()
