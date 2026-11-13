"""FastAPI app: MJPEG stream, HTML fragments, form handlers, uploads, static files."""

import asyncio
import html
import re
import threading
import uuid
from pathlib import Path
from urllib.parse import quote

import cv2
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from idcard import config, store
from idcard.camera import CameraWorker

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_VIDEO_FRAMES = 5000
SETTING_INPUTS = ["band_color", "tolerance", "on_threshold", "off_threshold",
                  "violation_seconds", "person_conf", "camera"]


def _default_factory(settings):
    from idcard.pipeline import Pipeline  # imported lazily so this module loads without it

    return Pipeline(settings)


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def page(title, body, refresh=False) -> HTMLResponse:
    meta = '<meta http-equiv="refresh" content="1">' if refresh else ""
    return HTMLResponse(
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">{meta}'
        f'<title>{esc(title)}</title><link rel="stylesheet" href="/style.css"></head>'
        f"<body>{body}</body></html>"
    )


def fill_settings_html(text: str, settings) -> str:
    """Write the current settings into value="..." of each input and select the device option."""
    for name in SETTING_INPUTS:
        value = esc(getattr(settings, name))

        def fix_tag(match, value=value):
            tag = match.group(0)
            if re.search(r'\svalue="[^"]*"', tag):
                return re.sub(r'(\svalue=")[^"]*(")', lambda m: m.group(1) + value + m.group(2),
                              tag, count=1)
            return re.sub(r"\s*/?>$", lambda m: f' value="{value}"' + m.group(0), tag)

        text = re.sub(rf'<input\b[^>]*\bname="{name}"[^>]*>', fix_tag, text)

    def fix_select(match):
        block = match.group(0)
        block = re.sub(r"\sselected(?:=\"[^\"]*\")?", "", block)
        return re.sub(rf'(<option\b[^>]*\bvalue="{re.escape(settings.device)}")',
                      r"\1 selected", block, count=1)

    return re.sub(r'<select\b[^>]*\bname="device"[^>]*>.*?</select>', fix_select, text,
                  flags=re.S)


def create_app(root: Path = REPO_ROOT, pipeline_factory=None, source=None) -> FastAPI:
    """`source` (optional) overrides settings.camera for this run, e.g. from the CLI."""
    root = Path(root)
    factory = pipeline_factory or _default_factory
    data_dir = root / "data"
    snapshots_dir = data_dir / "snapshots"
    uploads_dir = data_dir / "uploads"
    for folder in (snapshots_dir, uploads_dir):
        folder.mkdir(parents=True, exist_ok=True)

    settings_path = root / "settings.json"
    settings = config.load(settings_path)
    if source:
        settings.camera = str(source)

    log_store = store.Store(data_dir / "idcard.db")
    camera = CameraWorker(settings, factory, log_store, snapshots_dir, root=root)
    camera.start()

    # Uploads use their own pipeline instance so they never interrupt the live stream.
    upload_lock = threading.Lock()
    upload_pipeline = {}

    async def lifespan(app):
        yield
        camera.stop()

    app = FastAPI(title="RVU ID card detector", lifespan=lifespan)
    app.state.settings = settings
    app.state.camera = camera
    app.state.store = log_store

    # ---- live stream and fragments ---------------------------------------------

    @app.get("/stream")
    async def stream(request: Request):
        async def frames():
            camera.add_viewer()
            seq = -1
            try:
                while not await request.is_disconnected():
                    seq, jpeg = await asyncio.to_thread(camera.wait_frame, seq, 1.0)
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            finally:
                camera.remove_viewer()

        return StreamingResponse(frames(),
                                 media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    @app.get("/fragments/stats")
    def stats_fragment():
        s = camera.stats()
        body = (f'<p class="stats">Wearing: <b>{s["wearing"]}</b> | '
                f'Not wearing: <b>{s["not_wearing"]}</b> | Unsure: <b>{s["unsure"]}</b> | '
                f'FPS: {s["fps"]:.1f} | Backend: {esc(s["backend"])}</p>')
        return page("Stats", body, refresh=True)

    @app.get("/fragments/history")
    def history_fragment():
        rows = log_store.list()
        if not rows:
            return page("History", "<p>No violations logged yet.</p>")
        lines = ["<table><tr><th>Time</th><th>Track</th><th>Score</th><th>Snapshot</th></tr>"]
        for r in rows:
            url = "/snapshots/" + quote(r["snapshot"])
            lines.append(
                f'<tr><td>{esc(r["time"])}</td><td>{r["track_id"]}</td><td>{r["score"]:.2f}</td>'
                f'<td><img src="{esc(url)}" alt="snapshot" height="80"></td></tr>')
        lines.append("</table>")
        return page("History", "".join(lines))

    @app.get("/history.csv")
    def history_csv():
        return Response(log_store.to_csv(), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="violations.csv"'})

    @app.post("/api/history/clear")
    def history_clear():
        for name in log_store.clear():
            (snapshots_dir / Path(name).name).unlink(missing_ok=True)
        return RedirectResponse("/history.html", status_code=303)

    # ---- settings --------------------------------------------------------------

    @app.post("/api/settings")
    async def update_settings(request: Request):
        form = {k: v for k, v in (await request.form()).items() if isinstance(v, str)}
        old_device = settings.device
        config.update_from_form(settings, form)
        config.save(settings, settings_path)
        if settings.device != old_device:
            with camera.pipeline_lock:
                if camera.pipeline is not None:
                    camera.pipeline.reload()
            with upload_lock:
                if upload_pipeline.get("p") is not None:
                    upload_pipeline["p"].reload()
        return RedirectResponse("/settings.html", status_code=303)

    @app.get("/settings.html")
    def settings_page():
        path = root / "web" / "settings.html"
        if not path.is_file():
            return Response("settings.html not found", status_code=404)
        return HTMLResponse(fill_settings_html(path.read_text(encoding="utf-8"), settings))

    # ---- uploads ---------------------------------------------------------------

    def run_image(path: Path, out: Path):
        frame = cv2.imread(str(path))
        if frame is None:
            return False
        with upload_lock:
            if upload_pipeline.get("p") is None:
                upload_pipeline["p"] = factory(settings)
            p = upload_pipeline["p"]
            p.reset()
            people = []
            # Smoothing needs several frames to decide, so feed the same image repeatedly.
            for i in range(15):
                people = p.process(frame, i * 0.5)
            cv2.imwrite(str(out), p.annotate(frame, people))
        return True

    def run_video(path: Path, out: Path):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if 1 < fps < 240 else 25.0
        writer = None
        with upload_lock:
            if upload_pipeline.get("p") is None:
                upload_pipeline["p"] = factory(settings)
            p = upload_pipeline["p"]
            p.reset()
            count = 0
            while count < MAX_VIDEO_FRAMES:
                ok, frame = cap.read()
                if not ok:
                    break
                people = p.process(frame, count / fps)
                shown = p.annotate(frame, people)
                if writer is None:
                    h, w = shown.shape[:2]
                    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             (w, h))
                writer.write(shown)
                count += 1
        cap.release()
        if writer is not None:
            writer.release()
        return count > 0

    @app.post("/api/upload")
    def upload(file: UploadFile = File(...)):
        ext = Path(file.filename or "").suffix.lower()
        back = '<p><a href="/upload.html">Back to upload</a></p>'
        if ext not in IMAGE_EXTS | VIDEO_EXTS:
            return page("Upload failed", "<h1>Unsupported file type</h1>"
                        f"<p>Allowed: {esc(', '.join(sorted(IMAGE_EXTS | VIDEO_EXTS)))}</p>" + back)
        token = uuid.uuid4().hex[:12]
        src = uploads_dir / f"{token}-input{ext}"
        with src.open("wb") as f:
            while chunk := file.file.read(1 << 20):
                f.write(chunk)
        if ext in IMAGE_EXTS:
            out = uploads_dir / f"{token}-result.jpg"
            ok = run_image(src, out)
        else:
            out = uploads_dir / f"{token}-result.mp4"
            ok = run_video(src, out)
        src.unlink(missing_ok=True)  # keep only the annotated result
        if not ok:
            return page("Upload failed", "<h1>Could not read that file</h1>" + back)
        url = esc("/uploads/" + out.name)
        if ext in IMAGE_EXTS:
            shown = f'<img src="{url}" alt="Annotated result">'
        else:
            shown = f'<p><a href="{url}" download>Download the annotated video</a></p>'
        return page("Result", f"<h1>Result</h1>{shown}" + back)

    # ---- static files (order matters: the "/" mount must be last) ---------------

    app.mount("/snapshots", StaticFiles(directory=snapshots_dir), name="snapshots")
    app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")
    app.mount("/", StaticFiles(directory=root / "web", html=True, check_dir=False), name="web")
    return app
