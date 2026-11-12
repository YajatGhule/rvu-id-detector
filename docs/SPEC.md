# Build contract (phase 1: skeleton, baseline detector, settings, log, uploads)

Read docs/PLAN.md for the big picture. This file is the contract between three parallel
workers. Stay inside the files you own. Do not edit pyproject.toml (ask in your report if a
dependency is missing), do not run `git` write commands, do not open the webcam
(/dev/video*), do not run `uv add`/`uv lock`. Run things with `uv run ...` from the repo root
(the venv is already synced). Python 3.12. Keep code small and readable; the owner is an
intermediate Python programmer who must be able to explain it.

## Paths
- repo root: ~/Projects/rvu-id-detector
- models/yolo11n-pose.onnx (committed), data/ (gitignored: data/idcard.db, data/snapshots/,
  data/uploads/), settings.json at repo root (gitignored)
- Server address is fixed: http://127.0.0.1:8000

## Settings (idcard/config.py, owner: SERVER worker)
```python
@dataclass
class Settings:
    band_color: str = "#1e40af"      # hex, lanyard band colour
    tolerance: float = 25.0          # colour distance tolerance, 5..80
    on_threshold: float = 0.6        # smoothed score >= this -> WEARING
    off_threshold: float = 0.35      # smoothed score <= this -> NOT_WEARING
    violation_seconds: float = 3.0   # continuous NOT_WEARING before logging
    person_conf: float = 0.4         # person detection confidence
    camera: str = "0"                # camera index, or a video file path
    device: str = "auto"             # auto|cpu|cuda|directml|openvino|coreml
def load(path) -> Settings          # missing/corrupt file -> defaults
def save(settings, path) -> None
def update_from_form(settings, form: dict) -> None   # validates + clamps, ignores unknown keys
```
The pipeline reads attributes from the live Settings object on every frame, so changes apply
immediately. Pipeline code must only use attribute access (tests may pass a SimpleNamespace).

## Pipeline (idcard/pipeline.py + color.py, fusion.py, geometry.py, device.py, backends/, owner: CORE worker)
```python
@dataclass
class Person:
    track_id: int
    box: tuple[int, int, int, int]      # x1,y1,x2,y2 in frame pixels
    state: str                          # "WEARING" | "NOT_WEARING" | "UNSURE"
    score: float                        # smoothed 0..1
    new_violation: bool                 # True exactly once per track, when NOT_WEARING has
                                        # lasted settings.violation_seconds

class Pipeline:
    backend_name: str                   # e.g. "CPUExecutionProvider"
    def __init__(self, settings, models_dir: Path | None = None)
    def process(self, frame_bgr: np.ndarray, now: float) -> list[Person]
    def annotate(self, frame_bgr, people) -> np.ndarray   # returns a new image with boxes+labels
    def reset(self) -> None             # forget tracks (new source / new upload)
    def reload(self) -> None            # rebuild the inference session (device changed)

# idcard/device.py
def available_devices() -> list[str]            # subset of cpu,cuda,directml,openvino,coreml present in onnxruntime
def choose_providers(preference: str, available: list[str] | None = None) -> list[str]
        # onnxruntime provider names, best first, always ending with CPUExecutionProvider;
        # "auto" order: cuda, directml, coreml, openvino, cpu; unavailable preference -> falls back
```
Colours when drawing: green WEARING, red NOT_WEARING, grey UNSURE. Label: `#id STATE score`.

## HTTP (idcard/server.py, camera.py, store.py, cli.py, owner: SERVER worker)
`create_app(root: Path = repo root, pipeline_factory=None) -> FastAPI`; `pipeline_factory(settings)`
defaults to `idcard.pipeline.Pipeline` and exists so tests can inject a fake.
- `GET /stream` multipart MJPEG of annotated frames. Camera is opened only while at least one
  client is connected, and released a few seconds after the last one leaves. If the camera
  cannot be opened, stream a generated "camera unavailable" image instead of failing.
- `GET /fragments/stats` tiny full HTML page (with `<meta http-equiv="refresh" content="1">`,
  links `/style.css`): counts of wearing / not wearing / unsure, FPS, backend name.
- `GET /fragments/history` full HTML page: table of violations, newest first: time, track id,
  score, snapshot thumbnail (`/snapshots/<file>`). Links `/style.css`.
- `GET /history.csv` CSV export. `POST /api/history/clear` deletes rows + snapshot files, 303 -> /history.html
- `POST /api/settings` form fields named exactly like the Settings attributes; saves
  settings.json; calls pipeline.reload() if device changed; 303 -> /settings.html
- `GET /settings.html` served from web/settings.html with the current values filled in server
  side (rewrite `value="..."` of the input with that `name`, and add `selected` to the right
  `<option>` of `<select name="device">`). No JavaScript anywhere.
- `POST /api/upload` multipart field `file` (image or video). Runs a fresh pipeline pass
  (reset first; for a single image call process a few times or otherwise make sure the state
  is decided rather than left UNSURE by smoothing), writes the annotated result to
  data/uploads/, returns a small HTML result page showing the image, or a download link for
  video, with a link back to /upload.html.
- everything else: static files from web/ (`/` -> index.html), `/snapshots/*` from data/snapshots.
- When `Person.new_violation` is true, save a crop of the person box to data/snapshots/ and
  insert a row in SQLite (store.py: `add`, `list`, `clear`, `to_csv`).
- CLI (`idcard`): `serve [--port 8000] [--source CAMERA_OR_FILE]`, `doctor` (prints onnxruntime
  providers, chosen backend, and FPS over 30 synthetic frames).

## Web pages (web/, owner: WEB worker)
index.html, settings.html, upload.html, history.html, about.html, style.css. No JavaScript, no
frameworks, no external resources. Every absolute reference to the backend uses
`http://127.0.0.1:8000/...` so the pages also work when opened straight from disk; links
between pages and to style.css are relative.
- index.html: `<img src="http://127.0.0.1:8000/stream" alt="Detector offline ...">` plus
  `<iframe src="http://127.0.0.1:8000/fragments/stats">` and a colour legend.
- settings.html: `<form method="post" action="http://127.0.0.1:8000/api/settings">` with
  `<input type="color" name="band_color" value="#1e40af">`, `<input type="range" name="tolerance" min="5" max="80" value="25">`,
  number inputs `on_threshold`, `off_threshold` (step 0.05), `violation_seconds`, `person_conf`,
  text input `camera`, `<select name="device">` with options auto, cpu, cuda, directml,
  openvino, coreml. Write every input as `name="..."` followed later in the same tag by
  `value="..."` (the server rewrites values by that pattern).
- upload.html: `<form method="post" enctype="multipart/form-data" action="http://127.0.0.1:8000/api/upload">` with `<input type="file" name="file">`.
- history.html: iframe of `/fragments/history`, link to `/history.csv`, a POST form button to `/api/history/clear`.
- about.html: plain-language explanation of the pipeline (from docs/PLAN.md).
