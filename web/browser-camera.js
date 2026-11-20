// RVU ID Card Detector - browser-camera.js
// Runs the detector inside the browser, for when the Python program is not running
// (for example on the GitHub Pages copy of this website).
//
// It follows the same steps as the Python code in idcard/:
//   1. the webcam picture comes from navigator.mediaDevices.getUserMedia
//   2. the YOLO11 pose model (models/yolo11n-pose.onnx) finds people and their shoulders
//      and hips, using ONNX Runtime Web (the JavaScript version of ONNX Runtime)
//   3. a simple tracker gives each person a number that stays the same between frames
//   4. the chest area is checked for a thin strap in the band colour
//   5. the result is smoothed over 15 frames so the label does not flicker
// Nothing leaves the computer: the picture is only drawn on a <canvas> on this page.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
const MODEL_URL = "models/yolo11n-pose.onnx";
const SIZE = 640; // the model looks at a 640 x 640 picture

// same numbers as the Python code
const PERSON_CONF = 0.4;
const KEYPOINT_CONF = 0.5;
const TOLERANCE = 25;
const ON_THRESHOLD = 0.6;
const OFF_THRESHOLD = 0.35;
const WINDOW = 15;
const COLORS = { WEARING: "#3cb43c", NOT_WEARING: "#dc2828", UNSURE: "#969696" };

const cam = {
  video: null,
  canvas: null,
  session: null,
  input: document.createElement("canvas"), // 640 x 640 picture for the model
  work: document.createElement("canvas"), // small chest picture for the colour check
  tracks: [],
  nextId: 1,
  running: false,
};

// ---- loading things ------------------------------------------------------------------
function loadScript(src) {
  return new Promise((resolve, reject) => {
    const tag = document.createElement("script");
    tag.src = src;
    tag.onload = resolve;
    tag.onerror = () => reject(new Error("could not load " + src));
    document.head.appendChild(tag);
  });
}

async function startBrowserCamera() {
  const button = document.getElementById("cam-start");
  const status = document.getElementById("cam-status");
  button.disabled = true;
  try {
    status.textContent = "Asking for the camera...";
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    // the <video> stays hidden on the page: we only copy its pictures onto the canvas
    cam.video = document.getElementById("cam-video");
    cam.video.srcObject = stream;
    await cam.video.play();

    status.textContent = "Loading the detector (about 12 MB, only the first time)...";
    if (!window.ort) await loadScript(ORT_URL + "ort.webgpu.min.js");
    ort.env.wasm.wasmPaths = ORT_URL;
    // use several CPU cores when the page is allowed to (the Node.js server allows it)
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    cam.session = await createSession();

    // swap the sample picture for a canvas that shows the camera
    const live = document.getElementById("live");
    cam.canvas = document.createElement("canvas");
    cam.canvas.id = "live";
    cam.canvas.width = cam.video.videoWidth;
    cam.canvas.height = cam.video.videoHeight;
    live.replaceWith(cam.canvas);
    showMessage("The detector is running in this browser. The camera picture stays on this computer.", false);
    status.textContent = "";
    button.hidden = true;
    document.getElementById("cam-stop").hidden = false;
    cam.running = true;
    loop();
  } catch (error) {
    console.error(error);
    status.textContent = "";
    button.disabled = false;
    showMessage("Could not start the camera: " + error.message, true);
  }
}

// try the graphics card first (WebGPU, much faster), and the CPU (WebAssembly) if that fails
async function createSession() {
  if (navigator.gpu) {
    try {
      const session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["webgpu"] });
      cam.backend = "graphics card";
      return session;
    } catch (error) {
      console.log("WebGPU did not work, using the CPU:", error.message);
    }
  }
  cam.backend = ort.env.wasm.numThreads > 1 ? `CPU, ${ort.env.wasm.numThreads} cores` : "CPU";
  return ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
}

function stopBrowserCamera() {
  cam.running = false;
  if (cam.video && cam.video.srcObject) {
    cam.video.srcObject.getTracks().forEach((track) => track.stop());
  }
  showMessage("Camera stopped.", false);
  document.getElementById("cam-stop").hidden = true;
  const button = document.getElementById("cam-start");
  button.hidden = false;
  button.disabled = false;
}

// ---- main loop: one frame at a time ----------------------------------------------------
async function loop() {
  if (!cam.running) return;
  const started = performance.now();
  const ctx = cam.canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(cam.video, 0, 0, cam.canvas.width, cam.canvas.height);
  const frame = ctx.getImageData(0, 0, cam.canvas.width, cam.canvas.height);

  const people = await detectPeople();
  const tracked = updateTracks(people);
  const bandColor = document.getElementById("cam-color").value;
  let wearing = 0, notWearing = 0, unsure = 0;
  // check every chest first, then draw, so the boxes do not end up in the colour check
  for (const t of tracked) smooth(t, frameScore(frame, t.keypoints, bandColor));
  for (const t of tracked) {
    if (t.state === "WEARING") wearing++;
    else if (t.state === "NOT_WEARING") notWearing++;
    else unsure++;
    drawBox(ctx, t, t.state);
  }
  const ms = Math.round(performance.now() - started);
  document.getElementById("cam-count").textContent =
    `People: ${tracked.length} | Wearing: ${wearing} | Not wearing: ${notWearing} | Not sure: ${unsure} | ${ms} ms per frame (${cam.backend})`;
  requestAnimationFrame(loop);
}

// ---- step 2: the pose model ---------------------------------------------------------------
async function detectPeople() {
  const w = cam.video.videoWidth, h = cam.video.videoHeight;
  // letterbox: shrink to fit 640 x 640 and fill the rest with grey (like Ultralytics)
  const scale = Math.min(SIZE / w, SIZE / h);
  const newW = Math.round(w * scale), newH = Math.round(h * scale);
  const padX = Math.round((SIZE - newW) / 2 - 0.1), padY = Math.round((SIZE - newH) / 2 - 0.1);
  cam.input.width = SIZE;
  cam.input.height = SIZE;
  const ictx = cam.input.getContext("2d", { willReadFrequently: true });
  ictx.fillStyle = "rgb(114,114,114)";
  ictx.fillRect(0, 0, SIZE, SIZE);
  ictx.drawImage(cam.video, padX, padY, newW, newH);
  const pixels = ictx.getImageData(0, 0, SIZE, SIZE).data;

  // the model wants three planes (red, green, blue) with values from 0 to 1
  const area = SIZE * SIZE;
  const data = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    data[i] = pixels[i * 4] / 255;
    data[area + i] = pixels[i * 4 + 1] / 255;
    data[2 * area + i] = pixels[i * 4 + 2] / 255;
  }
  const tensor = new ort.Tensor("float32", data, [1, 3, SIZE, SIZE]);
  const result = await cam.session.run({ [cam.session.inputNames[0]]: tensor });
  const out = result[cam.session.outputNames[0]].data; // shape 1 x 56 x 8400
  const n = 8400;

  // every column is one guess: cx, cy, w, h, score, then 17 keypoints (x, y, confidence)
  let found = [];
  for (let j = 0; j < n; j++) {
    const score = out[4 * n + j];
    if (score < 0.1) continue;
    const cx = out[j], cy = out[n + j], bw = out[2 * n + j], bh = out[3 * n + j];
    const box = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2].map((v, k) =>
      k % 2 === 0 ? (v - padX) / scale : (v - padY) / scale);
    const keypoints = [];
    for (let k = 0; k < 17; k++) {
      keypoints.push([
        (out[(5 + k * 3) * n + j] - padX) / scale,
        (out[(6 + k * 3) * n + j] - padY) / scale,
        out[(7 + k * 3) * n + j],
      ]);
    }
    found.push({ box, score, keypoints });
  }
  // non maximum suppression: keep the best box and drop the ones that overlap it
  found.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const f of found) {
    if (kept.every((k) => iou(k.box, f.box) < 0.45)) kept.push(f);
  }
  return kept;
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a[2] - a[0]) * (a[3] - a[1]), areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / Math.max(areaA + areaB - inter, 1e-6);
}

// ---- step 3: tracker (match each person to last frame's boxes by overlap) -------------
function updateTracks(people) {
  const result = [];
  const free = [...cam.tracks];
  for (const p of people) {
    let best = null, bestIou = 0.3;
    for (const t of free) {
      const overlap = iou(t.box, p.box);
      if (overlap > bestIou) { best = t; bestIou = overlap; }
    }
    if (best) {
      free.splice(free.indexOf(best), 1);
    } else if (p.score >= PERSON_CONF) {
      best = { id: cam.nextId++, scores: [], state: "UNSURE" };
    } else {
      continue; // weak guess that matches nobody: ignore it
    }
    best.box = p.box;
    best.keypoints = p.keypoints;
    best.lost = 0;
    result.push(best);
  }
  // keep unmatched tracks for a little while so a person keeps their number
  for (const t of free) t.lost++;
  cam.tracks = [...result, ...free.filter((t) => t.lost < 30)];
  return result;
}

// ---- step 4: where is the chest, and does it have a strap? ---------------------------
function torsoBox(kp, frameW, frameH) {
  const [ls, rs, lh, rh] = [kp[5], kp[6], kp[11], kp[12]];
  if (ls[2] < KEYPOINT_CONF || rs[2] < KEYPOINT_CONF) return null; // shoulders hidden
  const faceConf = Math.max(kp[0][2], kp[1][2], kp[2][2]);
  if (faceConf < 0.5 || ls[0] <= rs[0]) return null; // facing away
  const midX = (ls[0] + rs[0]) / 2, midY = (ls[1] + rs[1]) / 2;
  const shoulderW = Math.hypot(ls[0] - rs[0], ls[1] - rs[1]);
  let torsoH = 1.4 * shoulderW;
  if (lh[2] >= KEYPOINT_CONF && rh[2] >= KEYPOINT_CONF) torsoH = (lh[1] + rh[1]) / 2 - midY;
  if (torsoH <= 0 || shoulderW < 0.35 * torsoH) return null; // odd pose or side-on
  const x1 = Math.max(0, Math.min(ls[0], rs[0]) + 0.05 * shoulderW);
  const x2 = Math.min(frameW, Math.max(ls[0], rs[0]) - 0.05 * shoulderW);
  const y1 = Math.max(0, midY - 0.1 * torsoH);
  const y2 = Math.min(frameH, midY + 0.85 * torsoH);
  if (x2 - x1 < 70 || y2 - y1 < 90) return null; // too far away to judge
  return [x1, y1, x2, y2];
}

function frameScore(frame, keypoints, bandColor) {
  const box = torsoBox(keypoints, frame.width, frame.height);
  if (!box) return null;
  // shrink the chest to 128 pixels high, so the rules do not depend on distance
  const [x1, y1, x2, y2] = box;
  const H = 128, W = Math.max(8, Math.round((x2 - x1) * H / (y2 - y1)));
  cam.work.width = W;
  cam.work.height = H;
  const wctx = cam.work.getContext("2d", { willReadFrequently: true });
  wctx.drawImage(cam.canvas, x1, y1, x2 - x1, y2 - y1, 0, 0, W, H);
  const px = wctx.getImageData(0, 0, W, H).data;

  // mark the pixels that are close to the band colour
  const target = toLab(...hexToRgb(bandColor));
  const colourful = Math.min(1, Math.hypot(target[1], target[2]) / 30);
  let mask = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) {
    const lab = toLab(px[i * 4], px[i * 4 + 1], px[i * 4 + 2]);
    let shade = Math.min(2.5, Math.max(0.5, target[0] / Math.max(lab[0], 1)));
    shade = 1 + colourful * (shade - 1);
    const dl = (1 - 0.75 * colourful) * (lab[0] - target[0]);
    const da = lab[1] * shade - target[1], db = lab[2] * shade - target[2];
    mask[i] = Math.sqrt(dl * dl + da * da + db * db) <= TOLERANCE ? 1 : 0;
  }
  mask = dilate(erode(mask, W, H), W, H); // remove small specks
  const matched = mask.reduce((sum, v) => sum + v, 0) / mask.length;
  if (matched >= 0.45) return null; // shirt is the band colour: cannot judge

  const straps = strapLengths(mask, W, H);
  if (straps.length > 4) return null; // striped shirt
  const total = straps.reduce((sum, v) => sum + v, 0) / H;
  return Math.min(1, total / 0.8);
}

function hexToRgb(hex) {
  const v = hex.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(v.slice(i, i + 2), 16));
}

// sRGB colour (0..255) to CIELAB, the same maths OpenCV uses
function toLab(r, g, b) {
  const lin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
  const [R, G, B] = [lin(r), lin(g), lin(b)];
  const x = (0.412453 * R + 0.357580 * G + 0.180423 * B) / 0.950456;
  const y = 0.212671 * R + 0.715160 * G + 0.072169 * B;
  const z = (0.019334 * R + 0.119193 * G + 0.950227 * B) / 1.088754;
  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const L = y > 0.008856 ? 116 * Math.cbrt(y) - 16 : 903.3 * y;
  return [L, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))];
}

// 3 x 3 erode and dilate (together they make a "morphological opening")
function erode(m, W, H) { return morph(m, W, H, (a, b) => a & b, 1); }
function dilate(m, W, H) { return morph(m, W, H, (a, b) => a | b, 0); }
function morph(m, W, H, join, start) {
  const out = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let v = start;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const yy = y + dy, xx = x + dx;
          const p = yy < 0 || yy >= H || xx < 0 || xx >= W ? start : m[yy * W + xx];
          v = join(v, p);
        }
      }
      out[y * W + x] = v;
    }
  }
  return out;
}

// find connected regions and keep the ones shaped like a strap (thin, long, upright)
function strapLengths(mask, W, H) {
  const dist = distanceToEdge(mask, W, H);
  const seen = new Uint8Array(W * H);
  const lengths = [];
  for (let start = 0; start < W * H; start++) {
    if (!mask[start] || seen[start]) continue;
    // flood fill one region
    const stack = [start];
    seen[start] = 1;
    let area = 0, distSum = 0, top = H, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    while (stack.length) {
      const i = stack.pop();
      const x = i % W, y = (i - x) / W;
      area++; distSum += dist[i]; top = Math.min(top, y);
      sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]]) {
        const xx = x + dx, yy = y + dy, j = yy * W + xx;
        if (xx >= 0 && xx < W && yy >= 0 && yy < H && mask[j] && !seen[j]) {
          seen[j] = 1;
          stack.push(j);
        }
      }
    }
    if (area < 15) continue;
    const thickness = Math.max(1, 4 * distSum / area);
    const length = area / thickness;
    const mx = sx / area, my = sy / area;
    const mu20 = sxx / area - mx * mx, mu02 = syy / area - my * my, mu11 = sxy / area - mx * my;
    const angle = 0.5 * Math.atan2(2 * mu11, mu20 - mu02) * 180 / Math.PI;
    const fromVertical = Math.abs(90 - Math.abs(angle));
    if (thickness > 0.15 * W) continue; // too wide: shirt or bag
    if (length < 4 * thickness) continue; // a blob or a logo
    if (length < 0.15 * H) continue; // too short
    if (top > 0.35 * H) continue; // straps start near the neck
    if (fromVertical > 60) continue; // lying flat: a stripe on clothes
    lengths.push(length);
  }
  return lengths;
}

// distance from each marked pixel to the nearest unmarked pixel (two pass chamfer)
function distanceToEdge(mask, W, H) {
  const d = new Float32Array(W * H);
  const big = 1e6, diag = 1.4142;
  for (let i = 0; i < W * H; i++) d[i] = mask[i] ? big : 0;
  const get = (x, y) => (x < 0 || y < 0 || x >= W || y >= H ? 0 : d[y * W + x]);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (!d[i]) continue;
      d[i] = Math.min(d[i], get(x - 1, y) + 1, get(x, y - 1) + 1, get(x - 1, y - 1) + diag, get(x + 1, y - 1) + diag);
    }
  }
  for (let y = H - 1; y >= 0; y--) {
    for (let x = W - 1; x >= 0; x--) {
      const i = y * W + x;
      if (!d[i]) continue;
      d[i] = Math.min(d[i], get(x + 1, y) + 1, get(x, y + 1) + 1, get(x + 1, y + 1) + diag, get(x - 1, y + 1) + diag);
    }
  }
  return d;
}

// ---- step 5: smoothing over the last 15 frames ------------------------------------------
function smooth(track, score) {
  track.scores.push(score);
  if (track.scores.length > WINDOW) track.scores.shift();
  const valid = track.scores.filter((s) => s !== null);
  if (valid.length < 3) {
    track.state = "UNSURE";
  } else {
    track.smoothed = valid.reduce((a, b) => a + b, 0) / valid.length;
    if (track.smoothed >= ON_THRESHOLD) track.state = "WEARING";
    else if (track.smoothed <= OFF_THRESHOLD) track.state = "NOT_WEARING";
  }
  return track.state;
}

// ---- drawing ------------------------------------------------------------------------------
function drawBox(ctx, t, state) {
  const [x1, y1, x2, y2] = t.box;
  const colour = COLORS[state];
  ctx.lineWidth = 3;
  ctx.strokeStyle = colour;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  const label = `#${t.id} ${state} ${(t.smoothed || 0).toFixed(2)}`;
  ctx.font = "16px sans-serif";
  const textW = ctx.measureText(label).width;
  const top = y1 - 22 >= 0 ? y1 - 22 : y1;
  ctx.fillStyle = colour;
  ctx.fillRect(x1, top, textW + 8, 22);
  ctx.fillStyle = "#fff";
  ctx.fillText(label, x1 + 4, top + 16);
}

// ---- buttons ------------------------------------------------------------------------------
document.getElementById("cam-start").addEventListener("click", startBrowserCamera);
document.getElementById("cam-stop").addEventListener("click", stopBrowserCamera);
