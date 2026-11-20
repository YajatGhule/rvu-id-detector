// RVU ID Card Detector - small local web server in Node.js (no packages needed)
//
// It serves the files in the web/ folder at http://localhost:3000 and opens the browser.
// The detection then runs inside the browser ("Use my camera" on the Home page), so Python
// is not needed. Webcams only work on https pages or on localhost, which is why we serve
// the site from this computer instead of opening index.html as a file.
//
// Run:  node node-server/server.js        (close the window or press Ctrl+C to stop)

const http = require("http");
const fs = require("fs");
const path = require("path");
const { exec } = require("child_process");

const PORT = Number(process.env.PORT) || 3000;
const WEB = path.join(__dirname, "..", "web");

// file extension -> content type (the browser needs the right type for .js, .wasm, ...)
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".json": "application/json",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
  ".txt": "text/plain",
};

const server = http.createServer((req, res) => {
  let urlPath = decodeURIComponent(req.url.split("?")[0]);
  if (urlPath.endsWith("/")) urlPath += "index.html";
  const file = path.join(WEB, path.normalize(urlPath));
  // never serve anything outside the web folder
  if (!file.startsWith(WEB)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }
  fs.stat(file, (err, stat) => {
    if (err || !stat.isFile()) {
      // e.g. /stream: that part needs the Python program, so the page falls back to the browser camera
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end("Not found");
      return;
    }
    res.writeHead(200, {
      "Content-Type": TYPES[path.extname(file).toLowerCase()] || "application/octet-stream",
      "Content-Length": stat.size,
    });
    fs.createReadStream(file).pipe(res);
  });
});

server.on("error", (err) => {
  if (err.code === "EADDRINUSE") {
    console.log(`Port ${PORT} is busy, so the detector is probably already running.`);
    openBrowser();
  } else {
    console.log("Could not start the server:", err.message);
  }
});

// only this computer can open the site (127.0.0.1), not other devices on the Wi-Fi
server.listen(PORT, "127.0.0.1", () => {
  console.log(`RVU ID Card Detector is running at http://localhost:${PORT}`);
  console.log("Press \"Use my camera\" on the Home page. Close this window to stop.");
  openBrowser();
});

function openBrowser() {
  if (process.env.NO_BROWSER) return; // used by the automatic test
  const url = `http://localhost:${PORT}/index.html`;
  const command =
    process.platform === "win32" ? `start "" "${url}"` :
    process.platform === "darwin" ? `open "${url}"` : `xdg-open "${url}"`;
  exec(command, () => {}); // if it fails the user can still type the address
}
