// RVU ID Card Detector - script.js
// One small file shared by every page. Each part only runs if its page has the element.

// ---- Module 4 basics: variables, arrays and objects ----------------------------------
const pages = [
  { file: "index.html", key: "1", name: "Home" },
  { file: "upload.html", key: "2", name: "Upload" },
  { file: "history.html", key: "3", name: "History" },
  { file: "settings.html", key: "4", name: "Settings" },
  { file: "about.html", key: "5", name: "About" },
  { file: "gallery.html", key: "6", name: "Gallery" },
];
const allowedFiles = ["jpg", "jpeg", "png", "bmp", "webp", "mp4", "avi", "mov", "mkv", "webm"];
const sampleImage = "img/sample-result.jpg";

// ---- small helper functions -----------------------------------------------------------
function showMessage(text, isError) {
  const box = document.getElementById("message");
  if (!box) return;
  box.textContent = text;
  box.className = isError ? "error" : "";
}

// arrow function
const getExtension = (name) => name.split(".").pop().toLowerCase();

// destructuring: take two fields out of an object
function describePage(page) {
  const { name, key } = page;
  return `${name} (press ${key})`;
}

// ---- 1. mark the current page in the menu --------------------------------------------
const current = location.pathname.split("/").pop() || "index.html";
document.querySelectorAll("nav a").forEach((link) => {
  if (link.getAttribute("href") === current) link.classList.add("active");
});

// ---- 2. keyboard shortcut: press 1 to 6 to change page (not while typing) ------------
document.addEventListener("keydown", (event) => {
  const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName);
  if (typing) return;
  const target = pages.find((page) => page.key === event.key);
  if (target) location.href = target.file;
});

// ---- 3. Home page: show a sample picture when the detector is not running -----------
const live = document.getElementById("live");
if (live) {
  const useSample = () => {
    if (live.src.endsWith(sampleImage)) return; // already showing the sample
    live.src = sampleImage; // this image is part of the website, so it always loads
    showMessage("The detector is not running here, so this is a sample result. " +
      "Start it with: uv run idcard serve", true);
  };
  live.addEventListener("error", useSample);
  // the error can happen before this script runs, so check once more now
  if (live.complete && live.naturalWidth === 0) useSample();
}

// ---- 4. Settings page: check the values before the form is sent ----------------------
const settingsForm = document.getElementById("settings-form");
if (settingsForm) {
  const color = document.getElementById("band_color");
  const colorText = document.getElementById("color-text");
  colorText.textContent = color.value;
  color.addEventListener("input", () => (colorText.textContent = color.value));

  settingsForm.addEventListener("submit", (event) => {
    const on = Number(document.getElementById("on_threshold").value);
    const off = Number(document.getElementById("off_threshold").value);
    const tolerance = Number(document.getElementById("tolerance").value);
    if (off >= on) {
      event.preventDefault();
      showMessage("NOT WEARING score must be lower than the WEARING score.", true);
    } else if (tolerance < 5 || tolerance > 80) {
      event.preventDefault();
      showMessage("Colour tolerance must be between 5 and 80.", true);
    } else {
      showMessage("Saving...", false);
    }
  });
}

// ---- 5. Upload page: check the file type and show its name ---------------------------
const uploadForm = document.getElementById("upload-form");
if (uploadForm) {
  const fileInput = document.getElementById("file");
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;
    const sizeMB = (file.size / 1024 / 1024).toFixed(1);
    showMessage(`Chosen: ${file.name} (${sizeMB} MB)`, false);
  });
  uploadForm.addEventListener("submit", (event) => {
    const file = fileInput.files[0];
    if (!file) {
      event.preventDefault();
      showMessage("Please choose a file first.", true);
    } else if (!allowedFiles.includes(getExtension(file.name))) {
      event.preventDefault();
      showMessage("This file type is not allowed. Use " + allowedFiles.join(", ") + ".", true);
    }
  });
}

// ---- 6. History page: ask before deleting everything (HTML dialog element) -----------
const askButton = document.getElementById("ask-delete");
const dialog = document.getElementById("confirm-dialog");
if (askButton && dialog) {
  askButton.addEventListener("click", () => dialog.showModal());
  document.getElementById("cancel-delete").addEventListener("click", () => dialog.close());
}

// ---- 7. About page: show or hide the details (DOM manipulation) ----------------------
const toggle = document.getElementById("toggle-details");
const details = document.getElementById("details");
if (toggle && details) {
  toggle.addEventListener("click", () => {
    details.hidden = !details.hidden;
    toggle.textContent = details.hidden ? "Show the technology table" : "Hide the technology table";
  });
}

// ---- 8. ES6 spread: the menu list for the console (open F12 to see it) --------------
const menuNames = [...pages.map((page) => page.name)];
console.log("Pages:", menuNames.join(", "), "|", describePage(pages[0]));
