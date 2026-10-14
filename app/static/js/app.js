const $ = (id) => document.getElementById(id);
let cfg = {}, source = null, timer = null, benchTimer = null;
const charts = {};

async function api(url, opts) {
  const r = await fetch(url, opts);
  let d = {};
  try { d = await r.json(); } catch (_) {}
  if (!r.ok) throw new Error(d.error || `${r.status} ${r.statusText}`);
  return d;
}
function showError(msg) { const a = $("alert"); a.textContent = msg; a.classList.toggle("d-none", !msg); }
const fmt = (v, nd = 2) => (v === null || v === undefined ? "—" : Number(v).toFixed(nd));

function paramsHtml(p) {
  const opt = (arr, d) => arr.map((x) => `<option ${x === d ? "selected" : ""}>${x}</option>`).join("");
  return `
    <label class="form-label small">Model (nano = fastest, medium = most accurate)</label>
    <select id="${p}-model" class="form-select form-select-sm">${opt(cfg.models, cfg.defaults.model)}</select>
    <label class="form-label small mt-2">Tracker</label>
    <select id="${p}-tracker" class="form-select form-select-sm">${opt(cfg.trackers, cfg.defaults.tracker)}</select>
    <div class="form-check form-switch mt-2"><input class="form-check-input" type="checkbox" id="${p}-tracking" checked>
      <label class="form-check-label small">Enable tracking</label></div>
    <label class="form-label small mt-2">Confidence: <span id="${p}-conf-v">0.25</span></label>
    <input id="${p}-conf" type="range" class="form-range" min="0.05" max="0.95" step="0.05" value="0.25">
    <label class="form-label small">IoU (NMS): <span id="${p}-iou-v">0.70</span></label>
    <input id="${p}-iou" type="range" class="form-range" min="0.1" max="0.95" step="0.05" value="0.7">`;
}
function readParams(p, fd = new FormData()) {
  fd.append("model", $(`${p}-model`).value);
  fd.append("tracker", $(`${p}-tracker`).value);
  fd.append("tracking", $(`${p}-tracking`).checked);
  fd.append("conf", $(`${p}-conf`).value);
  fd.append("iou", $(`${p}-iou`).value);
  return fd;
}

// ---------- dashboard ----------
const CARDS = [
  ["frames", "Frames processed"], ["total_detections", "Total detections"], ["current_visible", "Currently visible"],
  ["unique_ids", "Unique track IDs"], ["avg_confidence", "Avg confidence"], ["avg_inference_ms", "Avg inference (ms)"],
  ["inference_fps", "Inference FPS"], ["processing_fps", "End-to-end FPS"], ["duration_s", "Duration (s)"],
];
function initDashboard() {
  $("cards").innerHTML = CARDS.map(([k, l]) =>
    `<div class="col-6 col-md-4 col-xl-2"><div class="card metric p-3"><div class="v" id="m-${k}">—</div><div class="l">${l}</div></div></div>`).join("");
  charts.timeline = new Chart($("chart-timeline"), { type: "line",
    data: { labels: [], datasets: [{ label: "Detections", data: [], borderColor: "#58a6ff", tension: .25, fill: false }] },
    options: { maintainAspectRatio: false, scales: { x: { title: { display: true, text: "Seconds" } } } } });
  charts.classes = new Chart($("chart-classes"), { type: "doughnut",
    data: { labels: [], datasets: [{ data: [] }] }, options: { maintainAspectRatio: false } });
  charts.bench = new Chart($("chart-bench"), { type: "bar",
    data: { labels: [], datasets: [{ label: "Inference FPS", data: [], backgroundColor: "#3fb950" },
                                   { label: "End-to-end FPS", data: [], backgroundColor: "#58a6ff" }] },
    options: { maintainAspectRatio: false } });
}
function renderStats(s, label) {
  $("dash-source").textContent = label || "no data";
  const has = s && s.frames > 0;
  CARDS.forEach(([k]) => {
    const v = has ? s[k] : null;
    $(`m-${k}`).textContent = k === "avg_confidence" ? fmt(v, 3) : ["frames", "total_detections", "current_visible", "unique_ids"].includes(k) ? (v ?? "—") : fmt(v, 1);
  });
  const cc = has ? s.class_counts : {};
  const hasClasses = Object.keys(cc).length > 0;
  $("empty-classes").style.display = hasClasses ? "none" : "flex";
  charts.classes.data.labels = Object.keys(cc);
  charts.classes.data.datasets[0].data = Object.values(cc);
  charts.classes.update("none");
  const tl = has ? s.timeline : [];
  $("empty-timeline").style.display = tl.length ? "none" : "flex";
  charts.timeline.data.labels = tl.map((p) => p.t);
  charts.timeline.data.datasets[0].data = tl.map((p) => p.count);
  charts.timeline.update("none");
}
function setReports(prefix, id) {
  const base = id ? `/api/videos/${id}` : "/api/webcam";
  $("report-links").innerHTML = ["json", "csv", "pdf"].map((f) =>
    `<a class="btn btn-sm btn-outline-secondary ms-1" href="${base}/report.${f}">Download ${f.toUpperCase()}</a>`).join("");
}

// ---------- webcam ----------
async function pollLive() {
  try {
    const s = await api("/api/webcam/status");
    renderStats(s.stats, "webcam session");
    $("live-meta").textContent = s.state === "running" ? `Live FPS (smoothed): ${s.live_fps}` : "";
    if (s.state !== "running") {
      liveStopped();
      if (s.error) showError(s.error);
    }
  } catch (e) { showError(e.message); }
}
function liveStopped() {
  clearInterval(timer); timer = null;
  $("live-feed").classList.add("d-none"); $("live-feed").removeAttribute("src");
  $("live-placeholder").classList.remove("d-none");
  $("live-start").disabled = false; $("live-stop").disabled = true;
}
$("live-start").onclick = async () => {
  showError("");
  const fd = readParams("live"); fd.append("camera_index", $("live-camera_index").value);
  $("live-start").disabled = true;
  try {
    await api("/api/webcam/start", { method: "POST", body: fd });
    $("live-feed").src = "/video_feed?t=" + Date.now();
    $("live-feed").classList.remove("d-none"); $("live-placeholder").classList.add("d-none");
    $("live-stop").disabled = false;
    setReports("live"); source = { type: "live" };
    clearInterval(timer); timer = setInterval(pollLive, 1000);
  } catch (e) { showError(e.message); $("live-start").disabled = false; }
};
$("live-stop").onclick = async () => {
  try { await api("/api/webcam/stop", { method: "POST" }); } catch (e) { showError(e.message); }
  liveStopped(); pollLive();
};

// ---------- video ----------
async function pollVideo(id) {
  try {
    const j = await api(`/api/videos/${id}`);
    const pct = Math.round(j.progress * 100);
    $("video-progress").style.width = pct + "%"; $("video-progress").textContent = pct + "%";
    $("video-status").textContent = `Status: ${j.status} · ${j.processed_frames}/${j.total_frames || "?"} frames · video duration ${j.video_duration_s}s`;
    renderStats(j.stats, `video ${id}`);
    if (j.status === "completed") {
      clearInterval(timer); $("video-start").disabled = false;
      const v = $("video-player"); v.src = `/api/videos/${id}/output`; v.classList.remove("d-none");
      $("video-placeholder").classList.add("d-none");
      $("video-downloads").innerHTML = `<a class="btn btn-sm btn-success" href="/api/videos/${id}/output?download=1">Download video</a>`;
      setReports("video", id);
    } else if (j.status === "failed") {
      clearInterval(timer); $("video-start").disabled = false; showError(j.error || "Processing failed.");
    }
  } catch (e) { clearInterval(timer); showError(e.message); $("video-start").disabled = false; }
}
$("video-start").onclick = async () => {
  showError("");
  const file = $("video-file").files[0];
  if (!file) return showError("Choose a video file first.");
  const fd = readParams("video"); fd.append("video", file);
  $("video-start").disabled = true; $("video-progress-wrap").classList.remove("d-none");
  $("video-status").textContent = "Uploading…";
  try {
    const j = await api("/api/videos", { method: "POST", body: fd });
    clearInterval(timer); timer = setInterval(() => pollVideo(j.id), 1000); pollVideo(j.id);
  } catch (e) { showError(e.message); $("video-start").disabled = false; }
};

// ---------- benchmark ----------
function renderBench(results) {
  $("bench-empty").classList.toggle("d-none", results.length > 0);
  $("bench-results").classList.toggle("d-none", results.length === 0);
  charts.bench.data.labels = results.map((r) => r.model);
  charts.bench.data.datasets[0].data = results.map((r) => r.inference_fps);
  charts.bench.data.datasets[1].data = results.map((r) => r.end_to_end_fps);
  charts.bench.update();
  $("bench-table").innerHTML = results.map((r) => `<tr><td>${r.model}</td><td>${r.frames}</td><td>${fmt(r.avg_inference_ms)}</td><td>${fmt(r.inference_fps)}</td><td>${fmt(r.end_to_end_fps)}</td><td>${r.device}</td></tr>`).join("");
}
async function pollBench(id) {
  try {
    const j = await api(`/api/benchmark/${id}`);
    const pct = Math.round(j.progress * 100);
    $("bench-progress").style.width = pct + "%"; $("bench-progress").textContent = pct + "%";
    $("bench-status").textContent = `Status: ${j.status} · ${j.results.length} model(s) done`;
    renderBench(j.results);
    if (j.status === "completed" || j.status === "failed") {
      clearInterval(benchTimer); $("bench-start").disabled = false;
      if (j.error) showError(j.error);
    }
  } catch (e) { clearInterval(benchTimer); showError(e.message); $("bench-start").disabled = false; }
}
$("bench-start").onclick = async () => {
  showError("");
  const file = $("bench-file").files[0];
  if (!file) return showError("Choose a video file for the benchmark.");
  const fd = new FormData(); fd.append("video", file); fd.append("max_frames", $("bench-frames").value);
  document.querySelectorAll(".bench-model:checked").forEach((c) => fd.append("models", c.value));
  $("bench-start").disabled = true; $("bench-progress-wrap").classList.remove("d-none");
  try {
    const j = await api("/api/benchmark", { method: "POST", body: fd });
    clearInterval(benchTimer); benchTimer = setInterval(() => pollBench(j.id), 1500);
  } catch (e) { showError(e.message); $("bench-start").disabled = false; }
};

// ---------- init ----------
(async function init() {
  try { cfg = await api("/api/models"); } catch (e) { return showError("Could not load configuration: " + e.message); }
  document.querySelectorAll(".params").forEach((el) => {
    el.innerHTML = paramsHtml(el.dataset.prefix);
    ["conf", "iou"].forEach((k) => {
      const i = $(`${el.dataset.prefix}-${k}`);
      i.oninput = () => ($(`${el.dataset.prefix}-${k}-v`).textContent = Number(i.value).toFixed(2));
    });
  });
  $("bench-models").innerHTML = cfg.models.map((m) =>
    `<div class="form-check"><input class="form-check-input bench-model" type="checkbox" value="${m}" id="bm-${m}" checked>
     <label class="form-check-label small" for="bm-${m}">${m}</label></div>`).join("");
  $("video-limits").textContent = `Formats: ${cfg.extensions.join(", ")} · max ${cfg.max_upload_mb} MB`;
  initDashboard(); renderStats(null);
})();
