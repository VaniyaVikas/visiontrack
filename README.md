# VisionTrack AI — Real-Time Object Detection & Tracking Platform

Flask + Ultralytics YOLO11 + ByteTrack/BoT-SORT + OpenCV, with a Bootstrap/Chart.js dashboard.

## Features
- **Webcam**: start/stop MJPEG stream with boxes, class, confidence, track IDs, trails, live FPS/latency.
- **Video upload**: extension/size/readability validation, frame-by-frame processing (no full video in RAM),
  progress, annotated H.264 output (needs `ffmpeg`; falls back to mp4v), preview + download.
- **Detection engine**: `ModelRegistry` (validated ids `yolo11n/s/m`, weights cached in `./weights`), configurable conf/IoU.
- **Tracking**: ByteTrack or BoT-SORT via `model.track(persist=True)`; movement trails.
- **Analytics**: frames, detections, unique IDs, per-class counts, avg confidence, latency, inference FPS vs end-to-end FPS, timeline chart.
- **Benchmark**: runs selected models on *your* video, 3 warm-up frames discarded, real measured timings only.
- **Reports**: JSON, CSV (every detection row), PDF (reportlab).

## Quick start (local)
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_weights.py          # optional; otherwise downloaded on first use
flask --app wsgi run --port 8000            # dev
# production-style:
gunicorn -c gunicorn.conf.py wsgi:app
```
Open http://localhost:8000. Tests: `pytest`.

## Docker
```bash
docker compose up --build
```
Webcam inside Docker only works on Linux with `devices: ["/dev/video0:/dev/video0"]` (see `docker-compose.yml`).
Video upload and benchmark work everywhere. For GPU use a CUDA base image + NVIDIA container toolkit.

## Webcam: what to know
The camera is opened by the **server process** (`cv2.VideoCapture`), so it works when the app runs on the
machine that has the camera. A cloud deployment cannot see a visitor's camera (browser `getUserMedia` +
frame upload would be required). Common issues: camera busy in another app, OS camera permission denied for
Terminal/Python (macOS/Windows privacy settings), wrong camera index. The session auto-stops if the stream is
lost or no browser is watching for `WEBCAM_IDLE_TIMEOUT` seconds, and always releases the device.

## Tracking caveats
- *Detections* = boxes across all frames; *currently visible* = boxes in the latest frame;
  *unique track IDs* = distinct IDs issued by the tracker.
- IDs are not identities: they can change after long occlusions, fast motion, or tracker reset
  (each new webcam/video session starts a fresh tracker). Treat "unique IDs" as an upper-bound estimate of distinct objects.

## FPS definitions
- **Inference FPS** = frames / summed model time. **End-to-end FPS** = frames / wall-clock (decode + inference + drawing + encode).

## Configuration (env vars)
`MAX_UPLOAD_MB` (200), `JOB_TTL_SECONDS` (3600), `MAX_STORED_JOBS` (20), `WEBCAM_IDLE_TIMEOUT` (20),
`VISIONTRACK_DATA_DIR`, `VISIONTRACK_WEIGHTS_DIR`.

## Security & cleanup
Filenames via `secure_filename` (stored as `input.<ext>`), per-job isolated directories, `MAX_CONTENT_LENGTH`,
decoder-level validation, original upload deleted after processing, expired/excess jobs deleted on each request and at startup.

## Architecture / design notes
```
app/
  __init__.py        app factory, service wiring
  api.py             REST + MJPEG routes
  detector.py        ModelRegistry, DetectionSession (infer, track, draw trails)
  video_processor.py probe/validate, video job, benchmark job
  streaming.py       webcam capture thread + MJPEG generator
  analytics.py       StatsCollector (streams detections to CSV, O(1) memory per detection)
  jobs.py            Job/JobManager (single worker queue, TTL cleanup)
  reports.py         JSON/PDF reports
```
- **Single Gunicorn worker + threads**: job state, tracker state and the webcam session are in-process. For horizontal
  scale, move job state to Redis/DB and run workers separately (not included).
- One YOLO instance per session because tracker state lives on the predictor; weights are cached on disk.
- Versions in `requirements.txt` are ranges (written without network access); after your first install run
  `pip freeze > requirements.lock` and pin from that. Weight names (`yolo11*.pt`) require `ultralytics>=8.3`.

