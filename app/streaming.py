"""Server-side webcam capture with an MJPEG stream.

The camera is opened by the *server* process (cv2.VideoCapture), i.e. the machine running Flask.
This is the right model for local use; a remote/cloud deployment cannot see the visitor's browser
camera (that would need getUserMedia + frame upload, which is out of scope here).
"""
import threading
import time

import cv2

from .analytics import StatsCollector
from .detector import DetectionSession


class CameraError(RuntimeError):
    pass


class WebcamStream:
    MAX_READ_FAILURES = 30

    def __init__(self, registry, data_dir, idle_timeout=20):
        self.registry, self.data_dir, self.idle_timeout = registry, data_dir, idle_timeout
        self._lock = threading.Lock()
        self._cond = threading.Condition()
        self._thread = None
        self._stop = threading.Event()
        self._jpeg = None
        self.state, self.error, self.params = "idle", None, {}
        self.collector = None
        self.live_fps = 0.0
        self._last_client = 0.0

    # ---- lifecycle -------------------------------------------------------------------------
    def start(self, params: dict):
        with self._lock:
            if self.state == "running":
                raise CameraError("Webcam session already running.")
            model = self.registry.load(params["model"])           # may download weights on first use
            cap = cv2.VideoCapture(params["camera_index"])
            if not cap.isOpened():
                cap.release()
                raise CameraError("No camera found or access denied. Check that a camera is connected, "
                                  "not used by another app, and that the OS grants this process camera access.")
            ok, _ = cap.read()
            if not ok:
                cap.release()
                raise CameraError("Camera opened but returned no frames.")
            self.params = params
            self.session = DetectionSession(model, self.registry.device, params["conf"], params["iou"],
                                            params["tracking"], params["tracker"])
            self.collector = StatsCollector(self.data_dir / "detections.csv")
            self._stop = threading.Event()
            self.state, self.error, self._jpeg, self.live_fps = "running", None, None, 0.0
            self._last_client = time.time()
            self._thread = threading.Thread(target=self._loop, args=(cap,), daemon=True, name="vt-webcam")
            self._thread.start()

    def stop(self):
        with self._lock:
            t = self._thread
            self._stop.set()
        if t:
            t.join(timeout=5)
        with self._lock:
            if self.state == "running":
                self.state = "stopped"

    def _loop(self, cap):
        n, failures, t_start = 0, 0, time.perf_counter()
        try:
            while not self._stop.is_set():
                if time.time() - self._last_client > self.idle_timeout:
                    self.error = "Stopped automatically: no viewer connected."
                    break
                t0 = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    failures += 1
                    if failures >= self.MAX_READ_FAILURES:
                        self.error = "Camera stream lost (device disconnected?)."
                        self.state = "error"
                        break
                    time.sleep(0.05)
                    continue
                failures = 0
                ann, dets, infer_ms = self.session.process(frame, n, time.perf_counter() - t_start)
                loop_ms = (time.perf_counter() - t0) * 1000
                self.live_fps = 0.9 * self.live_fps + 0.1 * (1000 / loop_ms) if self.live_fps else 1000 / loop_ms
                self.collector.record(n, time.perf_counter() - t_start, dets, infer_ms, loop_ms)
                cv2.putText(ann, f"FPS {self.live_fps:.1f} | infer {infer_ms:.0f} ms | objects {len(dets)}",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
                ok, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self._cond:
                        self._jpeg = buf.tobytes()
                        self._cond.notify_all()
                n += 1
        except Exception as exc:
            self.error, self.state = f"Webcam loop failed: {exc}", "error"
        finally:
            cap.release()                      # always free the device
            self.collector.finish()
            if self.state == "running":
                self.state = "stopped"
            with self._cond:
                self._cond.notify_all()

    # ---- MJPEG -------------------------------------------------------------------------------
    def frames(self):
        while self.state == "running":
            self._last_client = time.time()
            with self._cond:
                self._cond.wait(timeout=2)
                jpeg = self._jpeg
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    def status(self):
        return {"state": self.state, "error": self.error, "params": self.params,
                "live_fps": round(self.live_fps, 2),
                "stats": self.collector.summary() if self.collector else None}
