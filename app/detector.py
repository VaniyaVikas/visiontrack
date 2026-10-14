"""Model registry + detection/tracking session. Models are never loaded per frame or per request."""
import threading
import time
from collections import deque

import cv2
import numpy as np

ALLOWED_MODELS = {"yolo11n": "yolo11n.pt", "yolo11s": "yolo11s.pt", "yolo11m": "yolo11m.pt"}
ALLOWED_TRACKERS = {"bytetrack": "bytetrack.yaml", "botsort": "botsort.yaml"}


def resolve_device():
    import torch
    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelRegistry:
    """Validates model ids and loads YOLO weights (auto-download into weights dir on first use).

    Tracking keeps tracker state on the model's predictor, so every session gets its own
    YOLO instance; weights files are cached on disk, so this is cheap after the first load.
    """

    def __init__(self, weights_dir):
        self.weights_dir = weights_dir
        self._lock = threading.Lock()
        self._device = None

    @property
    def device(self):
        if self._device is None:
            self._device = resolve_device()
        return self._device

    @staticmethod
    def validate(model_id: str, tracker: str | None = None):
        if model_id not in ALLOWED_MODELS:
            raise ValueError(f"Unknown model '{model_id}'. Allowed: {', '.join(ALLOWED_MODELS)}")
        if tracker is not None and tracker not in ALLOWED_TRACKERS:
            raise ValueError(f"Unknown tracker '{tracker}'. Allowed: {', '.join(ALLOWED_TRACKERS)}")

    def load(self, model_id: str):
        self.validate(model_id)
        from ultralytics import YOLO
        from ultralytics.utils.downloads import attempt_download_asset
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            path = attempt_download_asset(str(self.weights_dir / ALLOWED_MODELS[model_id]))
            return YOLO(str(path))


_PALETTE = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
            (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
            (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0)]


def color_for(key: int):
    return _PALETTE[key % len(_PALETTE)]


class DetectionSession:
    """Runs detection (optionally with tracking) on frames and keeps recent trails per track ID."""

    def __init__(self, model, device, conf=0.25, iou=0.7, tracking=True,
                 tracker="bytetrack", imgsz=640, trail_len=30):
        self.model, self.device = model, device
        self.conf, self.iou, self.tracking = conf, iou, tracking
        self.tracker_cfg = ALLOWED_TRACKERS[tracker]
        self.imgsz, self.trail_len = imgsz, trail_len
        self.trails: dict[int, deque] = {}
        self._last_seen: dict[int, int] = {}

    def infer(self, frame):
        kw = dict(conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=self.device, verbose=False)
        t0 = time.perf_counter()
        if self.tracking:
            res = self.model.track(frame, persist=True, tracker=self.tracker_cfg, **kw)[0]
        else:
            res = self.model.predict(frame, **kw)[0]
        return res, (time.perf_counter() - t0) * 1000

    def process(self, frame, frame_no, timestamp_s):
        """Returns (annotated_frame, detections, inference_ms)."""
        res, infer_ms = self.infer(frame)
        dets = self._parse(res)
        return self.annotate(frame.copy(), dets, frame_no), dets, infer_ms

    @staticmethod
    def _parse(res):
        out, boxes = [], res.boxes
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(cls)
        for b, c, k, i in zip(xyxy, conf, cls, ids):
            out.append({"class_id": int(k), "class_name": str(res.names[int(k)]),
                        "confidence": float(c), "bbox": [float(v) for v in b],
                        "track_id": int(i) if i is not None else None})
        return out

    def annotate(self, frame, dets, frame_no=0):
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d["bbox"])
            tid = d.get("track_id")
            color = color_for(tid if tid is not None else d["class_id"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{d['class_name']} {d['confidence']:.2f}"
            if tid is not None:
                label = f"#{tid} " + label
                trail = self.trails.setdefault(tid, deque(maxlen=self.trail_len))
                trail.append((int((x1 + x2) / 2), int(y2)))
                self._last_seen[tid] = frame_no
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(y1 - th - base - 4, 0)
            cv2.rectangle(frame, (x1, top), (x1 + tw + 4, top + th + base + 4), color, -1)
            cv2.putText(frame, label, (x1 + 2, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for tid, trail in list(self.trails.items()):
            if frame_no - self._last_seen.get(tid, frame_no) > self.trail_len:  # forget stale tracks
                del self.trails[tid], self._last_seen[tid]
                continue
            if len(trail) > 1:
                cv2.polylines(frame, [np.array(trail, np.int32)], False, color_for(tid), 2, cv2.LINE_AA)
        return frame
