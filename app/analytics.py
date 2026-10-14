
import csv
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

CSV_FIELDS = ["frame", "timestamp_s", "class_id", "class_name", "confidence",
              "x1", "y1", "x2", "y2", "track_id"]


class StatsCollector:
    def __init__(self, csv_path: Path | None = None):
        self._lock = threading.Lock()
        self.frames = 0
        self.total_detections = 0
        self.class_counts: Counter = Counter()
        self.unique_ids: set = set()
        self.unique_by_class: dict[str, set] = defaultdict(set)
        self.conf_sum = 0.0
        self.infer_ms_sum = 0.0
        self.e2e_ms_sum = 0.0
        self.current_visible = 0
        self._per_second: Counter = Counter()
        self._started = time.perf_counter()
        self._ended: float | None = None
        self.csv_path = Path(csv_path) if csv_path else None
        self._fh = self._writer = None
        if self.csv_path:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=CSV_FIELDS)
            self._writer.writeheader()

    def record(self, frame_no, timestamp_s, detections, infer_ms, e2e_ms):
        with self._lock:
            self.frames += 1
            self.infer_ms_sum += infer_ms
            self.e2e_ms_sum += e2e_ms
            self.current_visible = len(detections)
            self.total_detections += len(detections)
            self._per_second[int(timestamp_s)] += len(detections)
            for d in detections:
                self.class_counts[d["class_name"]] += 1
                self.conf_sum += d["confidence"]
                if d.get("track_id") is not None:
                    self.unique_ids.add(d["track_id"])
                    self.unique_by_class[d["class_name"]].add(d["track_id"])
                if self._writer:
                    x1, y1, x2, y2 = d["bbox"]
                    self._writer.writerow({
                        "frame": frame_no, "timestamp_s": round(timestamp_s, 3),
                        "class_id": d["class_id"], "class_name": d["class_name"],
                        "confidence": round(d["confidence"], 4),
                        "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1),
                        "track_id": d.get("track_id"),
                    })

    def finish(self):
        with self._lock:
            self._ended = self._ended or time.perf_counter()
            if self._fh:
                self._fh.flush()
                self._fh.close()
                self._fh = self._writer = None

    def _timeline(self, max_points=120):
        if not self._per_second:
            return []
        last = max(self._per_second)
        step = max(1, -(-(last + 1) // max_points))
        buckets = Counter()
        for sec, n in self._per_second.items():
            buckets[sec // step * step] += n
        return [{"t": t, "count": buckets.get(t, 0)} for t in range(0, last + 1, step)]

    def summary(self) -> dict:
        with self._lock:
            end = self._ended or time.perf_counter()
            wall = max(end - self._started, 1e-9)
            f = self.frames
            return {
                "frames": f,
                "total_detections": self.total_detections,
                "unique_ids": len(self.unique_ids),
                "current_visible": self.current_visible,
                "class_counts": dict(self.class_counts),
                "unique_by_class": {k: len(v) for k, v in self.unique_by_class.items()},
                "avg_confidence": self.conf_sum / self.total_detections if self.total_detections else None,
                "avg_inference_ms": self.infer_ms_sum / f if f else None,
                # Inference FPS: model time only. Processing FPS: wall-clock, includes decode/draw/encode.
                "inference_fps": f / (self.infer_ms_sum / 1000) if self.infer_ms_sum > 0 else None,
                "processing_fps": f / wall if f else None,
                "duration_s": wall,
                "timeline": self._timeline(),
            }
