"""Frame-by-frame video processing and model benchmarking (never loads a whole video into RAM)."""
import shutil
import subprocess
import time
from pathlib import Path

import cv2

from .analytics import StatsCollector
from .detector import DetectionSession


def probe_video(path: Path) -> dict:
    """Validate that OpenCV can actually decode the file; raises ValueError otherwise."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Video could not be opened (corrupted file or unsupported codec).")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError("Video contains no readable frames (empty or unsupported codec).")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or fps > 240:
            fps = 25.0
        h, w = frame.shape[:2]
        return {"fps": round(fps, 2), "total_frames": max(total, 0), "width": w, "height": h,
                "duration_s": round(total / fps, 2) if total > 0 else 0.0}
    finally:
        cap.release()


def _transcode_h264(src: Path, dst: Path) -> bool:
    """Browsers can't play OpenCV's mp4v output reliably; convert to H.264 if ffmpeg is present."""
    if not shutil.which("ffmpeg"):
        return False
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)]
    return subprocess.run(cmd, capture_output=True, timeout=3600).returncode == 0 and dst.exists()


def run_video_job(job, registry):
    p = job.params
    job.status = "processing"
    cap = writer = collector = None
    try:
        info = probe_video(job.src)
        job.video_info, job.total_frames, job.duration_s = info, info["total_frames"], info["duration_s"]
        model = registry.load(p["model"])
        session = DetectionSession(model, registry.device, p["conf"], p["iou"], p["tracking"], p["tracker"])
        job.csv = job.dir / "detections.csv"
        collector = job.collector = StatsCollector(job.csv)
        raw = job.dir / "raw.mp4"
        cap = cv2.VideoCapture(str(job.src))
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"],
                                 (info["width"], info["height"]))
        if not writer.isOpened():
            raise RuntimeError("Could not create output video writer.")
        n = 0
        while not job.cancel.is_set():
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            annotated, dets, infer_ms = session.process(frame, n, n / info["fps"])
            writer.write(annotated)
            collector.record(n, n / info["fps"], dets, infer_ms, (time.perf_counter() - t0) * 1000)
            n += 1
            job.processed_frames = n
        writer.release(); writer = None
        if n == 0:
            raise RuntimeError("No frames could be processed.")
        final = job.dir / "annotated.mp4"
        if not _transcode_h264(raw, final):
            raw.replace(final)   # fallback: mp4v (may not preview in some browsers; still downloadable)
        raw.unlink(missing_ok=True)
        job.output = final
        job.total_frames = n
        job.status = "completed"
    except Exception as exc:
        job.status, job.error = "failed", str(exc)
    finally:
        if cap: cap.release()
        if writer: writer.release()
        if collector: collector.finish()
        if job.src and job.status != "processing":
            job.src.unlink(missing_ok=True)   # original upload is no longer needed


def run_benchmark_job(job, registry):
    """Measures every selected model on the same frames. All numbers come from real timings."""
    p = job.params
    job.status = "processing"
    try:
        info = probe_video(job.src)
        job.video_info = info
        max_frames = p["max_frames"]
        job.total_frames = max_frames * len(p["models"])
        for model_id in p["models"]:
            model = registry.load(model_id)
            session = DetectionSession(model, registry.device, p["conf"], p["iou"],
                                       tracking=False, imgsz=p["imgsz"])
            cap = cv2.VideoCapture(str(job.src))
            infer_ms, e2e_ms, dets_n, n, warm = [], [], 0, 0, 3
            wall0 = None
            try:
                while n < max_frames + warm and not job.cancel.is_set():
                    t0 = time.perf_counter()
                    ok, frame = cap.read()
                    if not ok:
                        break
                    ann, dets, ims = session.process(frame, n, 0)
                    total = (time.perf_counter() - t0) * 1000
                    n += 1
                    if n <= warm:     # discard warm-up frames (lazy init, cudnn autotune, ...)
                        wall0 = time.perf_counter()
                        continue
                    infer_ms.append(ims); e2e_ms.append(total); dets_n += len(dets)
                    job.processed_frames += 1
            finally:
                cap.release()
            if not infer_ms:
                raise RuntimeError(f"Not enough frames to benchmark {model_id}.")
            wall = time.perf_counter() - wall0
            job.results.append({
                "model": model_id, "frames": len(infer_ms), "device": str(registry.device),
                "avg_inference_ms": sum(infer_ms) / len(infer_ms),
                "inference_fps": 1000 * len(infer_ms) / sum(infer_ms),
                "end_to_end_fps": len(infer_ms) / wall,
                "avg_detections_per_frame": dets_n / len(infer_ms),
            })
        job.status = "completed"
    except Exception as exc:
        job.status, job.error = "failed", str(exc)
