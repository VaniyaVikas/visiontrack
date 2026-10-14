import os
from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = Path(os.getenv("VISIONTRACK_DATA_DIR", BASE_DIR / "instance"))
    WEIGHTS_DIR = Path(os.getenv("VISIONTRACK_WEIGHTS_DIR", BASE_DIR / "weights"))
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024
    ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
    JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "3600"))
    MAX_STORED_JOBS = int(os.getenv("MAX_STORED_JOBS", "20"))
    WEBCAM_IDLE_TIMEOUT = int(os.getenv("WEBCAM_IDLE_TIMEOUT", "20"))
    DEFAULT_MODEL = "yolo11n"
    DEFAULT_TRACKER = "bytetrack"
    BENCHMARK_MAX_FRAMES = 300
