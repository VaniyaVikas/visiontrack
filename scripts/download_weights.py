"""Pre-download YOLO weights into ./weights (run once while online)."""
from pathlib import Path
from ultralytics.utils.downloads import attempt_download_asset
from app.detector import ALLOWED_MODELS

dest = Path(__file__).resolve().parent.parent / "weights"
dest.mkdir(exist_ok=True)
for name in ALLOWED_MODELS.values():
    print(attempt_download_asset(str(dest / name)))
