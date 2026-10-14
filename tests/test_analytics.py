import numpy as np
import pytest

from app.analytics import StatsCollector
from app.detector import DetectionSession


def det(cls, conf, tid=None):
    return {"class_id": 0, "class_name": cls, "confidence": conf, "bbox": [1, 2, 30, 40], "track_id": tid}


def test_metrics_from_measured_values(tmp_path):
    c = StatsCollector(tmp_path / "d.csv")
    c.record(0, 0.0, [det("person", 0.8, 1), det("car", 0.6, 2)], 10.0, 20.0)
    c.record(1, 0.5, [det("person", 0.9, 1)], 20.0, 30.0)
    c.finish()
    s = c.summary()
    assert s["frames"] == 2 and s["total_detections"] == 3
    assert s["unique_ids"] == 2 and s["current_visible"] == 1
    assert s["class_counts"] == {"person": 2, "car": 1}
    assert s["avg_confidence"] == pytest.approx((0.8 + 0.6 + 0.9) / 3)
    assert s["avg_inference_ms"] == pytest.approx(15.0)
    assert s["inference_fps"] == pytest.approx(2 / 0.030)   # model time only
    assert len((tmp_path / "d.csv").read_text().splitlines()) == 4


def test_empty_summary_has_no_fabricated_values():
    s = StatsCollector().summary()
    assert s["frames"] == 0 and s["avg_confidence"] is None and s["inference_fps"] is None


def test_untracked_detections_do_not_create_ids():
    c = StatsCollector()
    c.record(0, 0, [det("dog", 0.5)], 5, 6)
    assert c.summary()["unique_ids"] == 0


def test_annotate_draws_and_keeps_trails():
    s = DetectionSession(model=None, device="cpu", tracking=True)
    frame = np.zeros((100, 100, 3), np.uint8)
    for i in range(3):
        out = s.annotate(frame.copy(), [det("person", 0.9, 7)], i)
    assert out.any() and len(s.trails[7]) == 3


def test_registry_validation():
    from app.detector import ModelRegistry
    with pytest.raises(ValueError):
        ModelRegistry.validate("nope")
    with pytest.raises(ValueError):
        ModelRegistry.validate("yolo11n", "sort")
