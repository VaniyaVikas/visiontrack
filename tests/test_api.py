import io

import cv2
import numpy as np


def _upload(client, data, name, **extra):
    return client.post("/api/videos", data={"video": (io.BytesIO(data), name), **extra},
                       content_type="multipart/form-data")


def test_healthz_and_models(client):
    assert client.get("/healthz").json == {"status": "ok"}
    assert "yolo11n" in client.get("/api/models").json["models"]


def test_rejects_bad_extension(client):
    r = _upload(client, b"x", "evil.exe")
    assert r.status_code == 400 and "Unsupported" in r.json["error"]


def test_rejects_corrupted_video(client):
    r = _upload(client, b"not a video at all" * 50, "bad.mp4")
    assert r.status_code == 400


def test_rejects_unknown_model_and_bad_threshold(client):
    assert _upload(client, b"x", "a.mp4", model="gpt").status_code == 400
    assert _upload(client, b"x", "a.mp4", conf="5").status_code == 400


def test_missing_file(client):
    assert client.post("/api/videos", data={}).status_code == 400


def test_file_too_large(client):
    assert _upload(client, b"0" * (3 * 1024 * 1024), "big.mp4").status_code == 413


def test_unknown_job_404(client):
    assert client.get("/api/videos/nope").status_code == 404


def test_webcam_idle_state(client):
    s = client.get("/api/webcam/status").json
    assert s["state"] == "idle" and s["stats"] is None
    assert client.get("/video_feed").status_code == 409
    assert client.get("/api/webcam/report.json").status_code == 404


def test_probe_valid_video(tmp_path):
    from app.video_processor import probe_video
    p = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for _ in range(5):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    info = probe_video(p)
    assert info["width"] == 64 and info["height"] == 48
