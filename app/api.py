import cv2  # noqa: F401  (ensures a clear import error early if OpenCV is missing)
from flask import (Blueprint, Response, current_app, jsonify, render_template, request,
                   send_file)
from werkzeug.utils import secure_filename

from .detector import ALLOWED_MODELS, ALLOWED_TRACKERS
from .reports import build_report, report_json, report_pdf
from .streaming import CameraError
from .video_processor import probe_video, run_benchmark_job, run_video_job

bp = Blueprint("api", __name__)


def svc():
    return current_app.extensions["vt"]


def err(msg, code=400):
    return jsonify(error=msg), code


@bp.errorhandler(ValueError)
def _value_error(e):
    return err(str(e), 400)


def _num(form, key, default, lo, hi):
    try:
        v = float(form.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number.")
    if not lo <= v <= hi:
        raise ValueError(f"'{key}' must be between {lo} and {hi}.")
    return v


def parse_params(form) -> dict:
    cfg = current_app.config
    model = form.get("model", cfg["DEFAULT_MODEL"])
    tracker = form.get("tracker", cfg["DEFAULT_TRACKER"])
    svc().registry.validate(model, tracker)
    return {"model": model, "tracker": tracker,
            "tracking": str(form.get("tracking", "true")).lower() in ("1", "true", "on", "yes"),
            "conf": _num(form, "conf", 0.25, 0.05, 0.95), "iou": _num(form, "iou", 0.7, 0.1, 0.95),
            "camera_index": int(_num(form, "camera_index", 0, 0, 10))}


def save_video(file, directory):
    name = secure_filename(file.filename or "")
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in current_app.config["ALLOWED_EXTENSIONS"]:
        raise ValueError("Unsupported file type. Allowed: "
                         + ", ".join(sorted(current_app.config["ALLOWED_EXTENSIONS"])))
    path = directory / f"input.{ext}"
    file.save(path)
    if path.stat().st_size == 0:
        raise ValueError("Uploaded file is empty.")
    probe_video(path)    # raises ValueError for corrupted/unreadable video
    return path


# ---- pages / meta ---------------------------------------------------------------------------
@bp.get("/")
def index():
    return render_template("index.html")


@bp.get("/healthz")
def healthz():
    return jsonify(status="ok")


@bp.get("/api/models")
def models():
    return jsonify(models=list(ALLOWED_MODELS), trackers=list(ALLOWED_TRACKERS),
                   defaults={"model": current_app.config["DEFAULT_MODEL"],
                             "tracker": current_app.config["DEFAULT_TRACKER"]},
                   extensions=sorted(current_app.config["ALLOWED_EXTENSIONS"]),
                   max_upload_mb=current_app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024))


# ---- webcam ---------------------------------------------------------------------------------
@bp.post("/api/webcam/start")
def webcam_start():
    params = parse_params(request.form or request.get_json(silent=True) or {})
    try:
        svc().stream.start(params)
    except CameraError as e:
        return err(str(e), 409)
    except Exception as e:
        return err(f"Could not start webcam session: {e}", 500)
    return jsonify(svc().stream.status())


@bp.post("/api/webcam/stop")
def webcam_stop():
    svc().stream.stop()
    return jsonify(svc().stream.status())


@bp.get("/api/webcam/status")
def webcam_status():
    return jsonify(svc().stream.status())


@bp.get("/video_feed")
def video_feed():
    if svc().stream.state != "running":
        return err("Webcam is not running.", 409)
    return Response(svc().stream.frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@bp.get("/api/webcam/report.<fmt>")
def webcam_report(fmt):
    s = svc().stream
    if s.collector is None:
        return err("No webcam session has been recorded yet.", 404)
    return _report_response(fmt, {"type": "webcam"}, s.params, s.collector.summary(), s.collector.csv_path)


# ---- video upload ---------------------------------------------------------------------------
@bp.post("/api/videos")
def upload_video():
    s = svc()
    s.jobs.cleanup()
    f = request.files.get("video")
    if not f or not f.filename:
        return err("No video file provided (form field 'video').")
    params = parse_params(request.form)
    job = s.jobs.new("video", params)
    try:
        job.src = save_video(f, job.dir)
    except Exception:
        s.jobs.discard(job.id)
        raise
    s.jobs.submit(job, run_video_job, s.registry)
    return jsonify(job.to_dict()), 202


def _job_or_404(job_id, kind):
    job = svc().jobs.get(job_id)
    return job if job and job.kind == kind else None


@bp.get("/api/videos/<job_id>")
def video_status(job_id):
    job = _job_or_404(job_id, "video")
    return jsonify(job.to_dict()) if job else err("Job not found or expired.", 404)


@bp.post("/api/videos/<job_id>/cancel")
def video_cancel(job_id):
    job = _job_or_404(job_id, "video")
    if not job:
        return err("Job not found or expired.", 404)
    job.cancel.set()
    return jsonify(job.to_dict())


@bp.get("/api/videos/<job_id>/output")
def video_output(job_id):
    job = _job_or_404(job_id, "video")
    if not job or not job.output or not job.output.exists():
        return err("Output not available.", 404)
    return send_file(job.output, mimetype="video/mp4", conditional=True,
                     as_attachment=request.args.get("download") == "1",
                     download_name=f"annotated_{job.id}.mp4")


@bp.get("/api/videos/<job_id>/report.<fmt>")
def video_report(job_id, fmt):
    job = _job_or_404(job_id, "video")
    if not job or not job.collector:
        return err("Job not found or expired.", 404)
    return _report_response(fmt, {"type": "video", "job_id": job.id, **job.video_info},
                            job.params, job.collector.summary(), job.csv)


def _report_response(fmt, source, params, stats, csv_path):
    report = build_report(source, params, stats)
    if fmt == "json":
        return Response(report_json(report), mimetype="application/json",
                        headers={"Content-Disposition": "attachment; filename=report.json"})
    if fmt == "csv":
        if not csv_path or not csv_path.exists():
            return err("Detections CSV not available.", 404)
        return send_file(csv_path, mimetype="text/csv", as_attachment=True, download_name="detections.csv")
    if fmt == "pdf":
        try:
            return Response(report_pdf(report), mimetype="application/pdf",
                            headers={"Content-Disposition": "attachment; filename=report.pdf"})
        except RuntimeError as e:
            return err(str(e), 501)
    return err("Unsupported report format. Use json, csv or pdf.", 400)


# ---- benchmark ------------------------------------------------------------------------------
@bp.post("/api/benchmark")
def start_benchmark():
    s = svc()
    s.jobs.cleanup()
    models = request.form.getlist("models") or list(ALLOWED_MODELS)
    for m in models:
        s.registry.validate(m)
    params = {"models": models, "conf": _num(request.form, "conf", 0.25, 0.05, 0.95),
              "iou": _num(request.form, "iou", 0.7, 0.1, 0.95), "imgsz": 640,
              "max_frames": int(_num(request.form, "max_frames", 100, 10,
                                     current_app.config["BENCHMARK_MAX_FRAMES"]))}
    f = request.files.get("video")
    if not f or not f.filename:
        return err("Upload a video to benchmark on (form field 'video').")
    job = s.jobs.new("benchmark", params)
    try:
        job.src = save_video(f, job.dir)
    except Exception:
        s.jobs.discard(job.id)
        raise
    s.jobs.submit(job, run_benchmark_job, s.registry)
    return jsonify(job.to_dict()), 202


@bp.get("/api/benchmark/<job_id>")
def benchmark_status(job_id):
    job = _job_or_404(job_id, "benchmark")
    return jsonify(job.to_dict()) if job else err("Job not found or expired.", 404)
