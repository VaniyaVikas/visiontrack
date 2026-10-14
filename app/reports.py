import io
import json
from datetime import datetime, timezone


def build_report(source: dict, params: dict, stats: dict | None) -> dict:
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "source": source,
            "parameters": params, "summary": stats,
            "notes": ["Unique track IDs are per-session identifiers; they can change after long "
                      "occlusions or tracker resets and are not guaranteed identities.",
                      "inference_fps uses model time only; processing_fps is wall-clock end-to-end."]}


def report_json(report: dict) -> bytes:
    return json.dumps(report, indent=2).encode()


def report_pdf(report: dict) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise RuntimeError("PDF export needs 'reportlab' (pip install reportlab).") from exc
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 800
    c.setFont("Helvetica-Bold", 16); c.drawString(50, y, "VisionTrack AI - Analysis Report"); y -= 30
    c.setFont("Helvetica", 10)
    s = report.get("summary") or {}

    def fmt(v, nd=2):
        return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    lines = [f"Generated: {report['generated_at']}",
             f"Model: {report['parameters'].get('model')}   Tracker: "
             f"{report['parameters'].get('tracker') if report['parameters'].get('tracking') else 'off'}",
             f"Confidence: {report['parameters'].get('conf')}   IoU: {report['parameters'].get('iou')}", "",
             f"Frames processed: {fmt(s.get('frames'))}", f"Total detections: {fmt(s.get('total_detections'))}",
             f"Unique track IDs observed: {fmt(s.get('unique_ids'))}",
             f"Average confidence: {fmt(s.get('avg_confidence'), 3)}",
             f"Average inference latency (ms): {fmt(s.get('avg_inference_ms'))}",
             f"Inference FPS: {fmt(s.get('inference_fps'))}   End-to-end FPS: {fmt(s.get('processing_fps'))}",
             f"Processing duration (s): {fmt(s.get('duration_s'))}", "", "Detections by class:"]
    lines += [f"  {k}: {v}" for k, v in sorted((s.get("class_counts") or {}).items(), key=lambda kv: -kv[1])]
    lines += ["", "Note: track IDs may change after long occlusions or tracker resets."]
    for line in lines:
        if y < 60:
            c.showPage(); y = 800; c.setFont("Helvetica", 10)
        c.drawString(50, y, line); y -= 15
    c.save()
    return buf.getvalue()
