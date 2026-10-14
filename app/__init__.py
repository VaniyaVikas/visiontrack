from pathlib import Path
from types import SimpleNamespace

from flask import Flask, jsonify

from .config import Config
from .detector import ModelRegistry
from .jobs import JobManager
from .streaming import WebcamStream


def create_app(overrides: dict | None = None, registry=None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    if overrides:
        app.config.update(overrides)
    data_dir = Path(app.config["DATA_DIR"])
    data_dir.mkdir(parents=True, exist_ok=True)

    registry = registry or ModelRegistry(Path(app.config["WEIGHTS_DIR"]))
    jobs = JobManager(data_dir / "jobs", app.config["JOB_TTL_SECONDS"], app.config["MAX_STORED_JOBS"])
    stream = WebcamStream(registry, data_dir / "webcam", app.config["WEBCAM_IDLE_TIMEOUT"])
    app.extensions["vt"] = SimpleNamespace(registry=registry, jobs=jobs, stream=stream)

    from .api import bp
    app.register_blueprint(bp)

    @app.errorhandler(413)
    def too_large(_):
        mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return jsonify(error=f"File too large (limit {mb} MB)."), 413

    return app
