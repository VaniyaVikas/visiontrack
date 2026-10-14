import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app  # noqa: E402


class FakeRegistry:
    """Avoids loading real YOLO weights in unit tests."""
    device = "cpu"

    @staticmethod
    def validate(model_id, tracker=None):
        from app.detector import ModelRegistry
        ModelRegistry.validate(model_id, tracker)

    def load(self, model_id):
        raise RuntimeError("model loading disabled in tests")


@pytest.fixture()
def client(tmp_path):
    app = create_app({"DATA_DIR": tmp_path, "TESTING": True, "MAX_CONTENT_LENGTH": 2 * 1024 * 1024},
                     registry=FakeRegistry())
    return app.test_client()
