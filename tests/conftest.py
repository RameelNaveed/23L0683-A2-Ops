import io

import pytest
from PIL import Image

from app import create_app
from app.config import Settings


class FakePredictor:
    """Stands in for the real model so tests run in seconds without TensorFlow."""
    mode = "local"

    def predict_bytes(self, data, filename=None):
        return {"label": "AI-generated", "prob_ai": 0.9, "confidence": 0.9, "threshold": 0.5,
                "model_name": "fake.keras", "model_version": "test", "mode": "local",
                "latency_ms": 1.0, "sha256": "0" * 64}

    def info(self):
        return {"mode": "local", "model_name": "fake.keras"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    app = create_app(predictor=FakePredictor(), settings=Settings())
    app.config["TESTING"] = True
    return app.test_client()


def png_bytes(size=(32, 32)):
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 30, 200)).save(buf, format="PNG")
    return buf.getvalue()
