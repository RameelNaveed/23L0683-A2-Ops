"""Task 4.1 inference module.

LocalPredictor loads the Keras model ONCE when it is created (at app start-up)
and reuses it for every request. RemotePredictor (Task 8) sends the image to
the model API hosted in the cloud and returns the same kind of answer.
"""
import hashlib
import os
import time

LABEL_AI = "AI-generated"
LABEL_REAL = "Real"


def verdict(prob_ai: float, threshold: float) -> dict:
    is_ai = prob_ai >= threshold
    confidence = prob_ai if is_ai else 1.0 - prob_ai
    return {
        "label": LABEL_AI if is_ai else LABEL_REAL,
        "prob_ai": round(float(prob_ai), 4),
        "confidence": round(float(confidence), 4),
        "threshold": float(threshold),
    }


class LocalPredictor:
    mode = "local"

    def __init__(self, model_path: str, threshold: float = 0.5, model_version: str = "unknown"):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        import tensorflow as tf  # imported here so tests can run without TensorFlow
        from visionguard.preprocess import prepare

        self._tf = tf
        self._prepare = prepare
        t0 = time.time()
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.img_size = int(self.model.input_shape[1])
        self.threshold = float(threshold)
        self.model_name = os.path.basename(model_path)
        self.model_version = model_version
        self.load_seconds = round(time.time() - t0, 2)
        # One warm-up call so the first real request is not slow.
        self.model(tf.zeros([1, self.img_size, self.img_size, 3]), training=False)

    def predict_bytes(self, data: bytes) -> dict:
        t0 = time.time()
        x = self._prepare(data, self.img_size)[None, ...]
        prob = float(self.model(x, training=False).numpy().ravel()[0])
        out = verdict(prob, self.threshold)
        out.update(
            model_name=self.model_name,
            model_version=self.model_version,
            mode=self.mode,
            latency_ms=round((time.time() - t0) * 1000, 1),
            sha256=hashlib.sha256(data).hexdigest(),
        )
        return out

    def info(self) -> dict:
        return {
            "mode": self.mode,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "img_size": self.img_size,
            "threshold": self.threshold,
            "load_seconds": self.load_seconds,
        }


class RemotePredictor:
    """Calls the cloud model API (Task 7) from the Flask app (Task 8)."""

    mode = "remote"

    def __init__(self, api_url: str, api_key: str, timeout: float = 60.0):
        if not api_url:
            raise ValueError("MODEL_API_URL is empty but INFERENCE_MODE=remote")
        import requests

        self._requests = requests
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.model_name = "remote-api"
        self.model_version = "see API"

    def predict_bytes(self, data: bytes, filename: str = "upload.png") -> dict:
        t0 = time.time()
        resp = self._requests.post(
            f"{self.api_url}/predict",
            headers={"X-API-Key": self.api_key},
            files={"image": (filename, data)},
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Model API returned {resp.status_code}: {resp.text[:200]}")
        out = resp.json()
        out.update(
            mode=self.mode,
            latency_ms=round((time.time() - t0) * 1000, 1),
            sha256=hashlib.sha256(data).hexdigest(),
        )
        self.model_name = out.get("model_name", self.model_name)
        self.model_version = out.get("model_version", self.model_version)
        return out

    def info(self) -> dict:
        try:
            r = self._requests.get(f"{self.api_url}/health", timeout=10)
            remote = r.json()
        except Exception as exc:  # noqa: BLE001 - shown on the health page only
            remote = {"error": str(exc)}
        return {"mode": self.mode, "api_url": self.api_url, "api_health": remote}
