"""Task 7: the VisionGuard model API on Vercel (serverless Python + TensorFlow Lite).

POST /predict   multipart form field "image", header "X-API-Key: <key>"  -> JSON verdict
GET  /health    no key needed                                           -> JSON status

The response has the same fields as api/server.py, so visionguard.predictor.RemotePredictor
(used by the Flask app in Task 8) works with either API.

Settings (Vercel environment variables): API_KEY, THRESHOLD, MODEL_VERSION, MODEL_FILE, MAX_UPLOAD_MB.
"""
import hashlib
import hmac
import logging
import os
import sys
import threading
import time
from urllib.parse import parse_qs, urlencode

import numpy as np
from flask import Flask, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from preprocess_pil import prepare  # noqa: E402
from validation import UploadError, validate_upload  # noqa: E402

try:
    from ai_edge_litert.interpreter import Interpreter  # Linux/macOS (Vercel)
except ImportError:  # local Windows test with full TensorFlow installed
    from tensorflow.lite import Interpreter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("visionguard-api")

API_KEY = os.getenv("API_KEY", "")
THRESHOLD = float(os.getenv("THRESHOLD", "0.81"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "champion-v2-tflite-fp16")
MODEL_FILE = os.getenv("MODEL_FILE", "champion_fp16.tflite")
MODEL_PATH = os.path.join(ROOT, "model", MODEL_FILE)
# Vercel rejects request bodies over 4.5 MB before they reach this code.
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "4.5"))


class LitePredictor:
    """Loads the .tflite model once per warm instance and reuses it."""

    mode = "cloud-tflite"

    def __init__(self, path: str):
        t0 = time.time()
        self.interp = Interpreter(model_path=path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.img_size = int(self.inp["shape"][1])
        self.lock = threading.Lock()  # one interpreter, so one inference at a time
        self.load_seconds = round(time.time() - t0, 2)

    def predict_bytes(self, data: bytes) -> dict:
        t0 = time.time()
        x = prepare(data, self.img_size)[None, ...]
        with self.lock:
            self.interp.set_tensor(self.inp["index"], x)
            self.interp.invoke()
            prob = float(self.interp.get_tensor(self.out["index"]).ravel()[0])
        is_ai = prob >= THRESHOLD
        return {
            "label": "AI-generated" if is_ai else "Real",
            "prob_ai": round(prob, 4),
            "confidence": round(prob if is_ai else 1.0 - prob, 4),
            "threshold": THRESHOLD,
            "model_name": MODEL_FILE,
            "model_version": MODEL_VERSION,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def info(self) -> dict:
        return {"mode": self.mode, "model_name": MODEL_FILE, "model_version": MODEL_VERSION,
                "img_size": self.img_size, "threshold": THRESHOLD, "load_seconds": self.load_seconds}


class RestoreVercelPath:
    """vercel.json rewrites every URL to /api/index?__vg_path=/<original path>.
    Flask would only see /api/index, so put the original path back before routing."""

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        query = parse_qs(environ.get("QUERY_STRING", ""), keep_blank_values=True)
        if "__vg_path" in query:
            environ["PATH_INFO"] = query.pop("__vg_path")[0] or "/"
            environ["QUERY_STRING"] = urlencode(query, doseq=True)
        return self.wsgi_app(environ, start_response)


predictor = LitePredictor(MODEL_PATH)
app = Flask(__name__)
app.wsgi_app = RestoreVercelPath(app.wsgi_app)
app.config["MAX_CONTENT_LENGTH"] = int(MAX_UPLOAD_MB * 1024 * 1024)
log.info("model loaded: %s", predictor.info())


def authorised() -> bool:
    sent = request.headers.get("X-API-Key", "")
    return bool(API_KEY) and hmac.compare_digest(sent, API_KEY)  # constant time compare


@app.get("/")
def root():
    return jsonify({"service": "VisionGuard model API", "endpoints": ["GET /health", "POST /predict"]})


@app.get("/health")
def health():
    return jsonify({"status": "ok", "api_key_configured": bool(API_KEY), "model": predictor.info()})


@app.post("/predict")
def predict():
    if not authorised():
        return jsonify({"error": "missing or wrong X-API-Key header"}), 401
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "send the image as multipart form field 'image'"}), 400
    data = file.read()
    try:
        name = validate_upload(file.filename, data)
    except UploadError as e:
        return jsonify({"error": e.message}), e.status
    result = predictor.predict_bytes(data)
    result["filename"] = name
    log.info("predict file=%s label=%s conf=%.3f", name, result["label"], result["confidence"])
    return jsonify(result)


@app.errorhandler(RequestEntityTooLarge)
def too_large(_e):
    return jsonify({"error": f"file too large (limit {MAX_UPLOAD_MB:g} MB)"}), 413
