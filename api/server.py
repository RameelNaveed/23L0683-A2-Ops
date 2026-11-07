"""Task 7: the model API that runs in the cloud (Google Cloud Run).

POST /predict   multipart form field "image", header "X-API-Key: <key>"  -> JSON verdict
GET  /health    no key needed                                           -> JSON status

Settings (environment variables): API_KEY, MODEL_PATH, THRESHOLD, MODEL_VERSION, MAX_UPLOAD_MB.
"""
import hmac
import logging
import os

from flask import Flask, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

from visionguard.predictor import LocalPredictor
from visionguard.validation import UploadError, validate_upload

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("visionguard-api")

API_KEY = os.getenv("API_KEY", "")
predictor = LocalPredictor(
    os.getenv("MODEL_PATH", "model/champion.keras"),
    float(os.getenv("THRESHOLD", "0.5")),
    os.getenv("MODEL_VERSION", "champion"),
)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(float(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024)
log.info("model loaded: %s", predictor.info())


def authorised() -> bool:
    sent = request.headers.get("X-API-Key", "")
    return bool(API_KEY) and hmac.compare_digest(sent, API_KEY)  # constant time compare


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": predictor.info()})


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
    return jsonify({"error": "file too large"}), 413
