"""VisionGuard Flask app (Tasks 4, 5 and 8)."""
import logging

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from app.database import Store
from app.config import Settings
from visionguard.validation import UploadError, validate_upload

CAVEAT = ("This is a statistical guess from a model trained on a small public dataset. "
          "It can be wrong, especially on edited, compressed, very small or unusual images. "
          "Do not use it as the only evidence that an image is fake or real.")

log = logging.getLogger("visionguard")


def build_predictor(s: Settings):
    from visionguard.predictor import LocalPredictor, RemotePredictor
    if s.inference_mode == "remote":
        return RemotePredictor(s.model_api_url, s.model_api_key)
    return LocalPredictor(s.model_path, s.threshold, s.model_version)


def create_app(predictor=None, settings: Settings | None = None) -> Flask:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = settings or Settings()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(s.max_upload_mb * 1024 * 1024)  # Flask rejects bigger bodies with 413
    app.config["SECRET_KEY"] = s.secret_key

    # Loaded ONCE here at start-up, never per request.
    app.predictor = predictor or build_predictor(s)
    app.store = Store(s.database_url)
    log.info("VisionGuard ready: %s", s.safe_dict())

    def error_page(message, status):
        return render_template("error.html", message=message, status=status), status

    @app.get("/")
    def index():
        return render_template("index.html", max_mb=s.max_upload_mb, mode=s.inference_mode)

    @app.post("/predict")
    def predict():
        file = request.files.get("image")
        if file is None or file.filename == "":
            return error_page("Please choose an image file before pressing Analyse.", 400)
        data = file.read()  # kept in memory only, never written to disk
        try:
            name = validate_upload(file.filename, data)
        except UploadError as e:
            return error_page(e.message, e.status)
        try:
            if s.inference_mode == "remote":
                result = app.predictor.predict_bytes(data, name)
            else:
                result = app.predictor.predict_bytes(data)
        except Exception as e:  # model or API failure
            log.exception("prediction failed")
            return error_page(f"The model could not process this image right now ({type(e).__name__}).", 502)
        row_id = app.store.add(name, result)
        log.info("prediction id=%s file=%s verdict=%s conf=%.3f", row_id, name, result["label"], result["confidence"])
        return render_template("result.html", r=result, filename=name, row_id=row_id, caveat=CAVEAT)

    @app.get("/history")
    def history():
        return render_template("history.html", rows=app.store.recent(100), stats=app.store.stats())

    @app.get("/health")
    def health():
        status, db_ok = 200, True
        try:
            app.store.ping()
        except Exception as e:  # noqa: BLE001
            db_ok, status = f"error: {type(e).__name__}", 503
        body = {"status": "ok" if status == 200 else "degraded", "database": db_ok,
                "model": app.predictor.info(), "settings": s.safe_dict()}
        return jsonify(body), status

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_e):
        return error_page(f"File is too large. The limit is {s.max_upload_mb:g} MB.", 413)

    @app.errorhandler(HTTPException)
    def http_error(e):
        return error_page(e.description, e.code)

    @app.errorhandler(Exception)
    def server_error(e):
        log.exception("unhandled error")
        return error_page("Something went wrong on our side. Please try again.", 500)

    return app
