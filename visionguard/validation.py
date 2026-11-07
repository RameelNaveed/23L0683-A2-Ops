"""Upload checks shared by the Flask app and the model API.

Uses Pillow only, so it works even where TensorFlow is not installed.
"""
import io
import os

from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}


class UploadError(Exception):
    """Raised for a bad upload. status is the HTTP code to send back."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.message = message
        self.status = status


def clean_filename(name: str) -> str:
    cleaned = secure_filename(name or "")
    return cleaned or "upload"


def validate_upload(filename: str, data: bytes) -> str:
    """Return the sanitized file name, or raise UploadError."""
    name = clean_filename(filename)
    ext = os.path.splitext(name)[1].lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadError(
            f"File type '.{ext or '?'}' is not allowed. Upload a JPG or PNG image.", 415
        )
    if not data:
        raise UploadError("The uploaded file is empty.", 400)
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()  # checks the file structure
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format
            img.load()  # actually decodes the pixels
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise UploadError(
            "This file has an image extension but could not be decoded as an image.", 422
        ) from exc
    if fmt not in ALLOWED_FORMATS:
        raise UploadError(f"Image format {fmt} is not supported. Use JPG or PNG.", 415)
    return name
