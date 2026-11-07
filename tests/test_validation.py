import pytest

from tests.conftest import png_bytes
from visionguard.predictor import verdict
from visionguard.validation import UploadError, validate_upload


def test_validate_ok():
    assert validate_upload("a.png", png_bytes()) == "a.png"


def test_validate_empty():
    with pytest.raises(UploadError) as e:
        validate_upload("a.png", b"")
    assert e.value.status == 400


def test_verdict_threshold():
    assert verdict(0.7, 0.6)["label"] == "AI-generated"
    assert verdict(0.5, 0.6)["label"] == "Real"
    assert verdict(0.2, 0.5)["confidence"] == 0.8
