import io

from tests.conftest import png_bytes


def upload(client, data, name):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)},
                       content_type="multipart/form-data")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["database"] is True


def test_valid_upload_is_logged(client):
    r = upload(client, png_bytes(), "cat.png")
    assert r.status_code == 200
    assert b"AI-generated" in r.data
    h = client.get("/history")
    assert b"cat.png" in h.data


def test_wrong_type_rejected(client):
    r = upload(client, b"hello", "notes.txt")
    assert r.status_code == 415


def test_too_large_rejected(client):
    r = upload(client, b"0" * (2 * 1024 * 1024), "big.png")
    assert r.status_code == 413


def test_undecodable_image_rejected(client):
    r = upload(client, b"\x89PNG\r\n\x1a\n" + b"garbage" * 50, "broken.png")
    assert r.status_code == 422


def test_filename_sanitized(client):
    r = upload(client, png_bytes(), "../../etc/pass wd.png")
    assert r.status_code == 200
    assert b"etc_pass_wd.png" in r.data
