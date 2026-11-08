"""TensorFlow-free copy of visionguard/preprocess.py for the Vercel API.

Same steps as the training/serving pipeline:
    bytes -> RGB float32 -> shorter side scaled to `size` (aspect ratio kept)
          -> centre square of size x size cut out, values kept in [0, 255]

Pillow decodes the image; the resize is an exact pixel-area average written in
numpy, which is what tf.image.resize(method="area") computes. The new size is
computed in float32 like TensorFlow does (in float64, 1600 * (224/1600) is
224.00000000000003 and would round up to 225). scripts/export_tflite.py --check
measures how closely this matches the TensorFlow version.
"""
import io

import numpy as np
from PIL import Image

SIZE = 224


def _area_matrix(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in) weights: each output pixel averages the input span it covers."""
    s = n_in / n_out
    m = np.zeros((n_out, n_in), np.float32)
    for o in range(n_out):
        a0, a1 = o * s, (o + 1) * s
        for i in range(int(np.floor(a0)), int(np.ceil(a1))):
            m[o, min(i, n_in - 1)] += min(i + 1, a1) - max(i, a0)
    return m / np.float32(s)


def area_resize(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    rows = _area_matrix(img.shape[0], out_h)
    cols = _area_matrix(img.shape[1], out_w)
    x = np.tensordot(rows, img, axes=(1, 0))                     # (out_h, w, 3)
    return np.tensordot(x, cols, axes=(1, 1)).transpose(0, 2, 1)  # (out_h, out_w, 3)


def prepare(image_bytes: bytes, size: int = SIZE) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as img:
        # like tf.io.decode_image(channels=3): alpha dropped, EXIF rotation ignored
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = np.float32(arr.shape[0]), np.float32(arr.shape[1])
    scale = np.float32(size) / min(h, w)
    new_h = max(int(np.ceil(h * scale)), size)
    new_w = max(int(np.ceil(w * scale)), size)

    arr = area_resize(arr, new_h, new_w)
    top, left = (new_h - size) // 2, (new_w - size) // 2
    arr = arr[top:top + size, left:left + size, :]
    return np.clip(arr, 0.0, 255.0).astype(np.float32)
