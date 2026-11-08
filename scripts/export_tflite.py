"""Task 7: compress the champion Keras model to TensorFlow Lite for the Vercel API.

Vercel cannot run full TensorFlow (its Python functions are limited to ~250 MB),
so the cloud API uses the small LiteRT interpreter and a .tflite file instead.

    python scripts/export_tflite.py                 # writes models/champion_{fp32,fp16,int8}.tflite
    python scripts/export_tflite.py --check 148     # also compares them with Keras on the test split
"""
import argparse
import os
import tempfile
import time

import numpy as np
import pandas as pd
import tensorflow as tf

MODELS = "models"
VARIANTS = {
    "fp32": None,                                   # no compression, same numbers as Keras
    "fp16": "float16",                              # weights stored as float16 (~half size)
    "int8": "dynamic",                              # weights stored as int8 (~quarter size)
}


def convert(keras_path: str) -> dict:
    model = tf.keras.models.load_model(keras_path, compile=False)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        saved = os.path.join(tmp, "saved")
        model.export(saved)  # Keras 3 -> TF SavedModel (inference graph only)
        for name, mode in VARIANTS.items():
            conv = tf.lite.TFLiteConverter.from_saved_model(saved)
            if mode == "float16":
                conv.optimizations = [tf.lite.Optimize.DEFAULT]
                conv.target_spec.supported_types = [tf.float16]
            elif mode == "dynamic":
                conv.optimizations = [tf.lite.Optimize.DEFAULT]
            path = os.path.join(MODELS, f"champion_{name}.tflite")
            with open(path, "wb") as f:
                f.write(conv.convert())
            out[name] = path
    return out


def check(keras_path: str, paths: dict, n: int, threshold: float):
    """Compare Keras (TF preprocessing) with each .tflite (Pillow preprocessing)."""
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    sys.path[:0] = [root_dir, os.path.join(root_dir, "vercel_api")]
    from preprocess_pil import prepare as prepare_pil
    from visionguard.preprocess import prepare as prepare_tf

    model = tf.keras.models.load_model(keras_path, compile=False)
    size = int(model.input_shape[1])
    df = pd.read_csv("data/splits/test.csv").head(n)
    root = os.getenv("DATA_ROOT", "data/raw")

    interps = {}
    for name, p in paths.items():
        it = tf.lite.Interpreter(model_path=p)
        it.allocate_tensors()
        interps[name] = it

    keras_p, pil_p, lite_p = [], [], {k: [] for k in interps}
    for rel in df["file"]:
        data = open(os.path.join(root, rel), "rb").read()
        keras_p.append(float(model(prepare_tf(data, size)[None], training=False).numpy().ravel()[0]))
        x = prepare_pil(data, size)[None]
        pil_p.append(float(model(x, training=False).numpy().ravel()[0]))
        for name, it in interps.items():
            it.set_tensor(it.get_input_details()[0]["index"], x)
            it.invoke()
            lite_p[name].append(float(it.get_tensor(it.get_output_details()[0]["index"]).ravel()[0]))

    y = df["y"].to_numpy()
    k = np.array(keras_p)

    def row(label, p, size_mb=None):
        p = np.array(p)
        acc = ((p >= threshold) == y).mean()
        agree = ((p >= threshold) == (k >= threshold)).mean()
        mad = np.abs(p - k).max()
        s = f"{size_mb:6.2f} MB" if size_mb else "        "
        print(f"{label:28s} {s}  acc={acc:.4f}  same verdict as Keras={agree:.4f}  max|dp|={mad:.4f}")

    print(f"\n{len(df)} test images, threshold {threshold}")
    row("Keras + TF preprocess", keras_p, os.path.getsize(keras_path) / 1e6)
    row("Keras + Pillow preprocess", pil_p)
    for name, p in lite_p.items():
        row(f"TFLite {name} + Pillow", p, os.path.getsize(paths[name]) / 1e6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/champion.keras")
    ap.add_argument("--check", type=int, default=0, help="number of test images to compare (0 = skip)")
    ap.add_argument("--threshold", type=float, default=0.81)
    a = ap.parse_args()
    t0 = time.time()
    paths = convert(a.model)
    for name, p in paths.items():
        print(f"{p}: {os.path.getsize(p) / 1e6:.2f} MB")
    print(f"converted in {time.time() - t0:.1f}s")
    if a.check:
        check(a.model, paths, a.check, a.threshold)
