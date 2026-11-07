"""Task 3.6: load the registered model by its alias and classify one image.

Usage:
    python scripts/predict_by_alias.py path\\to\\image.jpg
    python scripts/predict_by_alias.py path\\to\\image.jpg --alias champion
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mlflow  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from visionguard.predictor import verdict  # noqa: E402
from visionguard.preprocess import prepare  # noqa: E402

load_dotenv(ROOT / ".env")
cli = argparse.ArgumentParser()
cli.add_argument("image")
cli.add_argument("--alias", default="champion")
cli.add_argument("--local", action="store_true", help="use the local mlflow.db instead of DagsHub")
args = cli.parse_args()

if args.local or not os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
name = os.getenv("REGISTERED_MODEL_NAME", f"{os.getenv('PROJECT_NAME', 'ROLLNO_A2_Ops')}_VisionGuard")

model = mlflow.tensorflow.load_model(f"models:/{name}@{args.alias}")
version = mlflow.MlflowClient().get_model_version_by_alias(name, args.alias).version
threshold = float(os.getenv("THRESHOLD", "0.5"))

with open(args.image, "rb") as fh:
    x = prepare(fh.read(), int(model.input_shape[1]))[None, ...]
v = verdict(float(model(x, training=False).numpy().ravel()[0]), threshold)
print(f"Loaded models:/{name}@{args.alias} (version {version})")
print(f"{args.image}: {v['label']}, confidence {v['confidence']:.1%}, P(AI) = {v['prob_ai']}, threshold {threshold}")
