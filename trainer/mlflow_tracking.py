"""Task 3: everything that talks to MLflow.

Where runs go is decided only by environment variables (from the git ignored .env):
MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD.
Without MLFLOW_TRACKING_URI, or with --local, runs go to ./mlflow.db.
"""
import os
import subprocess
import sys
from pathlib import Path

import keras
import mlflow
import yaml

from trainer.config import EXPERIMENT, ROOT

LOCAL_STORE = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"


def connect(local=False, experiment=EXPERIMENT):
    target = LOCAL_STORE if local else os.getenv("MLFLOW_TRACKING_URI", LOCAL_STORE)
    mlflow.set_tracking_uri(target)
    mlflow.set_experiment(experiment)
    print(f"[mlflow] {target}  experiment={experiment}")


def provenance(seed, extra=None):
    """The three tags the brief asks for: git commit, DVC data version and seed."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "unavailable"
    pointer = Path(ROOT, "data", "raw.dvc")
    try:
        data_version = yaml.safe_load(pointer.read_text())["outs"][0]["md5"]
    except Exception:
        data_version = "data/raw.dvc not found"
    tags = {"git_commit": commit, "dvc_data_version": data_version, "seed": str(seed)}
    tags.update(extra or {})
    return tags


def flatten(cfg, prefix=""):
    flat = {}
    for key, val in cfg.items():
        if key == "phases":
            for i, phase in enumerate(val, start=1):
                flat.update(flatten(phase, f"phase{i}_"))
        elif isinstance(val, dict):
            flat.update(flatten(val, f"{prefix}{key}_"))
        else:
            flat[f"{prefix}{key}"] = val
    return flat


def epoch_logger(step_offset):
    """Manual per epoch logging. step_offset keeps phase 2 on the same x axis as phase 1."""
    return keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: mlflow.log_metrics(
        {k: float(v) for k, v in (logs or {}).items()}, step=step_offset + epoch))


def environment_file(path):
    Path(path).write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
                          encoding="utf-8")
    return path
