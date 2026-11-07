"""Tasks 3.1 and 3.2: MLflow tool lab on Fashion-MNIST.

Local (3.1):   python labs/mlflow_toy.py --lr 0.001 --local
               python labs/mlflow_toy.py --lr 0.01  --local
               mlflow ui --backend-store-uri sqlite:///mlflow.db
DagsHub (3.2): python labs/mlflow_toy.py --lr 0.001     (reads MLFLOW_* from .env)
"""
import argparse
import os
from pathlib import Path

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


class EpochLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--units", type=int, default=128)
    ap.add_argument("--local", action="store_true")
    a = ap.parse_args()

    local_uri = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    uri = local_uri if a.local else os.getenv("MLFLOW_TRACKING_URI", local_uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("toy_fashion_mnist")
    print("Tracking to:", uri)

    (x, y), (xt, yt) = keras.datasets.fashion_mnist.load_data()
    x, xt = x[:20000] / 255.0, xt[:5000] / 255.0
    y, yt = y[:20000], yt[:5000]
    keras.utils.set_random_seed(42)
    model = keras.Sequential([keras.Input((28, 28)), keras.layers.Flatten(),
                              keras.layers.Dense(a.units, activation="relu"),
                              keras.layers.Dense(10, activation="softmax")])
    model.compile(keras.optimizers.Adam(a.lr), "sparse_categorical_crossentropy", metrics=["accuracy"])

    with mlflow.start_run(run_name=f"toy_lr_{a.lr}"):
        mlflow.set_tags({"purpose": "tool lab", "dataset": "fashion_mnist_subset"})
        mlflow.log_params({"lr": a.lr, "epochs": a.epochs, "units": a.units, "batch_size": 128})
        h = model.fit(x, y, validation_data=(xt, yt), epochs=a.epochs, batch_size=128,
                      callbacks=[EpochLogger()], verbose=2)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(h.history["loss"], label="train loss")
        ax.plot(h.history["val_loss"], label="val loss")
        ax.set_xlabel("epoch")
        ax.legend()
        ax.set_title(f"lr = {a.lr}")
        mlflow.log_figure(fig, "loss_curve.png")
        mlflow.tensorflow.log_model(model, name="model")
    print("Done. Final val accuracy:", round(h.history["val_accuracy"][-1], 4))


if __name__ == "__main__":
    main()
