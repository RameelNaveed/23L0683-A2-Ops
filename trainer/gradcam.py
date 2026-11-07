"""Bonus: Grad-CAM. Needs test_predictions.csv from `python -m trainer evaluate`.

Models here are flat (the backbone was built on our own input), so the last
4D activation can be read straight from a two output Keras model.
"""
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

from trainer.config import DATA_ROOT, RUNS  # noqa: E402
from visionguard.preprocess import prepare_file  # noqa: E402


def last_feature_map(model):
    for lyr in reversed(model.layers):
        if len(lyr.output.shape) == 4:
            return lyr.name
    raise ValueError("model has no 4D activation")


def cam(probe, image):
    with tf.GradientTape() as tape:
        fmap, prob = probe(image[None, ...], training=False)
        target = prob[:, 0]
    grads = tape.gradient(target, fmap)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    heat = tf.nn.relu(tf.reduce_sum(weights * fmap, axis=-1))[0]
    heat = heat / (tf.reduce_max(heat) + 1e-9)
    size = image.shape[0]
    return tf.image.resize(heat[..., None], (size, size), method="bicubic")[..., 0].numpy(), float(target[0])


def run(name):
    folder = RUNS / name
    model = keras.models.load_model(folder / "best_model.keras")
    size = int(model.input_shape[1])
    layer = last_feature_map(model)
    probe = keras.Model(model.inputs, [model.get_layer(layer).output, model.output])

    preds = pd.read_csv(folder / "test_predictions.csv")
    preds["certainty"] = (preds.p_ai - 0.5).abs()
    right = preds[preds.y == preds.pred].nlargest(4, "certainty").assign(group="correct")
    wrong = preds[preds.y != preds.pred].nlargest(4, "certainty").assign(group="wrong")
    chosen = pd.concat([right, wrong])

    fig, axes = plt.subplots(2, len(chosen), figsize=(2.7 * len(chosen), 6))
    axes = np.array(axes).reshape(2, -1)
    for j, (_, r) in enumerate(chosen.iterrows()):
        img = prepare_file(str(DATA_ROOT / r.file), size).numpy()
        heat, p = cam(probe, img)
        axes[0, j].imshow(img / 255.0)
        axes[0, j].set_title(f"{r.group}: {'AI' if r.y else 'Real'}\nP(AI)={p:.2f}", fontsize=8)
        axes[1, j].imshow(img / 255.0)
        axes[1, j].imshow(heat, cmap="inferno", alpha=0.5)
        for ax in axes[:, j]:
            ax.axis("off")
    fig.suptitle(f"Grad-CAM on layer {layer} (bright = strongest influence on P(AI))")
    fig.tight_layout()
    fig.savefig(folder / "gradcam.png", dpi=110)
    print("wrote", folder / "gradcam.png")
