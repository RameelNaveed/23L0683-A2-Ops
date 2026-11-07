"""Task 2.1: tf.data streams.

Evaluation stream (val, test):
    file list -> read + prepare (shorter side to S, centre crop) -> cache -> batch -> prefetch

Training stream:
    file list split by class -> read, decode, shorter side to 1.15 x S -> cache
      -> repeat + shuffle (each class on its own)
      -> sample_from_datasets 50/50, so every batch is class balanced
      -> random S x S crop, random horizontal flip, random JPEG re-compression
      -> batch -> prefetch
Augmentation happens after the cache, so it is new every epoch, and only here.
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

from trainer.config import DATA_ROOT, FIGS, SEED, SPLITS  # noqa: E402
from visionguard.preprocess import centre_square, prepare_file, shorter_side_to, to_rgb  # noqa: E402

AUTO = tf.data.AUTOTUNE
MARGIN = 1.15


def split_table(name):
    df = pd.read_csv(SPLITS / f"{name}.csv")
    df["path"] = [str(DATA_ROOT / f) for f in df.file]
    return df


def eval_stream(name, size=224, batch=32):
    df = split_table(name)
    ds = tf.data.Dataset.from_tensor_slices((df.path.values, df.y.values.astype("float32")))
    ds = ds.map(lambda p, y: (prepare_file(p, size), y), num_parallel_calls=AUTO).cache()
    return ds.batch(batch).prefetch(AUTO), df


def _loose(path, size):
    return shorter_side_to(to_rgb(tf.io.read_file(path)), int(math.ceil(size * MARGIN)))


def _jitter(img, size):
    img = tf.image.random_crop(img, [size, size, 3])
    img = tf.image.random_flip_left_right(img)
    # Re-save as JPEG at a random quality, so file compression cannot become a class cue.
    img = tf.image.random_jpeg_quality(img / 255.0, 70, 100) * 255.0
    img.set_shape([size, size, 3])
    return img


def _centre(img, size):
    img = centre_square(img, size)
    img.set_shape([size, size, 3])
    return img


def train_stream(size=224, batch=32, augment=True, balance=True, seed=SEED):
    df = split_table("train")
    shape = _jitter if augment else _centre

    def per_class(part):
        ds = tf.data.Dataset.from_tensor_slices((part.path.values, part.y.values.astype("float32")))
        ds = ds.map(lambda p, y: (_loose(p, size), y), num_parallel_calls=AUTO).cache()
        return ds.repeat().shuffle(len(part), seed=seed)

    if balance:
        streams = [per_class(df[df.y == k]) for k in (0, 1)]
        ds = tf.data.Dataset.sample_from_datasets(streams, weights=[0.5, 0.5], seed=seed)
    else:
        ds = per_class(df)
    ds = ds.map(lambda x, y: (tf.clip_by_value(shape(x, size), 0, 255), y), num_parallel_calls=AUTO)
    steps = max(1, len(df) // batch)
    return ds.batch(batch).prefetch(AUTO), steps, df


def preview(size=224):
    ds, _, _ = train_stream(size, 16)
    x, y = next(iter(ds))
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.6))
    for ax, img, lab in zip(axes.ravel(), x.numpy(), y.numpy()):
        ax.imshow(np.clip(img / 255.0, 0, 1))
        ax.set_title("AI" if lab == 1 else "Real", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Training batch after augmentation: {int(y.numpy().sum())} AI and "
                 f"{int((y.numpy() == 0).sum())} real (class balanced sampling)")
    fig.tight_layout()
    fig.savefig(FIGS / "2.1_training_batch.png", dpi=110)
    lo, hi = float(x.numpy().min()), float(x.numpy().max())
    print("batch:", tuple(x.shape), x.dtype.name, f"values {lo:.0f} to {hi:.0f}")
    print("wrote", FIGS / "2.1_training_batch.png")
