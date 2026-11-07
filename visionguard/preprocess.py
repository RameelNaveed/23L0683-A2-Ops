"""Image preparation shared by the trainer, the web app and the cloud API.

Serving and evaluation use `prepare`:
    bytes -> RGB float32 -> shorter side scaled to `size` (aspect ratio kept)
          -> centre square of size x size cut out

Nothing is stretched and no padding is added; the cost is that the outer edges
of very wide or very tall images are cut away. Pixel values stay in [0, 255];
each saved model scales them itself.
"""
import tensorflow as tf

SIZE = 224


def to_rgb(image_bytes):
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return tf.cast(img, tf.float32)


def shorter_side_to(img, target):
    """Scale so the shorter side equals `target`, keeping the aspect ratio."""
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    scale = tf.cast(target, tf.float32) / tf.minimum(h, w)
    new_hw = tf.cast(tf.math.ceil(tf.stack([h * scale, w * scale])), tf.int32)
    new_hw = tf.maximum(new_hw, target)
    return tf.image.resize(img, new_hw, method="area", antialias=True)


def centre_square(img, size):
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    top, left = (h - size) // 2, (w - size) // 2
    return img[top:top + size, left:left + size, :]


def prepare(image_bytes, size: int = SIZE):
    img = centre_square(shorter_side_to(to_rgb(image_bytes), size), size)
    img = tf.clip_by_value(img, 0.0, 255.0)
    img.set_shape([size, size, 3])
    return img


def prepare_file(path, size: int = SIZE):
    return prepare(tf.io.read_file(path), size)
