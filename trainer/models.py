"""Task 2.2 (a CNN from scratch) and Task 2.3 (a pretrained backbone)."""
import keras
from keras import layers

BACKBONES = {
    # name: (constructor, input scaling we add, layer name prefixes of the top block)
    "efficientnetb0": (keras.applications.EfficientNetB0, None, ("block7a", "top_conv")),
    "mobilenetv2": (keras.applications.MobileNetV2, "minus1_to_1", ("block_16", "Conv_1")),
}


def separable_cnn(size, adapt_ds, dropout=0.3):
    """Stem + three stages of depthwise separable convolutions.

    normalise : per channel mean and variance learned from training images (adapt),
                stored inside the model
    stem      : 5x5 conv, stride 2: cheap first look at the full image, halves resolution
    stages    : two SeparableConv2D per stage. Each splits a 3x3 conv into a per channel
                spatial filter plus a 1x1 mixing step, about 8 times fewer weights than a
                normal conv, which suits a small dataset. BatchNorm + ReLU after each,
                MaxPool at the end of a stage.
    head      : global average pooling, dropout, one sigmoid unit = P(AI generated)
    """
    norm = layers.Normalization(axis=-1, name="normalise")
    norm.adapt(adapt_ds)
    inp = keras.Input((size, size, 3), name="pixels")
    x = norm(inp)
    x = layers.Conv2D(24, 5, strides=2, padding="same", use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(momentum=0.9, name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    for s, width in enumerate((48, 96, 192), start=1):
        for k in ("a", "b"):
            x = layers.SeparableConv2D(width, 3, padding="same", use_bias=False, name=f"s{s}{k}_sep")(x)
            x = layers.BatchNormalization(momentum=0.9, name=f"s{s}{k}_bn")(x)
            x = layers.ReLU(name=f"s{s}{k}_relu")(x)
        x = layers.MaxPooling2D(2, name=f"s{s}_pool")(x)
    x = layers.GlobalAveragePooling2D(name="pool")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return keras.Model(inp, out, name="separable_cnn"), set()


def pretrained(size, backbone="efficientnetb0", weights="imagenet", dropout=0.3):
    """Backbone built on our own input tensor, so all its layers sit flat in one model."""
    ctor, scaling, _ = BACKBONES[backbone]
    inp = keras.Input((size, size, 3), name="pixels")
    x = inp
    if scaling == "minus1_to_1":
        x = layers.Rescaling(1 / 127.5, offset=-1.0, name="to_minus1_1")(x)
    base = ctor(include_top=False, weights=weights, input_tensor=x)
    h = layers.GlobalAveragePooling2D(name="pool")(base.output)
    h = layers.Dropout(dropout, name="head_dropout")(h)
    out = layers.Dense(1, activation="sigmoid", name="p_ai")(h)
    model = keras.Model(inp, out, name=f"{backbone}_classifier")
    owned = {lyr.name for lyr in base.layers if lyr.weights}
    return model, owned


def apply_unfreeze(model, owned, backbone, mode):
    """mode: 'none' (backbone frozen), 'top' (last block), or an int n (last n weighted layers)."""
    if not owned:
        return
    backbone_layers = [lyr for lyr in model.layers if lyr.name in owned]
    if mode == "none" or mode == 0:
        chosen = set()
    elif mode == "top":
        prefixes = BACKBONES[backbone][2]
        chosen = {lyr.name for lyr in backbone_layers if lyr.name.startswith(prefixes)}
    else:
        chosen = {lyr.name for lyr in backbone_layers[-int(mode):]}
    for lyr in backbone_layers:
        # BatchNorm stays frozen: re-estimating its statistics from small batches hurts.
        lyr.trainable = lyr.name in chosen and not isinstance(lyr, layers.BatchNormalization)


def compile_for(model, lr, weight_decay=1e-4):
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"), keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )


def trainable_now(model):
    return [lyr.name for lyr in model.layers if lyr.trainable and lyr.trainable_weights]


def layer_rows(model):
    rows = []
    for lyr in model.layers:
        rows.append({"name": lyr.name, "kind": type(lyr).__name__,
                     "output": str(tuple(lyr.output.shape)) if hasattr(lyr, "output") else "",
                     "weights": lyr.count_params()})
    return rows
