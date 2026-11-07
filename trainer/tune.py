"""Task 2.7: hyperparameter search with Keras Tuner.

toy   : Hyperband on Fashion-MNIST (number of units, learning rate)
apply : Bayesian optimisation on VisionGuard with EfficientNetB0
        (learning rate, dropout, number of unfrozen backbone layers), 8 trials
"""
import keras
import keras_tuner as kt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from trainer.config import OUT, ROOT, SEED


def trials_frame(tuner, metric):
    rows = [{"trial": t.trial_id, **t.hyperparameters.values, metric: t.score}
            for t in tuner.oracle.trials.values() if t.score is not None]
    return pd.DataFrame(rows).sort_values(metric, ascending=False)


def influence(frame, metric, names):
    """Spearman rank correlation of each setting with the score; |rho| near 1 = strong effect."""
    out = {}
    for n in names:
        if frame[n].nunique() > 1 and frame[metric].nunique() > 1:
            rho, _ = spearmanr(frame[n].astype(float), frame[metric])
            out[n] = 0.0 if np.isnan(rho) else round(float(rho), 3)
        else:
            out[n] = 0.0
    return pd.Series(out).reindex(sorted(out, key=lambda k: -abs(out[k])))


def toy():
    (x, y), (xv, yv) = keras.datasets.fashion_mnist.load_data()
    x, y, xv, yv = x[:20000] / 255.0, y[:20000], xv[:5000] / 255.0, yv[:5000]

    def build(hp):
        net = keras.Sequential([
            keras.Input((28, 28)), keras.layers.Flatten(),
            keras.layers.Dense(hp.Int("units", 32, 512, step=32), activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ])
        net.compile(keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
                    "sparse_categorical_crossentropy", metrics=["accuracy"])
        return net

    tuner = kt.Hyperband(build, objective="val_accuracy", max_epochs=6, factor=3, seed=SEED,
                         directory=str(ROOT / "tuner_runs"), project_name="fashion_toy", overwrite=True)
    tuner.search(x, y, validation_data=(xv, yv), batch_size=128, verbose=1)
    frame = trials_frame(tuner, "val_accuracy")
    frame.to_csv(OUT / "2.7_toy_trials.csv", index=False)
    print(frame.to_string(index=False))
    print("\nbest:", tuner.get_best_hyperparameters(1)[0].values)


def apply(trials=8, epochs=5, size=224, weights="imagenet"):
    from trainer import models
    from trainer.dataset import eval_stream, train_stream

    keras.utils.set_random_seed(SEED)
    train_ds, steps, _ = train_stream(size, 32)
    val_ds, _ = eval_stream("val", size)

    def build(hp):
        model, owned = models.pretrained(size, "efficientnetb0", None if weights == "none" else weights,
                                         hp.Float("dropout", 0.1, 0.5, step=0.1))
        models.apply_unfreeze(model, owned, "efficientnetb0", hp.Choice("unfreeze_layers", [0, 15, 30, 60]))
        models.compile_for(model, hp.Float("lr", 1e-5, 1e-3, sampling="log"))
        return model

    tuner = kt.BayesianOptimization(build, objective=kt.Objective("val_auc", "max"), max_trials=trials,
                                    num_initial_points=3, seed=SEED, directory=str(ROOT / "tuner_runs"),
                                    project_name="visionguard_bo", overwrite=True)
    tuner.search(train_ds, steps_per_epoch=steps, validation_data=val_ds, epochs=epochs, verbose=2,
                 callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=2)])
    frame = trials_frame(tuner, "val_auc")
    frame.to_csv(OUT / "2.7_visionguard_trials.csv", index=False)
    print(frame.to_string(index=False))
    eff = influence(frame, "val_auc", ["lr", "dropout", "unfreeze_layers"])
    eff.to_csv(OUT / "2.7_setting_influence.csv", header=["spearman_rho_with_val_auc"])
    print("\nRank correlation of each setting with val_auc:\n" + eff.to_string())
    print("\nbest:", tuner.get_best_hyperparameters(1)[0].values)
