"""Tasks 2.2 to 2.4 and 3.3 to 3.5: train one preset from experiments.yaml.

Each preset is a list of phases. A phase sets which backbone layers may learn,
its own learning rate and its own epoch budget. Callbacks in every phase:
  * EarlyStopping on val_loss, best weights restored
  * ModelCheckpoint on val_loss, keeping the single best model over ALL phases
  * ReduceLROnPlateau on val_loss
"""
import json

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from trainer import models, mlflow_tracking  # noqa: E402
from trainer.dataset import eval_stream, train_stream  # noqa: E402
from trainer.config import RUNS, SPLITS, preset  # noqa: E402


def draw(history, title, path, best=None, phase_starts=()):
    epochs = range(1, len(history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ("loss", "accuracy")):
        ax.plot(epochs, history[key], marker=".", label="train")
        ax.plot(epochs, history[f"val_{key}"], marker=".", label="validation")
        for start in phase_starts:
            ax.axvline(start - 0.5, color="grey", ls=":", lw=1)
        if best:
            ax.axvline(best, color="crimson", ls="--", lw=1.2, label=f"best epoch ({best})")
        ax.set_title(f"{key}")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


def run(name, max_epochs=None, local=False, weights="imagenet"):
    cfg = preset(name)
    keras.utils.set_random_seed(cfg["seed"])
    size, batch = cfg["img_size"], cfg["batch_size"]
    folder = RUNS / name
    folder.mkdir(parents=True, exist_ok=True)
    best_path = folder / "best_model.keras"
    best_path.unlink(missing_ok=True)

    train_ds, steps, train_df = train_stream(size, batch, cfg["augment"], cfg["balance"], cfg["seed"])
    val_ds, _ = eval_stream("val", size, batch)

    if cfg["family"] == "cnn":
        sample, _ = eval_stream("train", size, batch)
        model, owned = models.separable_cnn(size, sample.map(lambda x, y: x).take(10), cfg["dropout"])
        backbone = None
    else:
        backbone = cfg["backbone"]
        model, owned = models.pretrained(size, backbone, None if weights == "none" else weights, cfg["dropout"])

    mlflow_tracking.connect(local)
    autolog = cfg["logging"] == "autolog"
    with mlflow.start_run(run_name=name) as active:
        mlflow.set_tags(mlflow_tracking.provenance(cfg["seed"], {
            "logging_mode": cfg["logging"], "family": cfg["family"],
            "compare_to": cfg.get("compare_to", "none")}))
        if autolog:
            mlflow.tensorflow.autolog(log_models=True, log_datasets=False, silent=True)
        else:
            mlflow.log_params({**mlflow_tracking.flatten({k: v for k, v in cfg.items() if k != "name"}),
                               "train_images": len(train_df), "steps_per_epoch": steps})

        frames, starts, best_loss = [], [], None
        for i, phase in enumerate(cfg["phases"], start=1):
            epochs = min(phase["epochs"], max_epochs) if max_epochs else phase["epochs"]
            models.apply_unfreeze(model, owned, backbone, phase.get("unfreeze", "none"))
            models.compile_for(model, phase["lr"], cfg["weight_decay"])
            trainable = "\n".join(models.trainable_now(model))
            (folder / f"trainable_layers_stage{i}.txt").write_text(trainable, encoding="utf-8")
            offset = sum(len(f) for f in frames)
            starts.append(offset + 1)
            callbacks = [
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["patience"],
                                              restore_best_weights=True),
                keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_loss", save_best_only=True,
                                                initial_value_threshold=best_loss),
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7),
                keras.callbacks.CSVLogger(str(folder / f"stage{i}_history.csv")),
            ]
            if not autolog:
                callbacks.append(mlflow_tracking.epoch_logger(offset))
            print(f"\n=== {name}: stage {i}, lr {phase['lr']}, unfreeze {phase.get('unfreeze', 'n/a')}, "
                  f"up to {epochs} epochs ===")
            hist = model.fit(train_ds, steps_per_epoch=steps, validation_data=val_ds, epochs=epochs,
                             callbacks=callbacks, verbose=2)
            frame = pd.DataFrame(hist.history)
            frames.append(frame)
            low = float(frame.val_loss.min())
            best_loss = low if best_loss is None else min(best_loss, low)
            if len(cfg["phases"]) > 1:
                draw(frame, f"{name}: stage {i}", folder / f"stage{i}_curves.png",
                     best=int(frame.val_loss.idxmin()) + 1)

        history = pd.concat(frames, ignore_index=True)
        history.index += 1
        best_epoch = int(history.val_loss.idxmin())
        history.to_csv(folder / "history.csv", index_label="epoch")
        draw(history, f"{name}: all stages", folder / "curves.png", best=best_epoch, phase_starts=starts[1:])
        model = keras.models.load_model(best_path)

        at_best = history.loc[best_epoch]
        summary = {"best_epoch": best_epoch, "epochs_trained": len(history)}
        summary.update({f"best_{k}": float(at_best[k]) for k in
                        ("val_loss", "val_accuracy", "val_auc", "val_precision", "val_recall")})
        mlflow.log_metrics(summary)

        lines = []
        model.summary(print_fn=lambda line, **_: lines.append(line))
        (folder / "model_summary.txt").write_text("\n".join(lines), encoding="utf-8")
        pd.DataFrame(models.layer_rows(model)).to_csv(folder / "layer_table.csv", index=False)
        mlflow_tracking.environment_file(folder / "pip_freeze.txt")

        for f in sorted(folder.glob("*")):
            if f.suffix in {".png", ".csv", ".txt"}:
                mlflow.log_artifact(str(f), "training")
        for f in sorted(SPLITS.glob("*")):
            mlflow.log_artifact(str(f), "data_splits")
        mlflow.log_artifact(str(best_path), "keras_model")
        uri = None if autolog else mlflow.tensorflow.log_model(model, name="classifier").model_uri

        record = {"run_id": active.info.run_id, "preset": name, "model_uri": uri, "img_size": size,
                  "family": cfg["family"], "backbone": backbone or "none", "logging": cfg["logging"], **summary}
        (folder / "run_info.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record, indent=2))
