"""Tasks 2.5 and 2.6: one look at the test set, with the threshold fixed beforehand.

Threshold: chosen on VALIDATION predictions to minimise cost, where a real photo
flagged as AI (false positive) costs FP_COST times as much as an AI image that
slips through (false negative). Default FP_COST = 3.
"""
import json

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn import metrics as skm  # noqa: E402

from trainer import mlflow_tracking  # noqa: E402
from trainer.dataset import eval_stream  # noqa: E402
from trainer.config import DATA_ROOT, OUT, RUNS  # noqa: E402


def pick_threshold(y, p, fp_cost):
    rows = []
    for t in np.linspace(0.02, 0.98, 97):
        hat = p >= t
        fp = int(((y == 0) & hat).sum())
        fn = int(((y == 1) & ~hat).sum())
        rows.append({"threshold": round(float(t), 2), "false_pos": fp, "false_neg": fn, "cost": fp_cost * fp + fn})
    table = pd.DataFrame(rows)
    ties = table[table.cost == table.cost.min()]
    return float(ties.threshold.iloc[len(ties) // 2]), table   # middle of equal cost thresholds


def scores(y, p, t):
    hat = (p >= t).astype(int)
    tn, fp, fn, tp = skm.confusion_matrix(y, hat, labels=[0, 1]).ravel()
    return {"accuracy": skm.accuracy_score(y, hat), "precision": skm.precision_score(y, hat, zero_division=0),
            "recall": skm.recall_score(y, hat, zero_division=0), "f1": skm.f1_score(y, hat, zero_division=0),
            "roc_auc": skm.roc_auc_score(y, p), "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def figures(y, p, t, name, folder):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    skm.ConfusionMatrixDisplay.from_predictions(y, (p >= t).astype(int), display_labels=["Real", "AI"],
                                                cmap="Purples", colorbar=False, ax=ax)
    ax.set_title(f"{name}, test set, threshold {t:.2f}")
    fig.tight_layout()
    fig.savefig(folder / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    skm.RocCurveDisplay.from_predictions(y, p, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], ls=":", c="grey")
    hat = p >= t
    ax.scatter([(hat & (y == 0)).sum() / max((y == 0).sum(), 1)], [(hat & (y == 1)).sum() / max((y == 1).sum(), 1)],
               c="crimson", zorder=4, label=f"threshold {t:.2f}")
    ax.set_xlabel("real images flagged as AI")
    ax.set_ylabel("AI images caught")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(folder / "roc_curve.png", dpi=130)
    plt.close(fig)


def cue(row):
    found = []
    if min(row.width, row.height) < 320:
        found.append(f"small ({int(row.width)}x{int(row.height)})")
    ratio = row.width / row.height
    if ratio > 1.7 or ratio < 0.6:
        found.append("edges cut off by centre crop")
    if row.colour_mode not in ("RGB", "RGBA"):
        found.append(f"mode {row.colour_mode}")
    if row.nbytes / (row.width * row.height) < 0.08:
        found.append("strong compression")
    return ", ".join(found) or "none detected"


def mistakes(table, t, name, folder):
    """The 10 most confident wrong predictions on the test set."""
    inv = pd.read_csv(OUT / "audit.csv")[["file", "width", "height", "colour_mode", "nbytes"]]
    wrong = table[table.y != table.pred].merge(inv, on="file", how="left")
    wrong = wrong.assign(off_by=(wrong.p_ai - t).abs()).nlargest(10, "off_by").reset_index(drop=True)
    wrong["kind"] = np.where(wrong.y == 0, "real called AI", "AI called real")
    wrong["cue"] = wrong.apply(cue, axis=1) if len(wrong) else []
    wrong["hypothesis"] = ""
    wrong.to_csv(folder / "misclassified.csv", index=False)
    if wrong.empty:
        return
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.5))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (i, r) in zip(axes.ravel(), wrong.iterrows()):
        with Image.open(DATA_ROOT / r.file) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"{i + 1}. {r.kind}\nP(AI) = {r.p_ai:.2f}", fontsize=9)
    fig.suptitle(f"{name}: 10 misclassified test images")
    fig.tight_layout()
    fig.savefig(folder / "misclassified.png", dpi=105)
    plt.close(fig)


def run(name, fp_cost=3.0, again=False, local=False):
    folder = RUNS / name
    seal = folder / ".test_set_used"
    if seal.exists() and not again:
        raise SystemExit(f"{name} has already been evaluated on the test set. The test set is used once.")
    record = json.loads((folder / "run_info.json").read_text())
    model = keras.models.load_model(folder / "best_model.keras")
    size = int(model.input_shape[1])

    val_ds, val_df = eval_stream("val", size)
    p_val = model.predict(val_ds, verbose=0).ravel()
    t, cost_table = pick_threshold(val_df.y.values, p_val, fp_cost)
    cost_table.to_csv(folder / "threshold_search.csv", index=False)

    test_ds, test_df = eval_stream("test", size)
    p = model.predict(test_ds, verbose=0).ravel()
    y = test_df.y.values
    s = scores(y, p, t)
    seal.write_text(f"threshold={t}")

    table = test_df[["file", "y"]].assign(p_ai=p, pred=(p >= t).astype(int))
    table.to_csv(folder / "test_predictions.csv", index=False)
    figures(y, p, t, name, folder)
    mistakes(table, t, name, folder)

    report = {"threshold": t, "fp_cost": fp_cost, "test": {k: float(v) for k, v in s.items()},
              "test_images": int(len(y))}
    (folder / "test_metrics.json").write_text(json.dumps(report, indent=2))
    print(f"Threshold from validation (FP cost {fp_cost}): {t:.2f}")
    print(pd.Series({k: s[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}).round(4).to_string())
    print(f"confusion: tn={s['tn']} fp={s['fp']} fn={s['fn']} tp={s['tp']}")

    mlflow_tracking.connect(local)
    with mlflow.start_run(run_id=record["run_id"]):
        mlflow.log_param("decision_threshold", t)
        mlflow.log_param("fp_cost", fp_cost)
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in s.items()})
        for f in ("confusion_matrix.png", "roc_curve.png", "misclassified.png", "misclassified.csv",
                  "test_metrics.json", "test_predictions.csv", "threshold_search.csv"):
            if (folder / f).exists():
                mlflow.log_artifact(str(folder / f), "test_evaluation")
