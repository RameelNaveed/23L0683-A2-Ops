"""Task 3.5: one table and one chart of every run in the experiment, from MLflow itself."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402

from trainer import mlflow_tracking  # noqa: E402
from trainer.config import EXPERIMENT, FIGS, OUT  # noqa: E402

WANTED = {
    "tags.mlflow.runName": "run", "tags.compare_to": "compared_with", "tags.logging_mode": "logging",
    "params.backbone": "backbone", "params.img_size": "img_size", "params.augment": "augment",
    "params.phase1_lr": "lr_phase1", "params.phase2_lr": "lr_phase2", "params.phase2_unfreeze": "phase2_unfreeze",
    "metrics.best_val_auc": "val_auc", "metrics.best_val_loss": "val_loss", "metrics.best_val_accuracy": "val_accuracy",
    "metrics.best_val_precision": "val_precision", "metrics.best_val_recall": "val_recall",
    "metrics.best_epoch": "best_epoch",
}


def run(local=False):
    mlflow_tracking.connect(local)
    found = mlflow.search_runs(experiment_names=[EXPERIMENT], order_by=["metrics.best_val_auc DESC"])
    if found.empty:
        raise SystemExit("No runs found in this experiment yet.")
    found = found[found.get("metrics.best_val_auc").notna()]
    board = found[[c for c in WANTED if c in found.columns]].rename(columns=WANTED)
    board = board.drop_duplicates("run", keep="first")
    board.to_csv(OUT / "run_comparison.csv", index=False)
    print(board.round(4).to_string(index=False))

    ranked = board.sort_values("val_auc")
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(ranked) + 1.5))
    ax.hlines(ranked.run, ranked.val_auc.min() - 0.02, ranked.val_auc, color="lightgrey")
    ax.plot(ranked.val_auc, ranked.run, "o", color="indigo", label="val AUC")
    ax.set_xlabel("validation AUC at the best epoch")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "3.5_run_comparison.png", dpi=120)
    print("wrote", FIGS / "3.5_run_comparison.png", "and", OUT / "run_comparison.csv")
