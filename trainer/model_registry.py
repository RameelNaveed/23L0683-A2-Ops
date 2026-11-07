"""Task 3.6: model registry, and handing the champion to the servers.
Loading by alias is the separate script scripts/predict_by_alias.py."""
import json
import shutil

import mlflow
from mlflow import MlflowClient

from trainer import mlflow_tracking
from trainer.config import MODEL_NAME, ROOT, RUNS


def register(name, alias=None, local=False):
    mlflow_tracking.connect(local)
    record = json.loads((RUNS / name / "run_info.json").read_text())
    if not record.get("model_uri"):
        raise SystemExit(f"{name} was an autolog run without a classifier model; register a manual run.")
    version = mlflow.register_model(record["model_uri"], MODEL_NAME)
    client = MlflowClient()
    client.update_model_version(MODEL_NAME, version.version,
                                description=f"preset {name}: {record['family']} {record['backbone']}, "
                                            f"best val AUC {record['best_val_auc']:.3f}")
    client.set_model_version_tag(MODEL_NAME, version.version, "preset", name)
    if alias:
        client.set_registered_model_alias(MODEL_NAME, alias, version.version)
    print(f"{MODEL_NAME} v{version.version} <- run {record['run_id']} ({name})"
          + (f", alias @{alias}" if alias else ""))


def export(alias="champion", local=False):
    mlflow_tracking.connect(local)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, alias)
    run = client.get_run(mv.run_id)
    fetched = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="keras_model/best_model.keras")
    target = ROOT / "models" / "champion.keras"
    target.parent.mkdir(exist_ok=True)
    shutil.copy(fetched, target)
    threshold = run.data.params.get("decision_threshold", "0.5")
    card = {"model": MODEL_NAME, "version": mv.version, "alias": alias, "run_id": mv.run_id,
            "preset": run.info.run_name, "threshold": float(threshold)}
    (target.parent / "champion.json").write_text(json.dumps(card, indent=2))
    print(f"copied version {mv.version} to {target}\nadd to .env:\n"
          f"MODEL_PATH=models/champion.keras\nTHRESHOLD={threshold}\nMODEL_VERSION={MODEL_NAME}:v{mv.version}")
