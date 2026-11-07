"""Paths, environment and experiment presets for the trainer."""
import copy
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_ROOT = Path(os.getenv("DATA_ROOT", ROOT / "data" / "raw"))
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "outputs"
RUNS = OUT / "runs"
FIGS = OUT / "figures"
AI_FOLDER = os.getenv("POSITIVE_CLASS", "AiArtData")   # label 1; every other folder is label 0
SEED = int(os.getenv("SEED", "42"))

PROJECT = os.getenv("PROJECT_NAME", "ROLLNO_A2_Ops")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", PROJECT)
MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", f"{PROJECT}_VisionGuard")

for _d in (SPLITS, RUNS, FIGS):
    _d.mkdir(parents=True, exist_ok=True)


def presets() -> dict:
    with open(ROOT / "experiments.yaml", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    merged = {}
    for name, body in spec["experiments"].items():
        cfg = copy.deepcopy(spec["defaults"])
        cfg.update(body)
        cfg["name"] = name
        merged[name] = cfg
    return merged


def preset(name: str) -> dict:
    table = presets()
    if name not in table:
        raise SystemExit(f"Unknown preset '{name}'. Choose from: {', '.join(table)}")
    return table[name]
