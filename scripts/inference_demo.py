"""Task 4.1 evidence: load the champion once, then classify two (or more) known images.

Usage:
    python scripts/inference_demo.py path\\to\\ai.jpg path\\to\\real.jpg
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from visionguard.predictor import LocalPredictor  # noqa: E402

load_dotenv()
t0 = time.time()
p = LocalPredictor(os.getenv("MODEL_PATH", "models/champion.keras"), float(os.getenv("THRESHOLD", "0.5")),
                   os.getenv("MODEL_VERSION", "champion"))
print(f"Model loaded once in {time.time() - t0:.2f}s: {p.info()}")
for path in sys.argv[1:]:
    with open(path, "rb") as f:
        r = p.predict_bytes(f.read())
    print(f"{path}: {r['label']} (confidence {r['confidence']:.2%}, p(AI)={r['prob_ai']}, {r['latency_ms']} ms)")
