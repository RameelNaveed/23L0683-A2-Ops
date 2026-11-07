"""Split lists for training: data/splits/{train,val,test}.csv.

Method: group aware and deterministic.
  * a group is one picture: exact copies share a SHA-256, near copies share a dHash
  * within each class, groups are ordered by sha256(seed + group key), a stable
    pseudo random order, and cut 70 / 15 / 15
  * so near copies can never be split across train and test, and the same seed
    always gives the same lists on any machine
Files TensorFlow cannot decode (for example CMYK JPEGs) are left out.
"""
import hashlib
import json

import pandas as pd
import tensorflow as tf

from trainer.config import AI_FOLDER, DATA_ROOT, OUT, SEED, SPLITS

SHARES = (("train", 0.70), ("val", 0.15), ("test", 0.15))


def decodable(rel):
    try:
        tf.io.decode_image(tf.io.read_file(str(DATA_ROOT / rel)), channels=3, expand_animations=False)
        return True
    except Exception:
        return False


def order_key(group, seed):
    return hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()


def run(seed: int = SEED):
    inv = pd.read_csv(OUT / "audit.csv")
    pool = inv[inv.keep].copy()
    pool["tf_ok"] = [decodable(f) for f in pool.file]
    dropped = int((~pool.tf_ok).sum())
    pool = pool[pool.tf_ok]
    pool["y"] = (pool.category == AI_FOLDER).astype(int)

    assigned = []
    for _, part in pool.groupby("category"):
        groups = sorted(part.group_key.unique(), key=lambda g: order_key(g, seed))
        n = len(groups)
        cut1, cut2 = round(n * SHARES[0][1]), round(n * (SHARES[0][1] + SHARES[1][1]))
        where = {g: ("train" if i < cut1 else "val" if i < cut2 else "test") for i, g in enumerate(groups)}
        assigned.append(part.assign(split=part.group_key.map(where)))
    pool = pd.concat(assigned)

    crossing = pool.groupby("group_key")["split"].nunique().gt(1).sum()
    if crossing:
        raise SystemExit(f"{crossing} picture groups ended up in more than one split")

    counts = {}
    for name, _ in SHARES:
        part = pool[pool.split == name][["file", "y", "category", "sha256"]]
        part.to_csv(SPLITS / f"{name}.csv", index=False)
        counts[name] = {"total": len(part), "ai": int(part.y.sum()), "real": int((part.y == 0).sum())}
    (SPLITS / "split_counts.json").write_text(json.dumps({"seed": seed, **counts}, indent=2))
    print(f"Left out {dropped} files TensorFlow cannot decode.")
    print(pd.DataFrame(counts).T.to_string())
    print("No picture group crosses splits.")
