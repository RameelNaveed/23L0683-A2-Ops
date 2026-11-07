"""Task 1.2: dataset audit, one row per file.

For each file: category (class folder), bytes, width, height, colour mode,
encoding, SHA-256 (exact copies) and a 64 bit difference hash (dHash) that also
catches near copies, such as the same picture saved again at another size or
JPEG quality. Files are read in parallel threads.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io

import numpy as np
import pandas as pd
from PIL import Image

from trainer.config import DATA_ROOT, OUT


def dhash(img: Image.Image) -> str:
    grey = np.asarray(img.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (grey[:, 1:] > grey[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def examine(item):
    path, category = item
    raw = path.read_bytes()
    rec = {"file": path.relative_to(DATA_ROOT).as_posix(), "category": category, "nbytes": len(raw),
           "sha256": hashlib.sha256(raw).hexdigest(), "width": np.nan, "height": np.nan,
           "colour_mode": "", "encoding": "", "dhash": "", "ok": True, "problem": ""}
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            rec.update(width=img.width, height=img.height, colour_mode=img.mode,
                       encoding=img.format or "", dhash=dhash(img))
    except Exception as err:
        rec.update(ok=False, problem=f"{type(err).__name__}: {err}"[:160])
    return rec


def label_groups(df: pd.DataFrame) -> pd.DataFrame:
    df["exact_group"] = df.groupby("sha256").ngroup()
    df["exact_copies"] = df.groupby("sha256")["file"].transform("size")
    near = df[df.ok].groupby("dhash")["sha256"].transform("nunique")
    df["near_copies"] = near.reindex(df.index).fillna(1).astype(int)
    # A dHash shared by many different files, or a flat hash, means "simple picture",
    # not "copy". Only small groups (2 to 4 files) count as near copies.
    flat = df.dhash.isin(["0000000000000000", "ffffffffffffffff"])
    df.loc[flat | (df.near_copies > 4), "near_copies"] = 1
    df["group_key"] = np.where(df.near_copies > 1, "d:" + df.dhash, "s:" + df.sha256)
    df["cross_class"] = df.groupby("group_key")["category"].transform("nunique") > 1
    first = ~df.duplicated("sha256", keep="first")
    df["keep"] = df.ok & first & ~df.cross_class
    return df


def run():
    folders = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir())
    if not folders:
        raise SystemExit(f"No class folders inside {DATA_ROOT}. Set DATA_ROOT in .env.")
    items = [(f, d.name) for d in folders for f in sorted(d.rglob("*")) if f.is_file()]
    print(f"Inspecting {len(items)} files in {len(folders)} folders under {DATA_ROOT}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        df = pd.DataFrame(list(pool.map(examine, items)))
    df = label_groups(df)
    df.to_csv(OUT / "audit.csv", index=False)

    good = df[df.ok]
    parts = {
        "Files per class": pd.DataFrame({
            "files": df.groupby("category").size(),
            "unreadable": (~df.ok).groupby(df.category).sum(),
            "exact_extra_copies": (df.ok & df.duplicated("sha256")).groupby(df.category).sum(),
            "kept": df.keep.groupby(df.category).sum(),
        }).astype(int),
        "Width (px)": good.groupby("category")["width"].describe(percentiles=[.5])[["min", "50%", "max"]],
        "Height (px)": good.groupby("category")["height"].describe(percentiles=[.5])[["min", "50%", "max"]],
        "File size (KB)": (good.assign(kb=good.nbytes / 1024).groupby("category")["kb"]
                           .describe(percentiles=[.5])[["min", "50%", "max"]].round(1)),
        "Colour modes": pd.crosstab(good.category, good.colour_mode),
        "Encodings": pd.crosstab(good.category, good.encoding),
    }
    text = []
    for title, table in parts.items():
        text += [f"== {title} ==", table.to_string(), ""]
    text += [
        f"Unreadable files: {int((~df.ok).sum())}",
        f"Exact duplicate files beyond the first copy: {int((df.ok & df.duplicated('sha256')).sum())}",
        f"Near duplicate groups (same dHash, different bytes): "
        f"{int(df[df.near_copies > 1].dhash.nunique())}",
        f"Files whose picture appears in both classes: {int(df.cross_class.sum())}",
        f"Files kept for splitting: {int(df.keep.sum())}",
    ]
    bad = df[~df.ok][["file", "problem"]]
    if len(bad):
        text += ["", "Unreadable:", bad.to_string(index=False)]
    report = "\n".join(text)
    (OUT / "audit_summary.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nPer file table: {OUT / 'audit.csv'}")
