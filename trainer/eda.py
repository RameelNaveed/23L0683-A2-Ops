"""Task 1.3: sample grid per class, and histograms of width, height and file size per class.

Writes outputs/figures/1.3_sample_grid.png and outputs/figures/1.3_histograms.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from trainer.config import DATA_ROOT, FIGS, OUT, SEED  # noqa: E402


def sample_grid(df, per_class=8):
    cats = sorted(df.category.unique())
    fig, axes = plt.subplots(len(cats), per_class, figsize=(2 * per_class, 2.4 * len(cats)))
    axes = np.atleast_2d(axes)
    for r, cat in enumerate(cats):
        part = df[df.category == cat]
        pick = part.sample(min(per_class, len(part)), random_state=SEED)
        for c, ax in enumerate(axes[r]):
            ax.axis("off")
            if c < len(pick):
                row = pick.iloc[c]
                with Image.open(DATA_ROOT / row.file) as im:
                    ax.imshow(im.convert("RGB"))
                ax.set_title(f"{cat}\n{int(row.width)}x{int(row.height)}", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "1.3_sample_grid.png", dpi=120)
    plt.close(fig)


def histograms(df):
    panels = [("width", "width (px)", df.width), ("height", "height (px)", df.height),
              ("size", "file size (KB)", df.nbytes / 1024)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (_, label, values) in zip(axes, panels):
        bins = np.linspace(values.min(), values.max(), 40)
        for cat in sorted(df.category.unique()):
            ax.hist(values[df.category == cat], bins=bins, alpha=0.55, label=cat)
        ax.set_xlabel(label)
        ax.set_ylabel("images")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "1.3_histograms.png", dpi=120)
    plt.close(fig)


def run():
    df = pd.read_csv(OUT / "audit.csv")
    df = df[df.keep]
    sample_grid(df)
    histograms(df)
    print("wrote", FIGS / "1.3_sample_grid.png", "and", FIGS / "1.3_histograms.png")
