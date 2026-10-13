"""Volcano (element term) per element + summary heat-map. python make_plots.py results/proteomics/vl_none"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ELEMENTS = ["Na", "Mg", "P", "K", "Ca", "Fe", "Cu", "Zn"]


def volcano_grid(folder, term="E", label_top=5, alpha=0.05):
    folder = Path(folder)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, el in zip(axes.ravel(), ELEMENTS):
        f = folder / f"lmm_{el}.csv"
        if not f.exists():
            ax.axis("off"); continue
        r = pd.read_csv(f)
        y = -np.log10(r[f"p_{term}"].clip(lower=1e-300)); sig = r[f"q_{term}"] < alpha
        ax.scatter(r[f"beta_{term}"][~sig], y[~sig], s=6, c="lightgray")
        ax.scatter(r[f"beta_{term}"][sig], y[sig], s=10, c="crimson")
        for i in r.loc[:, f"p_{term}"].nsmallest(label_top).index:
            ax.annotate(str(r.loc[i, "label"])[:14], (r.loc[i, f"beta_{term}"], y[i]), fontsize=6)
        ax.set_title(f"{el}  (FDR<{alpha}: {int(sig.sum())})"); ax.set_xlabel("beta per 1 SD of element")
        ax.set_ylabel("-log10 p")
    fig.tight_layout(); fig.savefig(folder / "volcano_elements.png", dpi=150); plt.close(fig)


def summary_heatmap(folder, alpha=0.05):
    folder = Path(folder)
    rows = {}
    for el in ELEMENTS:
        f = folder / f"lmm_{el}.csv"
        if f.exists():
            r = pd.read_csv(f)
            rows[el] = {c[2:]: int((r[c] < alpha).sum()) for c in r if c.startswith("q_") and c != "q_E_global"}
    df = pd.DataFrame(rows).T
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(df.values, cmap="Reds"); ax.set_xticks(range(df.shape[1]), df.columns, rotation=45, ha="right")
    ax.set_yticks(range(df.shape[0]), df.index)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, df.iat[i, j], ha="center", va="center", fontsize=8)
    ax.set_title(f"# features with FDR < {alpha}  (rows = element model)")
    fig.tight_layout(); fig.savefig(folder / "summary_heatmap.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    volcano_grid(sys.argv[1]); summary_heatmap(sys.argv[1])
