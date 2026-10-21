"""Follow-up of the top protein pairs: is the pair's signal stable when each animal is left out (diagnostic only)?
usage: python pair_followup.py --data-dir <dir> --out-dir results_v2 --elements Mg P Cu"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import full_analysis as fa
from figures_v2 import COL, MK, INK, SURF

def r2_pair(E, A, B, S):
    X = np.column_stack([np.ones(len(E)), S, A, B]); b, *_ = np.linalg.lstsq(X, E, rcond=None)
    X0 = np.column_stack([np.ones(len(E)), S]); b0, *_ = np.linalg.lstsq(X0, E, rcond=None)
    rss = ((E - X @ b) ** 2).sum(); rss0 = ((E - X0 @ b0) ** 2).sum(); return 1 - rss / rss0   # partial R2 beyond stage

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir"); ap.add_argument("--out-dir", default="results_v2"); ap.add_argument("--elements", nargs="*", default=["Mg", "P", "Cu"]); a = ap.parse_args()
    out = Path(a.out_dir); d, Y, ann, xl, meta = fa.load(a.data_dir); S = fa.stage_dummies(d).values; g = d.nhp_id.values
    for el in a.elements:
        r = pd.read_csv(out / "tri" / f"protein_pairs_{el}.csv").head(20); E = d[el].values
        full, minr, worst = [], [], []
        for _, row in r.iterrows():
            A = Y[row.protein_A].values; B = Y[row.protein_B].values; full.append(r2_pair(E, A, B, S)); vals = []
            for an in np.unique(g):
                k = g != an; vals.append(r2_pair(E[k], A[k], B[k], S[k]))
            minr.append(min(vals)); worst.append(np.unique(g)[int(np.argmin(vals))])
        r["partialR2_raw_all_samples"] = full; r["LOAO_min_partialR2"] = minr; r["LOAO_worst_animal"] = worst
        r.to_csv(out / "tri" / f"protein_pairs_{el}_top20_followup.csv", index=False, float_format="%.4g")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
        for ax, (_, row) in zip(axes, r.head(3).iterrows()):
            sa, sb = np.sign(row.partial_r_E_A_given_B), np.sign(row.partial_r_E_B_given_A)
            diff = sa * fa.z(Y[row.protein_A].values) + sb * fa.z(Y[row.protein_B].values)
            for sg in fa.STAGES:
                m = (d.ART_stage == sg).values; ax.scatter(diff[m], E[m], c=COL[sg], marker=MK[sg], s=40, ec=SURF)
            op = lambda s: "+" if s > 0 else "-"
            ax.set_xlabel(f"{'' if sa > 0 else '-'}z({row.label_A}) {op(sb)} z({row.label_B})"); ax.set_ylabel(el)
            ax.set_title(f"{row.label_A} vs {row.label_B}\nr(A,B)={row.r_A_B:.2f}  FWER p={row.FWER_p:.2f}", fontsize=9)
        fig.suptitle(f"{el}: top protein pairs - a difference (or sum) of two co-varying proteins tracks the element", y=1.0); fig.tight_layout(); fig.savefig(out / f"fig8_top_pairs_{el}.png", dpi=130); plt.close(fig)
        print(el, r[["label_A", "label_B", "FWER_p", "partialR2_raw_all_samples", "LOAO_min_partialR2", "LOAO_worst_animal"]].head(8).round(3).to_string())
