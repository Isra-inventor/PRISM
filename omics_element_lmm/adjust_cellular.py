"""Does the within-animal element signal survive adjustment for cell-content proxies (red-cell score, platelet score)?
usage: python adjust_cellular.py --data-dir D --out-dir results_v2 --elements Mg P ..."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import full_analysis as fa
import within_animal_models as W
from lmm_engine import bh

PLATELET = ["PF4", "PPBP", "GP1BA", "GP1BB", "GP9", "ITGA2B", "ITGB3", "SELP", "THBS1", "TUBB1", "CD9", "CLEC1B", "GP6", "FLNA", "TLN1", "PDGFA", "PDGFB", "F13A1", "VCL", "TPM4"]
RBC = ["HBB", "HBD", "CA1", "CAT", "BLVRB", "FECH", "ALAS2", "SOD1", "PRDX2", "SLC4A1", "G6PD"]

def score(Y, ann, names):
    idx = [f for g in names for f in ann.index[ann.label == g][:1]]
    return Y[idx].apply(fa.z).mean(axis=1), [ann.loc[i, "label"] for i in idx]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir"); ap.add_argument("--out-dir", default="results_v2"); ap.add_argument("--elements", nargs="*", default=["Mg", "P", "Cu", "Na"]); ap.add_argument("--B", type=int, default=100); a = ap.parse_args()
    d, Y, ann, xl, meta = fa.load(a.data_dir); D = W.Data(d, Y); out = Path(a.out_dir) / "within"
    # scores computed WITHOUT the proteins that are element hits are not removed here (they are marker-defined, independent of the elements)
    pl, plnames = score(Y, ann, PLATELET); rb, rbnames = score(Y, ann, RBC); print("platelet markers used:", plnames); print("red-cell markers used:", rbnames)
    covs = {"platelet": pl.loc[D.d.index].values, "redcell": rb.loc[D.d.index].values}
    from scipy import stats
    for el in a.elements:
        e = D.element(el); sd = e.std(ddof=1); ew = D._within((e / sd)[:, None])[:, 0]
        print(f"\n{el}: within-animal correlation of element with platelet score r={np.corrcoef(ew, D._within(covs['platelet'][:,None])[:,0])[0,1]:+.2f}, red-cell score r={np.corrcoef(ew, D._within(covs['redcell'][:,None])[:,0])[0,1]:+.2f}")
        for name, cv in [("none", []), ("platelet", ["platelet"]), ("redcell", ["redcell"]), ("platelet+redcell", ["platelet", "redcell"])]:
            T0 = D.T.copy()
            if cv: D.T = np.column_stack([T0] + [D._within(covs[c][:, None]) for c in cv])
            o = D.fit(e); nt, nf = W.perm_null(D, el, a.B, 5); p = W.emp_p(nt, np.abs(o["t"])); q = bh(p); D.T = T0
            print(f"   adjusted for {name:17s}: q<0.05 = {int((q<.05).sum()):4d} | q<0.10 = {int((q<.10).sum()):4d} | p<0.001 = {int((p<.001).sum()):4d}")
