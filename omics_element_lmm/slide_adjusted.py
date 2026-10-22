"""Slide (microarray) batch check: within-animal model with a separate baseline for every animal x slide combination.
Within-group comparisons then never cross a slide boundary.   usage: python slide_adjusted.py --data-dir D --out-dir results_v2"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import full_analysis as fa, within_animal_models as W
from lmm_engine import bh

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir"); ap.add_argument("--out-dir", default="results_v2"); ap.add_argument("--B", type=int, default=500); a = ap.parse_args()
    d, Y, ann, xl, meta = fa.load(a.data_dir); m = pd.read_csv(Path(a.data_dir) / "sample_metadata.tsv", sep="\t").set_index("nhp_week_index").loc[d.index]
    base = W.Data(d, Y); d2 = d.copy(); d2["nhp_id"] = d.nhp_id + "|" + m.SlideId.astype(str).values; D = W.Data(d2, Y)
    print(f"animal-only groups: {base.k} animals, N={base.N} | animal x slide groups with >=2 samples: {D.k}, N={D.N} (samples that are alone on their slide drop out of the within-slide comparison)")
    print({g: int((D.g == g).sum()) for g in D.animals})
    rows = []
    for el in fa.ELS:
        o0 = base.fit(base.element(el), het=False); nt0, _ = W.perm_null(base, el, 1, 1) if False else (None, None)
        e = D.element(el); o = D.fit(e, het=False); T = []; rng = np.random.default_rng(sum(map(ord, el)))
        for _ in range(a.B):
            ee = e.copy()
            for g in D.animals:
                r = D.g == g; ee[r] = rng.permutation(ee[r])
            T.append(np.abs(D.fit(ee, het=False)["t"]))
        nt = np.sort(np.concatenate(T)); p = W.emp_p(nt, np.abs(o["t"]))
        # same samples, animal-only baseline, for a like-for-like slope comparison
        ob = base.fit(base.element(el), het=False); common = pd.Series(ob["beta"], index=base.cols).corr(pd.Series(o["beta"], index=D.cols))
        rows.append(dict(element=el, p_lt_0_001_slide_adjusted=int((p < .001).sum()), q_lt_0_10_slide_adjusted=int((bh(p) < .10).sum()), expected_by_chance=round(len(p) * .001, 1), slope_correlation_with_unadjusted=round(common, 2)))
        print(rows[-1])
    pd.DataFrame(rows).to_csv(Path(a.out_dir) / "within" / "slide_adjusted_summary.csv", index=False)
