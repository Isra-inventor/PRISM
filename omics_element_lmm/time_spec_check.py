"""Robustness of the within-animal element signal to how treatment time is modelled (parametric t, which calibrated well)."""
import argparse, numpy as np, pandas as pd
from scipy import stats
import full_analysis as fa, within_animal_models as W
from lmm_engine import bh
ap = argparse.ArgumentParser(); ap.add_argument("--data-dir"); ap.add_argument("--elements", nargs="*", default=["Mg", "P", "Cu", "Na", "Fe", "Ni"]); a = ap.parse_args()
d, Y, ann, xl, meta = fa.load(a.data_dir); D = W.Data(d, Y); w = D.d.week.values; lt = np.log2(w + 1.0); lt = (lt - lt.mean()) / lt.std(ddof=1)
st = pd.get_dummies(D.d.ART_stage).astype(float).values[:, 1:]
specs = {"no time term": np.zeros((D.N, 0)), "linear log-time": lt[:, None], "quadratic log-time (default)": np.column_stack([lt, lt ** 2]), "cubic log-time": np.column_stack([lt, lt ** 2, lt ** 3]),
         "ART stage (3 dummies)": st, "stage + quadratic": np.column_stack([st, lt, lt ** 2])}
rows = []
for el in a.elements:
    for nm, S in specs.items():
        D.T = D._within(S) if S.shape[1] else np.zeros((D.N, 0)); o = D.fit(D.element(el), het=False)
        rows.append(dict(element=el, time_model=nm, p_lt_0_001=int((o["p"] < .001).sum()), q_lt_0_05_parametric=int((bh(o["p"]) < .05).sum()), expected_p_lt_0_001_by_chance=round(len(o["p"]) * .001, 1)))
r = pd.DataFrame(rows); print(r.pivot(index="time_model", columns="element", values="p_lt_0_001").to_string()); print("\nq<0.05 (parametric):"); print(r.pivot(index="time_model", columns="element", values="q_lt_0_05_parametric").to_string())
r.to_csv("results_v2/within/time_spec_robustness.csv", index=False)
