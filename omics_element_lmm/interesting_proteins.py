"""Rank candidate proteins per element by combining
   (a) per-animal consistency (per_animal_<E>.csv): same sign in all 5 animals, size of the per-animal rho, permutation q of the mean rho
   (b) pooled-model evidence (per_element/proteomics_<E>.csv): within-animal p (M4), element p (M1), rank-normalised element p (M8)
 Tier 1: same sign in all 5 animals, every |rho| >= 0.4, mean |rho| >= 0.7, perm q(mean rho) < 0.25, AND pooled within-animal p (M4) < 0.01
 Tier 2: same sign in >= 4 animals, mean |rho| >= 0.6, pooled p (M4 within or M1) < 0.01
 The same rules are applied to Ni and Fe-corrected Ni (negative controls) to show how many 'candidates' chance produces.
usage: python interesting_proteins.py --data-dir D --out-dir results_v2"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import full_analysis as fa

A = ["DEAB", "DEL5", "GA43", "T544", "T623"]

def table(out, el):
    pa = pd.read_csv(out / "per_animal" / f"per_animal_{el}.csv").set_index("feature")
    pe = pd.read_csv(out / "per_element" / f"proteomics_{el}.csv").set_index("feature")
    t = pa.join(pe[["TargetFullName", "UniProt", "M1_stage_beta_E", "M1_stage_p_E", "M4_wb_beta_within", "M4_wb_p_within", "M4_wb_q_within", "M8_rank_p_E", "M3_week_p_E", "M7_hemo_p_E", "is_hemolysis_marker"]], how="left", rsuffix="_pe")
    R = t[[f"rho_{a}" for a in A]].values
    sgn = np.sign(t.mean_rho.values)[:, None]
    t["min_oriented_rho"] = (R * sgn).min(1); t["mean_abs_rho"] = np.abs(R.mean(1)); t["n_same_sign"] = t.n_animals_same_sign
    t["tier"] = 0
    t2 = (t.n_same_sign >= 4) & (t.mean_abs_rho >= .6) & ((t.M4_wb_p_within < .01) | (t.M1_stage_p_E < .01)); t.loc[t2, "tier"] = 2
    t1 = (t.n_same_sign == 5) & (t.min_oriented_rho >= .4) & (t.mean_abs_rho >= .7) & (t.perm_q_mean_rho < .25) & (t.M4_wb_p_within < .01); t.loc[t1, "tier"] = 1
    t.insert(0, "element", el); t = t.reset_index()
    return t

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default="results_v2"); a = ap.parse_args(); out = Path(a.out_dir)
    els = ["Na", "Mg", "P", "K", "Ca", "Fe", "Cu", "Zn", "Ni", "Ni_resFe"]; allt = []; rows = []
    for el in els:
        t = table(out, el); allt.append(t)
        rows.append(dict(element=el, tier1=int((t.tier == 1).sum()), tier2=int((t.tier == 2).sum()), all5_same_sign=int((t.n_same_sign == 5).sum()),
                         all5_and_meanabs_ge_0_7=int(((t.n_same_sign == 5) & (t.mean_abs_rho >= .7)).sum())))
    S = pd.DataFrame(rows); print(S.to_string(index=False)); S.to_csv(out / "per_animal" / "candidate_counts_by_element.csv", index=False)
    T = pd.concat(allt); cand = T[T.tier > 0].sort_values(["element", "tier", "perm_q_mean_rho"])
    keep = ["element", "tier", "feature", "label", "TargetFullName", "UniProt"] + [f"rho_{a}" for a in A] + ["mean_rho", "perm_q_mean_rho", "I2", "M4_wb_p_within", "M1_stage_p_E", "M8_rank_p_E", "M3_week_p_E", "M7_hemo_p_E", "is_hemolysis_marker"]
    cand[keep].to_csv(out / "per_animal" / "candidate_proteins_all_elements.csv", index=False, float_format="%.3g")
    # proteins that are candidates for >1 element
    m = cand[~cand.element.isin(["Ni", "Ni_resFe"])].groupby("feature").agg(label=("label", "first"), elements=("element", lambda s: ",".join(sorted(set(s)))), n_el=("element", "nunique")).query("n_el>1")
    print("\ncandidates shared by several elements:\n", m.sort_values("n_el", ascending=False).head(15).to_string())
    for el in ["Na", "Mg", "P"]:
        c = cand[(cand.element == el) & (cand.tier == 1)]
        print(f"\n--- {el} tier 1 ({len(c)})"); print(c[["label"] + [f"rho_{a}" for a in A] + ["mean_rho", "perm_q_mean_rho", "M4_wb_p_within"]].head(25).round(3).to_string(index=False))
