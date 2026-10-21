"""Proteins associated with the element in >=k animals (same direction). Per-animal association = exact within-animal permutation p <= alpha.
Null = element shuffled within each animal (1000 draws, all proteins at once) -> distribution of the number of proteins meeting the criterion.
usage: python per_animal_k_of_5.py --data-dir D --out-dir results_v2 --element Cu --alpha 0.10"""
import argparse, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import full_analysis as fa
from per_animal_assoc import rank_cols

rng = np.random.default_rng(11)

def exact_table(n):
    r = np.arange(n, dtype=float); r = (r - r.mean()) / np.sqrt(((r - r.mean()) ** 2).sum())
    vals = np.array([ (r[list(p)] * r).sum() for p in itertools.permutations(range(n)) ]); return np.sort(np.abs(vals))

def pvals(rho, tab):
    return (len(tab) - np.searchsorted(tab, np.abs(rho) - 1e-9, side="left")) / len(tab)

def classify(Rho, tabs, alpha):
    """matrix animals x proteins of -1/0/+1 : significant negative / none / significant positive"""
    S = np.zeros_like(Rho, dtype=int)
    for i, tab in enumerate(tabs): S[i] = np.where(pvals(Rho[i], tab) <= alpha, np.sign(Rho[i]), 0)
    return S

def counts(S):
    pos, neg = (S > 0).sum(0), (S < 0).sum(0); return np.maximum(pos, neg), (pos > 0) & (neg > 0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir"); ap.add_argument("--out-dir", default="results_v2"); ap.add_argument("--element", default="Cu"); ap.add_argument("--alpha", type=float, default=0.10); a = ap.parse_args()
    d, Y, ann, xl, meta = fa.load(a.data_dir); el = a.element; out = Path(a.out_dir) / "per_animal"
    cnt = d.nhp_id.value_counts(); animals = sorted([x for x, c in cnt.items() if c >= 4]); k = len(animals)
    tabs = [exact_table(int(cnt[x])) for x in animals]
    print("smallest possible exact two-sided p per animal:", {x: round(float(1 - np.searchsorted(t, t.max() - 1e-9) / len(t)) if False else float((len(t) - np.searchsorted(t, t.max() - 1e-9, side='left')) / len(t)), 4) for x, t in zip(animals, tabs)})
    Rz, rc, idx = [], [], []
    for x in animals:
        m = (d.nhp_id == x).values; Ya = rank_cols(Y.values[m]); Ya = (Ya - Ya.mean(0)) / np.sqrt(((Ya - Ya.mean(0)) ** 2).sum(0) + 1e-12)
        c = stats.rankdata(d[el].values[m]); c = (c - c.mean()) / np.sqrt(((c - c.mean()) ** 2).sum()); Rz.append(Ya); rc.append(c)
    Rho = np.vstack([Rz[i].T @ rc[i] for i in range(k)]); S = classify(Rho, tabs, a.alpha); same, conflict = counts(S)
    obs = {kk: int((same >= kk).sum()) for kk in range(1, k + 1)}
    nulls = {kk: [] for kk in range(1, k + 1)}
    for _ in range(1000):
        Rn = np.vstack([Rz[i].T @ rc[i][rng.permutation(len(rc[i]))] for i in range(k)]); sn, _ = counts(classify(Rn, tabs, a.alpha))
        for kk in nulls: nulls[kk].append(int((sn >= kk).sum()))
    rows = []
    for kk in range(1, k + 1):
        nv = np.array(nulls[kk]); rows.append(dict(min_animals_same_direction=kk, observed_proteins=obs[kk], null_mean=nv.mean(), null_95pct=np.quantile(nv, .95), null_max=nv.max(), perm_p=(1 + (nv >= obs[kk]).sum()) / 1001))
    tab = pd.DataFrame(rows); print(f"\nelement {el}, per-animal association = exact permutation p <= {a.alpha}; animals: {dict(zip(animals, [int(cnt[x]) for x in animals]))}"); print(tab.round(2).to_string(index=False))
    res = pd.DataFrame({"feature": Y.columns, "label": ann.loc[Y.columns, "label"].values})
    for i, x in enumerate(animals): res[f"rho_{x}"] = Rho[i]; res[f"sig_{x}"] = S[i]
    res["n_animals_same_direction"] = same; res["direction_conflict"] = conflict; res["mean_rho"] = Rho.mean(0)
    res = res[res.n_animals_same_direction >= 2].sort_values(["n_animals_same_direction", "mean_rho"], key=lambda s: s if s.name == "n_animals_same_direction" else s.abs(), ascending=False)
    res.to_csv(out / f"proteins_in_ge2_animals_{el}.csv", index=False, float_format="%.3g"); tab.to_csv(out / f"k_of_5_vs_chance_{el}.csv", index=False, float_format="%.4g")
    print("\nproteins associated (p<=%.2f, same direction) in >=3 animals:" % a.alpha, int((res.n_animals_same_direction >= 3).sum())); print(res.head(12)[["label"] + [f"rho_{x}" for x in animals] + ["n_animals_same_direction"]].round(2).to_string(index=False))
