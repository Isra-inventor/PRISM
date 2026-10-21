"""
Per-animal protein ~ element associations, then comparison across animals.   (all samples kept)

Stage 1  (within each animal with >=4 samples, ordered by week; no stage/time term, only that animal's own samples)
          per protein: Spearman rho(protein, element) and OLS slope (log2 protein per 1 overall SD of element) with SE.
Stage 2  across animals
   a) random-effects meta-analysis of the per-animal slopes (DerSimonian-Laird + Hartung-Knapp t, k-1 df), Cochran Q, I2, tau2
   b) permutation test of the mean per-animal Spearman rho: element values shuffled WITHIN each animal (null is the same for every
      protein without ties, so it is computed once with 2e6 draws and p-values are exact up to Monte-Carlo error); BH-FDR over proteins
   c) leave-one-animal-out replication: choose the top-N proteins from the other animals, test the sign/rank agreement in the held-out animal
      (null: element shuffled within the held-out animal)
   d) similarity of the whole per-animal association profiles between animals (correlation of rho vectors across all proteins) vs permutation null
usage: python per_animal_assoc.py --data-dir D --out-dir results_v2 --element Cu
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import full_analysis as fa
from lmm_engine import bh

rng = np.random.default_rng(7)


def rank_cols(M):
    return np.apply_along_axis(stats.rankdata, 0, M)


def null_mean_rho(sizes, B=2_000_000):
    """null distribution of mean_a rho_a (exact shuffling within animals)."""
    tot = np.zeros(B, dtype=np.float32)
    for n in sizes:
        r = np.arange(n, dtype=np.float32); r = (r - r.mean()) / np.sqrt(((r - r.mean()) ** 2).sum())
        perm = np.argsort(rng.random((B, n)), axis=1)
        tot += (r[perm] * r[None, :]).sum(1)
    return tot / len(sizes)


def stage1(d, Y, el, animals):
    sd_el = d[el].std(ddof=1); out = {}
    for a in animals:
        m = (d.nhp_id == a).values; c = d[el].values[m]; n = m.sum(); Ya = Y.values[m]
        cc = (c - c.mean()) / sd_el; Yc = Ya - Ya.mean(0); Sxx = (cc ** 2).sum()
        b = (cc[:, None] * Yc).sum(0) / Sxx; sse = (Yc ** 2).sum(0) - b ** 2 * Sxx
        se = np.sqrt(np.maximum(sse, 1e-12) / (n - 2) / Sxx)
        rho = np.corrcoef(rank_cols(Ya).T, stats.rankdata(c))[-1, :-1] if False else None
        Rz = rank_cols(Ya); Rz = (Rz - Rz.mean(0)) / np.sqrt(((Rz - Rz.mean(0)) ** 2).sum(0) + 1e-12)
        rc = stats.rankdata(c); rc = (rc - rc.mean()) / np.sqrt(((rc - rc.mean()) ** 2).sum())
        out[a] = dict(b=b, se=se, rho=Rz.T @ rc, n=n, Rz=Rz, rc=rc, weeks=d.week.values[m])
    return out


def meta(s1, animals):
    B = np.vstack([s1[a]["b"] for a in animals]); SE = np.vstack([s1[a]["se"] for a in animals]); k = len(animals)
    w = 1 / SE ** 2; fe = (w * B).sum(0) / w.sum(0); Q = (w * (B - fe) ** 2).sum(0)
    C = w.sum(0) - (w ** 2).sum(0) / w.sum(0); tau2 = np.maximum(0, (Q - (k - 1)) / C)
    ws = 1 / (SE ** 2 + tau2); est = (ws * B).sum(0) / ws.sum(0)
    qhk = (ws * (B - est) ** 2).sum(0) / (k - 1) / ws.sum(0); se = np.sqrt(np.maximum(qhk, 1 / ws.sum(0) * 0 + qhk) )
    se = np.sqrt(np.maximum(qhk, 1.0 / ws.sum(0)))        # HK with the usual max(1,.) safeguard
    p = 2 * stats.t.sf(np.abs(est / se), k - 1); I2 = np.maximum(0, (Q - (k - 1)) / Q)
    return est, se, p, Q, stats.chi2.sf(Q, k - 1), I2, tau2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir"); ap.add_argument("--out-dir", default="results_v2"); ap.add_argument("--element", default="Cu")
    ap.add_argument("--min-n", type=int, default=4); ap.add_argument("--topn", type=int, default=100); a = ap.parse_args()
    out = Path(a.out_dir) / "per_animal"; out.mkdir(parents=True, exist_ok=True); el = a.element
    d, Y, ann, xl, meta_ = fa.load(a.data_dir)
    cnt = d.nhp_id.value_counts(); animals = sorted([x for x, c in cnt.items() if c >= a.min_n]); skipped = {x: int(c) for x, c in cnt.items() if c < a.min_n}
    print("animals used:", {x: int(cnt[x]) for x in animals}, "| not usable (single sample, no within-animal information):", skipped)
    s1 = stage1(d, Y, el, animals); k = len(animals)
    res = pd.DataFrame({"feature": Y.columns, "label": ann.loc[Y.columns, "label"].values})
    for x in animals:
        res[f"rho_{x}"] = s1[x]["rho"]; res[f"slope_{x}"] = s1[x]["b"]
    R = np.vstack([s1[x]["rho"] for x in animals]); res["mean_rho"] = R.mean(0); res["n_animals_same_sign"] = np.maximum((R > 0).sum(0), (R < 0).sum(0)); res["rho_SD_between_animals"] = R.std(0, ddof=1)
    est, se, p, Q, pQ, I2, tau2 = meta(s1, animals)
    res["meta_slope"] = est; res["meta_se_HK"] = se; res["meta_p"] = p; res["meta_q"] = bh(p); res["Q"] = Q; res["Q_p"] = pQ; res["I2"] = I2; res["tau2"] = tau2
    sizes = [s1[x]["n"] for x in animals]; null = null_mean_rho(sizes); an = np.abs(null); sn = np.sort(an)
    res["perm_p_mean_rho"] = (1 + len(sn) - np.searchsorted(sn, np.abs(res.mean_rho.values) - 1e-9, side="left")) / (1 + len(sn)); res["perm_q_mean_rho"] = bh(res["perm_p_mean_rho"].values)
    res = res.sort_values("perm_p_mean_rho"); res.to_csv(out / f"per_animal_{el}.csv", index=False, float_format="%.4g")
    print(f"proteins with mean rho permutation q<0.05: {(res.perm_q_mean_rho < .05).sum()} | q<0.10: {(res.perm_q_mean_rho < .10).sum()} | meta (HK) q<0.05: {(res.meta_q < .05).sum()}")
    print(f"null sd of mean rho {null.std():.3f}; observed SD of mean rho across proteins {res.mean_rho.std():.3f}  (ratio {res.mean_rho.std()/null.std():.2f}; 1 = nothing beyond chance)")
    print("median I2 among proteins with perm q<0.05:", float(res[res.perm_q_mean_rho < .05].I2.median()) if (res.perm_q_mean_rho < .05).any() else None)
    sig = res[res.perm_q_mean_rho < .05]
    if len(sig): print("same-sign in all", k, "animals:", int((sig.n_animals_same_sign == k).sum()), "of", len(sig))
    # (c) leave-one-animal-out replication
    rows = []; Rfull = R
    for h, hname in enumerate(animals):
        others = [i for i in range(k) if i != h]; m_oth = Rfull[others].mean(0); top = np.argsort(-np.abs(m_oth))[:a.topn]; sgn = np.sign(m_oth[top])
        obs = float(np.mean(Rfull[h, top] * sgn)); agree = float(np.mean(np.sign(Rfull[h, top]) == sgn))
        Rz, rc = s1[hname]["Rz"][:, top], s1[hname]["rc"]; nn = len(rc); nullv = []
        for _ in range(3000):
            pr = rc[rng.permutation(nn)]; nullv.append(float(np.mean((Rz.T @ pr) * sgn)))
        nullv = np.array(nullv); pv = (1 + (nullv >= obs).sum()) / (1 + len(nullv))
        rows.append(dict(held_out_animal=hname, n_top=a.topn, mean_oriented_rho_in_held_out=obs, pct_same_sign_in_held_out=100 * agree, null_mean=float(nullv.mean()), perm_p=pv))
    rep = pd.DataFrame(rows); rep.to_csv(out / f"replication_leave_one_animal_out_{el}.csv", index=False, float_format="%.4g"); print(rep.round(3).to_string())
    # (d) similarity of association profiles between animals
    def sim(Rm):
        return np.corrcoef(Rm)
    S_obs = sim(Rfull); iu = np.triu_indices(k, 1); nullS = []
    for _ in range(100):
        Rn = []
        for x in animals:
            pr = s1[x]["rc"][rng.permutation(s1[x]["n"])]; Rn.append(s1[x]["Rz"].T @ pr)
        nullS.append(np.corrcoef(np.vstack(Rn))[iu])
    nullS = np.array(nullS); S = pd.DataFrame(S_obs, index=animals, columns=animals); S.to_csv(out / f"animal_profile_similarity_{el}.csv", float_format="%.3f")
    print("\nproteome-wide similarity of Cu-association profiles between animals (Pearson of rho vectors):\n", S.round(2).to_string())
    print("observed mean off-diagonal %.3f | null mean %.3f (sd %.3f, max over 100 perms %.3f)" % (S_obs[iu].mean(), nullS.mean(), nullS.mean(1).std(), nullS.mean(1).max()))
    np.save(out / f"_null_{el}.npy", null[:100000])


if __name__ == "__main__":
    main()
