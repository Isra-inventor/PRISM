"""
Within-animal element ~ protein analysis, designed for few animals / few samples per animal.

For every protein (log2) and element E, using the animals with >=2 samples (24 samples, 5 animals):

  common model   y ~ animal(fixed intercepts) + E + t + t^2          t = log2(week+1) (shared smooth time trend removed)
                 -> E coefficient = repeated-measures (within-animal) association, net of treatment time
  heterogeneity  y ~ animal + animal:E + t + t^2   vs common model  -> F test "do animals have different slopes?"
  moderation     residual variances shrunk across all proteins (empirical Bayes, limma-style) -> moderated t / F

Calibrated by permuting the element WITHIN animals.  Singletons (3 animals) carry no within-animal information.
usage: python within_animal_models.py --data-dir D --out-dir results_v2 [--elements Cu Fe ...] [--calibrate]
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats, special
import full_analysis as fa
from lmm_engine import bh

A_MIN = 2


def trigamma_inverse(x):
    x = np.asarray(x, float); y = 0.5 + 1 / x
    for _ in range(100):
        tri = special.polygamma(1, y); dif = tri * (1 - tri / x) / special.polygamma(2, y); y = y + dif
        if np.max(-dif / y) < 1e-8: break
    return y


def fit_prior(s2, d):
    """limma fitFDist for a common residual df d: returns (d0, s0^2)."""
    x = np.maximum(s2, 1e-12); z = np.log(x); e = z - special.digamma(d / 2) + np.log(d / 2)
    em = e.mean(); ev = e.var(ddof=1) - special.polygamma(1, d / 2)
    if ev > 0:
        d0 = 2 * float(trigamma_inverse(ev)); s0 = np.exp(em + special.digamma(d0 / 2) - np.log(d0 / 2))
    else:
        d0, s0 = np.inf, np.exp(em)
    return d0, s0


def shrink(s2, d, prior=None):
    d0, s0 = prior or fit_prior(s2, d)
    if np.isinf(d0): return np.full_like(s2, s0), d0, s0
    return (d0 * s0 + d * s2) / (d0 + d), d0, s0


class Data:
    def __init__(self, d, Y):
        cnt = d.nhp_id.value_counts(); self.animals = sorted([a for a, c in cnt.items() if c >= A_MIN])
        m = d.nhp_id.isin(self.animals).values; self.idx = np.where(m)[0]; self.g = d.nhp_id.values[m]
        self.k = len(self.animals); self.N = m.sum(); self.d = d[m]
        self.Yw = self._within(Y.values[m]); t = np.log2(d.week.values[m] + 1.0); t = (t - t.mean()) / t.std(ddof=1)
        self.T = np.column_stack([self._within(t[:, None]), self._within((t ** 2)[:, None])])
        self.cols = Y.columns; self.sdE = None
        self.Q = {a: (self.g == a) for a in self.animals}

    def _within(self, M):
        M = np.asarray(M, float).copy()
        for a in self.animals:
            r = self.g == a; M[r] -= M[r].mean(0)
        return M

    def element(self, el):
        return self.d[el].values.astype(float)

    def fit(self, e, het=True):
        """e = raw element values for the N samples (already permuted if desired)."""
        N, k = self.N, self.k; sd = e.std(ddof=1)
        ew = self._within((e / sd)[:, None])[:, 0]
        Xr = np.column_stack([ew, self.T]); G = np.linalg.inv(Xr.T @ Xr); XtY = Xr.T @ self.Yw
        b = G @ XtY; rss = (self.Yw ** 2).sum(0) - (b * XtY).sum(0); dr = N - k - Xr.shape[1]; s2 = rss / dr
        s2m, d0, _ = shrink(s2, dr); se = np.sqrt(s2m * G[0, 0]); tm = b[0] / se; dfm = dr + (d0 if np.isfinite(d0) else 1e6)
        out = dict(beta=b[0], se=se, t=tm, p=2 * stats.t.sf(np.abs(tm), dfm), d_res=dr, d0=d0,
                   t_plain=b[0] / np.sqrt(s2 * G[0, 0]), p_plain=2 * stats.t.sf(np.abs(b[0] / np.sqrt(s2 * G[0, 0])), dr),
                   partial_r=tm / np.sqrt(tm ** 2 + dfm), rss_red=rss)
        if het:
            Ea = np.column_stack([np.where(self.Q[a], ew, 0.0) for a in self.animals])
            Xf = np.column_stack([Ea, self.T]); Gf = np.linalg.inv(Xf.T @ Xf); XtYf = Xf.T @ self.Yw; bf = Gf @ XtYf
            rssf = (self.Yw ** 2).sum(0) - (bf * XtYf).sum(0); df_ = N - k - Xf.shape[1]; s2f = rssf / df_
            s2fm, d0f, _ = shrink(s2f, df_); Fv = ((rss - rssf) / (k - 1)) / s2fm; dff = df_ + (d0f if np.isfinite(d0f) else 1e6)
            out.update(F_het=Fv, p_het=stats.f.sf(Fv, k - 1, dff), p_het_plain=stats.f.sf(((rss - rssf) / (k - 1)) / s2f, k - 1, df_),
                       slopes=bf[:k], slope_se=np.sqrt(np.outer(np.diag(Gf)[:k], s2fm)))
        return out


def calibrate(D, el, B=100, seed=3):
    rng = np.random.default_rng(seed); e0 = D.element(el); P = {"mod": [], "plain": [], "het": [], "het_plain": []}
    for _ in range(B):
        e = e0.copy()
        for a in D.animals:
            r = D.g == a; e[r] = rng.permutation(e[r])
        o = D.fit(e); P["mod"].append(o["p"]); P["plain"].append(o["p_plain"]); P["het"].append(o["p_het"]); P["het_plain"].append(o["p_het_plain"])
    rows = []
    for k_, v in P.items():
        v = np.concatenate(v); rows.append(dict(element=el, test=k_, fpr_05=(v < .05).mean(), fpr_01=(v < .01).mean(), fpr_001=(v < .001).mean()))
    return pd.DataFrame(rows)


def perm_null(D, el, B, seed):
    rng = np.random.default_rng(seed); e0 = D.element(el); T, F = [], []
    for _ in range(B):
        e = e0.copy()
        for a in D.animals:
            r = D.g == a; e[r] = rng.permutation(e[r])
        o = D.fit(e); T.append(np.abs(o["t"])); F.append(o["F_het"])
    return np.sort(np.concatenate(T)), np.sort(np.concatenate(F))


def emp_p(null_sorted, obs):
    return (1 + len(null_sorted) - np.searchsorted(null_sorted, obs, side="left")) / (1 + len(null_sorted))


def run(D, ann, el, B=200):
    o = D.fit(D.element(el)); nt, nf = perm_null(D, el, B, seed=sum(ord(c) for c in el) * 7919); r = pd.DataFrame({"feature": D.cols, "label": ann.loc[D.cols, "label"].values})
    for c in ["TargetFullName", "UniProt"]:
        if c in ann: r[c] = ann.loc[D.cols, c].values
    pt = emp_p(nt, np.abs(o["t"])); pf = emp_p(nf, o["F_het"])
    r["within_slope"] = o["beta"]; r["se_moderated"] = o["se"]; r["t_moderated"] = o["t"]; r["p_parametric"] = o["p"]; r["p"] = pt; r["q"] = bh(pt); r["partial_r"] = o["partial_r"]
    r["q_parametric"] = bh(o["p"])
    for i, a in enumerate(D.animals): r[f"slope_{a}"] = o["slopes"][i]; r[f"slope_se_{a}"] = o["slope_se"][i]
    r["F_heterogeneity"] = o["F_het"]; r["p_heterogeneity_parametric"] = o["p_het"]; r["p_heterogeneity"] = pf; r["q_heterogeneity"] = bh(pf)
    sgn = np.sign(o["beta"])[:, None]; r["n_animals_same_sign"] = (np.sign(o["slopes"].T) == sgn).sum(1); r["prior_df_d0"] = o["d0"]
    return r.sort_values("p").reset_index(drop=True), o


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir"); ap.add_argument("--out-dir", default="results_v2"); ap.add_argument("--elements", nargs="*", default=fa.ELS)
    ap.add_argument("--calibrate", action="store_true"); ap.add_argument("--B", type=int, default=200); a = ap.parse_args()
    d, Y, ann, xl, meta = fa.load(a.data_dir); D = Data(d, Y); out = Path(a.out_dir) / "within"; out.mkdir(parents=True, exist_ok=True)
    print(f"within-animal analysis: {D.k} animals {dict(pd.Series(D.g).value_counts())}, N={D.N}, proteins={Y.shape[1]}")
    if a.calibrate:
        cal = pd.concat([calibrate(D, el, a.B) for el in a.elements]); cal.to_csv(out / "calibration_permutation.csv", index=False, float_format="%.4f"); print(cal.round(4).to_string(index=False)); raise SystemExit
    rows = []
    for el in a.elements:
        r, o = run(D, ann, el, a.B); r.to_csv(out / f"within_{el}.csv", index=False, float_format="%.4g")
        lam = float(np.median(stats.chi2.isf(r.p_parametric.clip(1e-300), 1)) / stats.chi2.ppf(.5, 1))
        se_med = float(np.median(o["se"])); mde = (stats.t.isf(5e-5, o["d_res"] + min(o["d0"], 1e3)) + 0.84) * se_med
        rows.append(dict(element=el, q_lt_0_05=int((r.q < .05).sum()), q_lt_0_10=int((r.q < .10).sum()), p_lt_0_001=int((r.p < .001).sum()), lambda_gc=lam,
                         parametric_q_lt_0_05=int((r.q_parametric < .05).sum()), het_q_lt_0_05=int((r.q_heterogeneity < .05).sum()),
                         prior_d0=o["d0"], min_detectable_slope_80pct_power_p1e4=mde))
        print(rows[-1])
    pd.DataFrame(rows).to_csv(out / "summary_within.csv", index=False, float_format="%.4g")
