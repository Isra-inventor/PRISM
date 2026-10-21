"""
Vectorised random-intercept linear mixed model (REML) for omics screening.

    y_f = X b_f + u_animal + e ,   u ~ N(0, s2*lam),  e ~ N(0, s2)

The design matrix X is identical for every feature (protein / metabolite), so
the REML profile likelihood over the variance ratio ``lam`` is evaluated for
ALL features at once on a grid.  Compared with fitting 11,000 separate
``statsmodels.MixedLM`` objects this

  * never crashes when the animal variance is ~0 (lam = 0 is just OLS),
  * uses REML (ML under-estimates variances -> anti-conservative p-values),
  * uses t-tests with finite df instead of z-tests (only 8 animals / 27 samples),
  * runs in seconds, which makes permutation calibration and looping over all
    elements cheap.
"""
import numpy as np
import pandas as pd
from scipy import stats

LAM_GRID = np.concatenate([[0.0], np.logspace(-3, 2.5, 80)])


def _group_index(groups):
    codes, uniq = pd.factorize(pd.Series(groups).astype(str))
    return codes, len(uniq)


def _gls_stats(X, Y, codes, G, lam):
    """GLS beta, (X'WX)^-1 and weighted RSS for every column of Y at ratio lam."""
    n = len(codes)
    ng = np.bincount(codes, minlength=G).astype(float)
    w = lam / (1.0 + lam * ng)                       # per-group shrinkage
    # W = I - Z diag(w) Z'   (per-group rank-1 correction)
    def Wmul(A):
        S = np.zeros((G, A.shape[1]))
        np.add.at(S, codes, A)
        return A - (w[:, None] * S)[codes]
    WX = Wmul(X)
    XtWX = X.T @ WX
    XtWX_inv = np.linalg.inv(XtWX)
    beta = XtWX_inv @ (WX.T @ Y)
    R = Y - X @ beta
    WR = Wmul(R)
    rss = np.einsum("ij,ij->j", R, WR)
    logdet_V = np.sum(np.log1p(lam * ng))
    logdet_XWX = np.linalg.slogdet(XtWX)[1]
    return beta, XtWX_inv, rss, logdet_V, logdet_XWX


def fit_reml(X, Y, groups, lam_grid=LAM_GRID):
    """
    X : (n, p) numpy design (with intercept), Y : (n, F) numpy, groups : len n.
    Returns dict of arrays: beta (p,F), se (p,F), lam (F,), s2 (F,), df, cov.
    """
    n, p = X.shape
    codes, G = _group_index(groups)
    crit = np.empty((len(lam_grid), Y.shape[1]))
    cache = []
    for i, lam in enumerate(lam_grid):
        beta, inv, rss, ldV, ldX = _gls_stats(X, Y, codes, G, lam)
        s2 = np.maximum(rss, 1e-300) / (n - p)
        crit[i] = (n - p) * np.log(s2) + ldV + ldX    # -2 REML (profiled s2)
        cache.append((beta, inv, s2))
    best = crit.argmin(axis=0)
    F = Y.shape[1]
    beta = np.empty((p, F)); se = np.empty((p, F)); s2o = np.empty(F)
    covs = np.empty((F, p, p))
    for i in np.unique(best):
        idx = np.where(best == i)[0]
        b, inv, s2 = cache[i]
        beta[:, idx] = b[:, idx]
        s2o[idx] = s2[idx]
        for j in idx:
            covs[j] = inv * s2[j]
        se[:, idx] = np.sqrt(np.outer(np.diag(inv), s2[idx]))
    return dict(beta=beta, se=se, lam=lam_grid[best], s2=s2o, cov=covs,
                at_boundary=(best == 0), codes=codes, G=G, n=n, p=p)


def term_tests(fit, X, term_idx, df_method="bw"):
    """
    t-test for one coefficient (term_idx int) -> (beta, se, t, df, p)
    df_method: 'bw' (default) = n-G-(p-1) ; 'resid' = n-p ; 'conservative' = min(n-p, G-1)
    """
    n, p, G = fit["n"], fit["p"], fit["G"]
    if isinstance(df_method, (int, float)):
        df = df_method
    else:
        df = {"resid": n - p, "conservative": max(1, min(n - p, G - 1)),
              "bw": bw_df(n, G, p)}[df_method]
    b, s = fit["beta"][term_idx], fit["se"][term_idx]
    t = b / s
    return b, s, t, df, 2 * stats.t.sf(np.abs(t), df)


def wald_f(fit, term_idxs, df_method="bw"):
    """Omnibus F test that several coefficients are jointly zero."""
    n, p, G = fit["n"], fit["p"], fit["G"]
    k = len(term_idxs)
    if isinstance(df_method, (int, float)):
        df2 = df_method
    else:
        df2 = {"resid": n - p, "conservative": max(1, min(n - p, G - 1)),
               "bw": bw_df(n, G, p)}[df_method]
    Fv = np.empty(fit["beta"].shape[1])
    for j in range(len(Fv)):
        b = fit["beta"][term_idxs, j]
        C = fit["cov"][j][np.ix_(term_idxs, term_idxs)]
        Fv[j] = b @ np.linalg.solve(C, b) / k
    return Fv, stats.f.sf(Fv, k, df2)


def bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty_like(pv); q[order] = np.minimum(ranked, 1)
    out[ok] = q
    return out


def bw_df(n, G, p):
    """Between-within style df: n - G - (p-1).  Empirically calibrated for this design
    (27 samples / 8 animals): permutation FPR 5.5% @0.05, 1.4% @0.01."""
    return max(3, n - G - (p - 1))
