# Derived from proBatch (Bioconductor; https://github.com/symbioticMe/proBatch, GitHub master,
# DESCRIPTION version 1.3.1, commit 524249e, April 2025).
# Original authors (DESCRIPTION): Jelena Cuklina, Chloe H. Lee, Patrick Pedrioli (aut);
# Yuliya Burankova (cre). Original license: GPL-3 (DESCRIPTION field; the repository has no
# separate license text file).
# Adapted to Python by PRISM, 2026-10-21.
# What changed: diagnostics run on PRISM's diagnostic matrix (v4 section 3) instead of filling
# missing values; PVCA uses statsmodels mixed models (the Bioconductor pvca package wraps lme4)
# with an eta-squared fallback when a model fails, and reports which was used per component.
"""Proteome-wide diagnostics: sample mean versus run order with batch boundaries, boxplot summaries,
sample correlation matrix, hierarchical clustering (Euclidean, complete), PCA of the transposed
matrix, PVCA (principal component threshold 0.6, variance-component threshold 0.01)."""

import numpy as np

from ...audit.diagnostic_matrix import pairwise_corr


def sample_means(Y, order=None, batch=None):
    with np.errstate(all="ignore"):
        m = np.nanmean(Y, axis=0)
    out = {"mean": m.tolist()}
    if order is not None:
        o = np.argsort(np.asarray(order, float), kind="mergesort")
        out["order"] = o.tolist()
        if batch is not None:
            b = [str(batch[j]) for j in o]
            out["batch_boundaries"] = [k for k in range(1, len(b)) if b[k] != b[k - 1]]
    return out


def boxplot_stats(Y):
    rows = []
    for j in range(Y.shape[1]):
        v = Y[np.isfinite(Y[:, j]), j]
        rows.append(None if not v.size else {k: float(x) for k, x in zip(("min", "q1", "median", "q3", "max"),
                                                                         np.quantile(v, [0, .25, .5, .75, 1]))})
    return rows


def correlation_matrix(Y):
    R, N = pairwise_corr(Y)
    return R, N


def hierarchical_clustering(Yc, method="complete", metric="euclidean"):
    """Yc: a complete diagnostic matrix (features x samples). -> leaf order and the merge table."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    Z = linkage(np.asarray(Yc, float).T, method=method, metric=metric)
    return {"order": leaves_list(Z).tolist(), "linkage": Z.tolist(), "method": method, "metric": metric}


def pca_transposed(Yc, k=5):
    """PCA of the transposed matrix (samples as observations), features centered."""
    A = np.asarray(Yc, float).T
    A = A - A.mean(0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    k = min(k, len(s))
    var = s ** 2 / max(A.shape[0] - 1, 1)
    return {"scores": (U[:, :k] * s[:k]).tolist(), "variance_explained": (var[:k] / var.sum()).tolist()}


def _n_pcs(pct, threshold):
    """pvcaBatchAssess: count PCs from the top until the cumulative share passes the threshold; at least 3."""
    counter, my_sum = 0, 1.0
    for v in pct[::-1]:
        my_sum -= v
        if my_sum <= threshold:
            counter += 1
    return max(3, counter)


def _terms(factors, interactions):
    t = [(f,) for f in factors]
    if interactions:
        t += [(a, b) for i, a in enumerate(factors) for b in factors[i + 1:]]
    return t


def _mixed(y, labels, terms):
    import warnings
    import pandas as pd
    import statsmodels.formula.api as smf
    df = pd.DataFrame({"y": y, "g": 1})
    vc = {}
    for t in terms:
        name = "__".join(t)
        df[name] = ["|".join(str(labels[f][i]) for f in t) for i in range(len(y))]
        vc[name] = f"0 + C({name})"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm("y ~ 1", df, groups="g", re_formula="0", vc_formula=vc)
        r = m.fit(reml=True)
    comps = np.clip(np.asarray(r.vcomp, float), 0, None)
    names = list(m.exog_vc.names)
    order = {n: c for n, c in zip(names, comps)}
    return [order["__".join(t)] for t in terms] + [float(r.scale)]


def _eta(y, labels, terms):
    out = []
    tot = ((y - y.mean()) ** 2).sum()
    for t in terms:
        key = ["|".join(str(labels[f][i]) for f in t) for i in range(len(y))]
        ss = 0.0
        for lv in set(key):
            ix = [i for i, k in enumerate(key) if k == lv]
            ss += len(ix) * (y[ix].mean() - y.mean()) ** 2
        out.append(ss / tot if tot > 0 else 0.0)
    s = sum(out)
    if s > 1:
        out = [x / s for x in out]
    return out + [max(0.0, 1 - sum(out))]


def pvca(Yc, labels, factors, pct_threshold=0.6, var_threshold=0.01, interactions=True):
    """Yc: complete matrix (features x samples); labels: {factor: [value per sample]}.
    -> {"weights": {term: share}, "pc_n", "method_per_pc", "small_terms"}."""
    A = np.asarray(Yc, float)
    A = A - A.mean(1, keepdims=True)
    C = np.corrcoef(A.T)
    ev, vec = np.linalg.eigh(C)
    o = np.argsort(ev)[::-1]
    ev, vec = ev[o], vec[:, o]
    ev = np.clip(ev, 0, None)
    pct = ev / ev.sum()
    pc_n = _n_pcs(pct, pct_threshold)
    terms = _terms(list(factors), interactions)
    M, methods = [], []
    for i in range(pc_n):
        y = vec[:, i]
        try:
            row = _mixed(y, labels, terms)
            methods.append("mixed_model_reml")
        except Exception:
            row = _eta(y, labels, terms)
            methods.append("eta_squared_fallback")
        row = np.asarray(row, float)
        row = row / row.sum() if row.sum() > 0 else row
        M.append(row * pct[i])
    s = np.sum(M, axis=0)
    w = s / s.sum()
    names = [":".join(t) for t in terms] + ["resid"]
    weights = dict(zip(names, w.tolist()))
    small = [k for k, v in weights.items() if v < var_threshold]
    if len(small) > 1:
        kept = {k: v for k, v in weights.items() if v >= var_threshold}
        kept["below " + str(var_threshold)] = sum(weights[k] for k in small)
        reported = kept
    else:
        reported = weights
    return {"weights": weights, "reported": reported, "pc_n": pc_n, "method_per_pc": methods,
            "pct_threshold": pct_threshold, "var_threshold": var_threshold, "interactions": interactions}
