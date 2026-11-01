"""ComBat, parametric empirical Bayes, location and scale (Johnson, Li and Rabinovic 2007), with an
intercept-only model (`~1`), as proBatch calls it. Implemented from the published method; the
numerical conventions (pooled variance over n, the prior estimates, the convergence rule 1e-4,
features with zero variance inside a batch left unadjusted) follow the documented behaviour of
R's sva::ComBat so the two can be compared (oracle test). No sva code is included.

Precondition (v4 section 6.2): no missing values in the features it corrects and at least 2
values per batch.
"""

from __future__ import annotations

import numpy as np


class ComBatError(Exception):
    pass


def _aprior(d):
    m, s2 = d.mean(), d.var(ddof=1)
    return (2 * s2 + m ** 2) / s2


def _bprior(d):
    m, s2 = d.mean(), d.var(ddof=1)
    return (m * s2 + m ** 3) / s2


def _it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=1e-4, max_iter=10000):
    n = np.isfinite(sdat).sum(1)
    g_old, d_old = g_hat.copy(), d_hat.copy()
    for _ in range(max_iter):
        g_new = (t2 * n * g_hat + d_old * g_bar) / (t2 * n + d_old)
        sum2 = ((sdat - g_new[:, None]) ** 2).sum(1)
        d_new = (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)
        change = max(np.max(np.abs(g_new - g_old) / np.abs(g_old)), np.max(np.abs(d_new - d_old) / np.abs(d_old)))
        g_old, d_old = g_new, d_new
        if change <= conv:
            break
    return g_new, d_new


def combat(X, batch):
    """X: features x samples, no NaN. batch: one label per sample. -> (X corrected, info)."""
    X = np.asarray(X, float)
    if np.isnan(X).any():
        raise ComBatError("ComBat needs a complete matrix (no missing values).")
    batch = np.asarray([str(b) for b in batch])
    levels = sorted(set(batch.tolist()))
    if len(levels) < 2:
        raise ComBatError("ComBat needs at least 2 batches.")
    idx = [np.where(batch == lv)[0] for lv in levels]
    if min(len(i) for i in idx) < 2:
        raise ComBatError("Every batch needs at least 2 samples.")
    zero = np.zeros(X.shape[0], bool)
    for i in idx:
        zero |= X[:, i].var(1, ddof=1) == 0
    keep = ~zero
    dat = X[keep]
    p, n = dat.shape
    nb = np.array([len(i) for i in idx], float)
    design = np.zeros((n, len(levels)))
    for k, i in enumerate(idx):
        design[i, k] = 1.0
    B_hat = np.linalg.solve(design.T @ design, design.T @ dat.T)          # batches x p
    grand = (nb / n) @ B_hat
    var_pooled = ((dat - (design @ B_hat).T) ** 2) @ np.full(n, 1.0 / n)
    stand = np.outer(grand, np.ones(n))
    s = (dat - stand) / np.sqrt(var_pooled)[:, None]
    gamma_hat = np.linalg.solve(design.T @ design, design.T @ s.T)       # batches x p
    delta_hat = np.vstack([s[:, i].var(1, ddof=1) for i in idx])
    gamma_bar = gamma_hat.mean(1)
    t2 = gamma_hat.var(1, ddof=1)
    a = np.array([_aprior(d) for d in delta_hat])
    b = np.array([_bprior(d) for d in delta_hat])
    g_star, d_star = [], []
    for k, i in enumerate(idx):
        g, d = _it_sol(s[:, i], gamma_hat[k], delta_hat[k], gamma_bar[k], t2[k], a[k], b[k])
        g_star.append(g)
        d_star.append(d)
    out = s.copy()
    for k, i in enumerate(idx):
        out[:, i] = (s[:, i] - g_star[k][:, None]) / np.sqrt(d_star[k])[:, None]
    out = out * np.sqrt(var_pooled)[:, None] + stand
    Xc = X.copy()
    Xc[keep] = out
    return Xc, {"levels": levels, "n_per_batch": nb.astype(int).tolist(), "par_prior": True, "model": "~1",
                "features_not_adjusted_zero_variance": int(zero.sum())}
