# Derived from proBatch (Bioconductor; https://github.com/symbioticMe/proBatch, GitHub master,
# DESCRIPTION version 1.3.1, commit 524249e, April 2025).
# Original authors (DESCRIPTION): Jelena Cuklina, Chloe H. Lee, Patrick Pedrioli (aut);
# Yuliya Burankova (cre). Original license: GPL-3 (DESCRIPTION field; the repository has no
# separate license text file).
# Adapted to Python by PRISM, 2026-10-21.
# What changed: rle_func hard-coded df$Intensity; the run-length check here uses the values being
# fitted. A feature whose fit fails or has fewer than min_measurements consecutive measured
# values is left unchanged and counted (as in proBatch). The optimised-span variant
# (loess_regression_opt) is not ported: it fits on x_all with y of a different length.
"""adjust_batch_trend_dm: per feature, per batch, LOESS of values against run order; the fitted
curve is removed as value + (mean of the fit in that batch - fit)."""

import numpy as np

from ...batch import loess


def max_consecutive_measured(y_in_order):
    """Longest run of non-missing values in run order (proBatch rle_func, on the given values)."""
    best = cur = 0
    for v in y_in_order:
        cur = cur + 1 if np.isfinite(v) else 0
        best = max(best, cur)
    return best


def fit_nonlinear(x, y, span=0.75, degree=2, min_measurements=8):
    """-> (fit at every x, engine) or (None, reason)."""
    order = np.argsort(x, kind="mergesort")
    if max_consecutive_measured(y[order]) < min_measurements:
        return None, f"fewer than {min_measurements} consecutive measured values"
    ok = np.isfinite(y)
    try:
        fit, eng = loess.fit_predict(x[ok], y[ok], x, span=span, degree=degree)
    except Exception as e:
        return None, f"loess failed: {e}"
    return fit, eng


def adjust_batch_trend(X, batch, order, span=0.75, degree=2, min_measurements=8, features=None):
    """X: features x samples (log scale). -> (X corrected, info). NaN stays NaN."""
    X = np.asarray(X, float)
    order = np.asarray(order, float)
    batch = np.asarray([str(b) for b in batch])
    out = X.copy()
    fitted, unchanged, engines, touched = 0, {}, set(), np.zeros(X.shape[0], bool)
    rows = range(X.shape[0]) if features is None else features
    for lv in sorted(set(batch.tolist())):
        idx = np.where(batch == lv)[0]
        for i in rows:
            fit, why = fit_nonlinear(order[idx], X[i, idx], span, degree, min_measurements)
            if fit is None:
                unchanged[why.split(":")[0]] = unchanged.get(why.split(":")[0], 0) + 1
                continue
            engines.add(why)
            out[i, idx] = X[i, idx] + (fit.mean() - fit)
            fitted += 1
            touched[i] = True
    return out, {"span": span, "degree": degree, "min_measurements": min_measurements, "fits": fitted,
                 "left_unchanged": unchanged, "loess_engine": sorted(engines), "touched": touched}
