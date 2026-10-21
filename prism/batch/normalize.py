"""S1 normalizations (v4 section 6.1). NaN stays NaN in both.

normalize_median: x - median(sample) + median(all observed values).
normalize_quantile: with no NaN, standard quantile normalization (the reference is the mean of the
sorted columns; tied values get the reference interpolated at their average rank). With NaN
(PRISM's NA-aware form): each sample's sorted observed values are interpolated to a common
quantile grid and averaged to build the reference; each observed value is mapped from its
quantile position (rank - 0.5) / n_observed. This differs from preprocessCore on data with NaN.
"""

from __future__ import annotations

import numpy as np

from ..audit.stats import rankdata


def median_center(X):
    with np.errstate(all="ignore"):
        med = np.nanmedian(X, axis=0)
    glob = float(np.nanmedian(X))
    return X - med[None, :] + glob, {"global_median": glob, "sample_medians": med.tolist()}


def _interp_rank(ref_sorted, r):
    """Reference value at (1-based, possibly fractional) rank r."""
    pos = np.asarray(r, float) - 1.0
    return np.interp(pos, np.arange(len(ref_sorted)), ref_sorted)


def quantile(X):
    X = np.asarray(X, float)
    p, n = X.shape
    out = np.full_like(X, np.nan)
    if not np.isnan(X).any():
        ref = np.sort(X, axis=0).mean(1)
        for j in range(n):
            out[:, j] = _interp_rank(ref, rankdata(X[:, j]))
        return out, {"form": "standard (no missing values)"}
    grid = np.linspace(0, 1, max(p, 2))
    cols = []
    for j in range(n):
        v = np.sort(X[np.isfinite(X[:, j]), j])
        if v.size == 0:
            continue
        pos = (np.arange(1, v.size + 1) - 0.5) / v.size
        cols.append(np.interp(grid, pos, v))
    ref = np.mean(cols, axis=0)
    for j in range(n):
        ok = np.isfinite(X[:, j])
        if not ok.any():
            continue
        r = rankdata(X[ok, j])
        out[ok, j] = np.interp((r - 0.5) / ok.sum(), grid, ref)
    return out, {"form": "NA-aware (PRISM): differs from preprocessCore on data with missing values",
                 "grid_points": len(grid)}
