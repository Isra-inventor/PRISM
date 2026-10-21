"""LOESS for drift correction (v4 section 7.3).

Preferred: scikit-misc (`skmisc.loess`), which wraps the original C/Fortran loess code used by R's
`stats::loess`, with surface = "direct" as in proBatch. Fallback when scikit-misc is missing:
PRISM's own direct local regression (tricube weights over the q = floor(n * span) nearest
points, local polynomial of the given degree, no robustness iterations: R's family "gaussian").
The engine used is reported in every op record.
"""

from __future__ import annotations

import numpy as np


def engine():
    try:
        import skmisc.loess  # noqa: F401
        return "skmisc"
    except Exception:
        return "prism_fallback"


def _fallback(x, y, xnew, span, degree):
    n = len(x)
    q = int(np.floor(n * span))
    q = max(min(q, n), degree + 1)
    out = np.empty(len(xnew))
    for i, x0 in enumerate(xnew):
        d = np.abs(x - x0)
        h = np.sort(d)[q - 1]
        if span > 1:
            h = h * span
        if h <= 0:
            h = 1e-12
        w = np.clip(1 - (d / h) ** 3, 0, None) ** 3
        V = np.vander(x - x0, degree + 1, increasing=True)
        WV = V * w[:, None]
        beta = np.linalg.lstsq(WV.T @ V, WV.T @ y, rcond=None)[0]
        out[i] = beta[0]
    return out


def fit_predict(x, y, xnew, span=0.75, degree=2):
    """Fit y ~ x and predict at xnew. -> (prediction, engine). Raises ValueError when the fit fails."""
    x, y, xnew = (np.asarray(a, float) for a in (x, y, xnew))
    if engine() == "skmisc":
        from skmisc.loess import loess
        lo = loess(x, y, span=span, degree=degree, family="gaussian", surface="direct")
        lo.fit()
        pred = np.asarray(lo.predict(xnew, stderror=False).values, float)
        if not np.isfinite(pred).all():
            raise ValueError("loess prediction is not finite")
        return pred, "skmisc"
    return _fallback(x, y, xnew, span, degree), "prism_fallback"
