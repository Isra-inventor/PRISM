# Derived from proBatch (Bioconductor; https://github.com/symbioticMe/proBatch, GitHub master,
# DESCRIPTION version 1.3.1, commit 524249e, April 2025).
# Original authors (DESCRIPTION): Jelena Cuklina, Chloe H. Lee, Patrick Pedrioli (aut);
# Yuliya Burankova (cre). Original license: GPL-3 (DESCRIPTION field; the repository has no
# separate license text file).
# Adapted to Python by PRISM, 2026-10-21.
# What changed: matrix interface; NaN stays NaN (medians and means over observed values);
# correct_batch_effects_dm hard-coded keep_all = FALSE (an R switch on FALSE returns NULL, so the
# function cannot return a matrix): the intended behaviour, a corrected matrix, is ported. The
# median-centering code path that returned the string measure_col for flagged values is not
# reproduced: flagged values are excluded from the medians and shifted like the others.
"""center_feature_batch_medians_dm / center_feature_batch_means_dm, correct_with_ComBat_dm (~1 model),
correct_batch_effects_dm (continuous then discrete)."""

import numpy as np

from ...batch import combat as combat_mod
from .drift import adjust_batch_trend


def center_feature_batch(X, batch, stat="median", exclude=None):
    """x - stat(feature, batch) + stat(feature, all). exclude: cells not used to compute the
    statistics (proBatch's flagged / imputed values); they are still shifted. NaN stays NaN."""
    X = np.asarray(X, float)
    batch = np.asarray([str(b) for b in batch])
    f = np.nanmedian if stat == "median" else np.nanmean
    Z = X if exclude is None else np.where(exclude, np.nan, X)
    out = X.copy()
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            glob = f(Z, axis=1)
            shifts = {}
            for lv in sorted(set(batch.tolist())):
                idx = np.where(batch == lv)[0]
                sb = f(Z[:, idx], axis=1)
                d = np.where(np.isfinite(sb) & np.isfinite(glob), glob - sb, 0.0)
                out[:, idx] = X[:, idx] + d[:, None]
                shifts[lv] = int(np.isfinite(sb).sum())
    return out, {"stat": stat, "features_with_batch_stat": shifts}


def combat_intercept(X, batch):
    return combat_mod.combat(X, batch)


def correct_batch_effects(X, batch, order=None, continuous=None, discrete="MedianCentering", **kw):
    """continuous = 'loess_regression' or None; discrete = MedianCentering | MeanCentering | ComBat."""
    info = {}
    if continuous is not None:
        if continuous != "loess_regression":
            raise ValueError("Only loess_regression is available (as in proBatch).")
        X, info["trend"] = adjust_batch_trend(X, batch, order, **kw)
        info["trend"].pop("touched", None)
    if discrete == "MedianCentering":
        X, info["discrete"] = center_feature_batch(X, batch, "median")
    elif discrete == "MeanCentering":
        X, info["discrete"] = center_feature_batch(X, batch, "mean")
    elif discrete == "ComBat":
        X, info["discrete"] = combat_intercept(X, batch)
    else:
        raise ValueError(f"Unknown discrete correction '{discrete}'.")
    return X, info
