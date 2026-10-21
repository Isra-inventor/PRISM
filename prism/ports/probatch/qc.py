# Derived from proBatch (Bioconductor; https://github.com/symbioticMe/proBatch, GitHub master,
# DESCRIPTION version 1.3.1, commit 524249e, April 2025).
# Original authors (DESCRIPTION): Jelena Cuklina, Chloe H. Lee, Patrick Pedrioli (aut);
# Yuliya Burankova (cre). Original license: GPL-3 (DESCRIPTION field; the repository has no
# separate license text file).
# Adapted to Python by PRISM, 2026-10-21.
# What changed: the CV filter in calculate_feature_CV was written as any(df_long$n) > 2, which
# never filters; the intended filter (features with more than 2 measured values) is ported.
# Correlations use pairwise-complete observations (proBatch fills or drops differently); the
# number of shared features is reported.
"""QC measures: sample correlation by replicate / batch, peptide-within-protein correlation,
per-feature CV (on the unlogged scale)."""

import numpy as np

from ...audit.diagnostic_matrix import pairwise_corr
from .transform import unlog


def feature_cv(X, batch=None, log_base=2, offset=None, min_n=3, logged=True):
    """CV (%) per feature overall and per batch on the linear scale; features with fewer than
    min_n measured values (proBatch's intended 'n > 2') get NaN."""
    X = np.asarray(X, float)
    L = unlog(X, log_base, offset) if logged else X
    def cv(A):
        n = np.isfinite(A).sum(1)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                c = 100 * np.nanstd(A, axis=1, ddof=1) / np.nanmean(A, axis=1)
        return np.where(n >= min_n, c, np.nan)
    out = {"total": cv(L)}
    if batch is not None:
        b = np.asarray([str(x) for x in batch])
        out["per_batch"] = {lv: cv(L[:, b == lv]) for lv in sorted(set(b.tolist()))}
    return out


def sample_correlation_classes(Y, replicate=None, batch=None):
    """Median Pearson r of sample pairs by class (same / different replicate, same / different batch)."""
    R, N = pairwise_corr(np.asarray(Y, float))
    n = R.shape[0]
    iu = np.triu_indices(n, 1)
    r = R[iu]
    ok = np.isfinite(r)
    out = {"n_pairs": int(ok.sum())}
    for name, lab in (("replicate", replicate), ("batch", batch)):
        if lab is None:
            continue
        same = np.array([bool(lab[i]) and lab[i] == lab[j] for i, j in zip(*iu)])
        out[name] = {"same_median_r": float(np.median(r[ok & same])) if (ok & same).any() else None,
                     "different_median_r": float(np.median(r[ok & ~same])) if (ok & ~same).any() else None}
    return out


def peptide_correlation(Y, protein_of):
    """Median correlation of peptides within the same protein against peptides of different proteins.
    protein_of: one protein id per feature (row); None when no map exists (not_applicable)."""
    if protein_of is None:
        return {"status": "not_applicable", "reason": "no peptide-to-protein map"}
    Y = np.asarray(Y, float)
    R, N = pairwise_corr(Y.T)
    p = R.shape[0]
    iu = np.triu_indices(p, 1)
    r = R[iu]
    ok = np.isfinite(r)
    same = np.array([protein_of[i] == protein_of[j] for i, j in zip(*iu)])
    return {"status": "computed", "within_protein_median_r": float(np.median(r[ok & same])) if (ok & same).any() else None,
            "between_protein_median_r": float(np.median(r[ok & ~same])) if (ok & ~same).any() else None,
            "n_within_pairs": int((ok & same).sum())}
