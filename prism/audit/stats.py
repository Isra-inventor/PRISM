"""Statistics for the Tier 1 audit, in numpy only (v3 §6.1).

Ranks with ties averaged, Spearman, Kruskal-Wallis H, BH q-values, quantiles, modified z, and
restricted permutation tests (whole subjects / within subject / free) with exhaustive
enumeration when few distinct permutations exist.
"""

from __future__ import annotations

import math
import zlib
from itertools import product

import numpy as np


def rankdata(x):
    """Average ranks (1-based) of a 1-D array without NaN."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x))
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return ranks


def rank_rows(A):
    """Average ranks along each row of a 2-D array without NaN."""
    return np.vstack([rankdata(r) for r in A]) if len(A) else np.zeros_like(A)


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / d if d > 0 else float("nan")


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan"), int(ok.sum())
    return pearson(rankdata(a[ok]), rankdata(b[ok])), int(ok.sum())


def kruskal_h(ranks, codes, k):
    """H from precomputed ranks of the response and integer group codes 0..k-1 (tie-corrected)."""
    n = len(ranks)
    sums = np.bincount(codes, weights=ranks, minlength=k)
    cnt = np.bincount(codes, minlength=k)
    h = 12.0 / (n * (n + 1)) * float(np.sum(sums ** 2 / np.maximum(cnt, 1))) - 3 * (n + 1)
    _, t = np.unique(ranks, return_counts=True)
    c = 1 - float(np.sum(t ** 3 - t)) / (n ** 3 - n) if n > 1 else 1
    return h / c if c > 0 else 0.0


def kruskal_h_rows(ranks, P, k):
    """kruskal_h for every row of a code matrix P (the response ranks are fixed)."""
    n = len(ranks)
    _, t = np.unique(ranks, return_counts=True)
    c = 1 - float(np.sum(t ** 3 - t)) / (n ** 3 - n) if n > 1 else 1
    total = np.zeros(P.shape[0])
    for j in range(k):
        m = P == j
        cnt = m.sum(1)
        sums = (m * ranks).sum(1)
        total += np.where(cnt > 0, sums ** 2 / np.maximum(cnt, 1), 0)
    h = 12.0 / (n * (n + 1)) * total - 3 * (n + 1)
    return h / c if c > 0 else np.zeros_like(h)


def bh(pvals):
    """Benjamini-Hochberg q-values; None stays None."""
    idx = [i for i, p in enumerate(pvals) if p is not None and p == p]
    out = [None] * len(pvals)
    if not idx:
        return out
    p = np.array([pvals[i] for i in idx])
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    q = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    for j, o in enumerate(order):
        out[idx[o]] = float(q[j])
    return out


def quantiles(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {f"p{int(q * 100)}": None for q in qs}
    return {f"p{int(q * 100)}": float(np.quantile(x, q)) for q in qs}


def summary(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    return {"n": int(len(x)), "min": float(x.min()), "p25": float(np.quantile(x, 0.25)),
            "median": float(np.median(x)), "p75": float(np.quantile(x, 0.75)), "max": float(x.max()),
            "mean": float(x.mean())}


def histogram(x, bins=30, log10=False):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if log10:
        x = np.log10(x[x > 0])
    if not len(x):
        return {"edges": [], "counts": [], "log10": log10}
    c, e = np.histogram(x, bins=bins)
    return {"edges": [round(float(v), 6) for v in e], "counts": [int(v) for v in c], "log10": log10}


def modified_z(x):
    """0.6745 (x - median) / MAD; when MAD = 0, (x - median) / (1.253314 * mean absolute deviation)."""
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad > 0:
        return 0.6745 * (x - med) / mad
    mean_ad = np.nanmean(np.abs(x - med))
    if mean_ad > 0:
        return (x - med) / (1.253314 * mean_ad)
    return np.zeros_like(x)


# ------------------------------------------------------------------ permutations


def _multinomial(counts):
    n = sum(counts)
    out = math.factorial(n)
    for c in counts:
        out //= math.factorial(c)
    return out


def _multiset_perms(items):
    """Every distinct ordering of a multiset (items: list of hashable), in a fixed order."""
    vals = sorted(set(items), key=repr)
    cnt = {v: items.count(v) for v in vals}
    n = len(items)
    cur = []

    def rec():
        if len(cur) == n:
            yield list(cur)
            return
        for v in vals:
            if cnt[v]:
                cnt[v] -= 1
                cur.append(v)
                yield from rec()
                cur.pop()
                cnt[v] += 1
    return rec()


class Permuter:
    """Permutations of a sample-level variable that respect the design (v3 §6.1):
    constant within subject -> permute whole subjects; varies within subject -> permute within
    subject; no subject -> permute samples freely."""

    def __init__(self, codes, subject=None, n_perm=999, exhaustive_max=10000, seed=0, key=""):
        self.codes = np.asarray(codes)
        n = len(self.codes)
        self.seed = [int(seed), zlib.crc32(key.encode("utf-8"))]
        if subject is None:
            self.mode = "free"
            self.blocks = [np.arange(n)]
        else:
            subject = np.asarray(subject)
            groups = {}
            for i, s in enumerate(subject):
                groups.setdefault(s, []).append(i)
            self.blocks = [np.array(groups[s]) for s in sorted(groups, key=repr)]
            const = all(len(set(self.codes[b].tolist())) == 1 for b in self.blocks)
            self.mode = "subjects" if const else "within"
        self.n_distinct = self._count()
        self.exhaustive = self.n_distinct <= exhaustive_max
        self.n_perm = n_perm

    def _count(self):
        if self.mode == "free":
            return _multinomial(list(np.unique(self.codes, return_counts=True)[1]))
        if self.mode == "subjects":
            vals = [self.codes[b[0]] for b in self.blocks]
            return _multinomial(list(np.unique(vals, return_counts=True)[1]))
        out = 1
        for b in self.blocks:
            out *= _multinomial(list(np.unique(self.codes[b], return_counts=True)[1]))
            if out > 10 ** 12:
                return out
        return out

    def matrix(self):
        """-> (P x n) permuted codes; row 0 is the observed arrangement when exhaustive."""
        n = len(self.codes)
        rows = []
        if self.exhaustive:
            if self.mode == "free":
                for p in _multiset_perms(self.codes.tolist()):
                    rows.append(p)
            elif self.mode == "subjects":
                for labels in _multiset_perms([self.codes[b[0]] for b in self.blocks]):
                    row = np.empty(n, dtype=self.codes.dtype)
                    for b, lab in zip(self.blocks, labels):
                        row[b] = lab
                    rows.append(row)
            else:
                per = [[(b, p) for p in _multiset_perms(self.codes[b].tolist())] for b in self.blocks]
                for combo in product(*per):
                    row = np.empty(n, dtype=self.codes.dtype)
                    for b, labels in combo:
                        row[b] = labels
                    rows.append(row)
            return np.array(rows)
        rng = np.random.default_rng(self.seed)
        out = np.empty((self.n_perm, n), dtype=self.codes.dtype)
        for k in range(self.n_perm):
            row = self.codes.copy()
            if self.mode == "free":
                row = row[rng.permutation(n)]
            elif self.mode == "subjects":
                labs = [self.codes[b[0]] for b in self.blocks]
                for b, j in zip(self.blocks, rng.permutation(len(self.blocks))):
                    row[b] = labs[j]
            else:
                for b in self.blocks:
                    row[b] = self.codes[b][rng.permutation(len(b))]
            out[k] = row
        return out

    def p_value(self, stats_perm, observed):
        """stats_perm: statistic for each row of matrix(). Larger is more extreme."""
        ge = int(np.sum(stats_perm >= observed - 1e-12 * max(1.0, abs(observed))))
        if self.exhaustive:
            total = len(stats_perm)
            return ge / total, 1.0 / total
        return (1 + ge) / (1 + len(stats_perm)), 1.0 / (1 + len(stats_perm))

    def describe(self):
        return {"scheme": {"free": "samples permuted freely", "subjects": "whole subjects permuted",
                           "within": "permuted within subject"}[self.mode],
                "exhaustive": bool(self.exhaustive), "n_distinct_permutations": int(min(self.n_distinct, 10 ** 12)),
                "n_permutations": int(self.n_distinct if self.exhaustive else self.n_perm)}


# ------------------------------------------------------------------ multivariate


def complete_features(Y):
    """Rows of Y observed in every sample (PCA and distances need complete data)."""
    return Y[np.isfinite(Y).all(1)]


def pca(Y, k):
    """PCA of feature-centered Y (features x samples) by SVD. -> scores (n x k), variance shares."""
    Yc = Y - Y.mean(1, keepdims=True)
    U, s, Vt = np.linalg.svd(Yc.T, full_matrices=False)     # samples x features
    k = max(0, min(k, len(s)))
    scores = U[:, :k] * s[:k]
    total = float((s ** 2).sum())
    # sign convention: the largest absolute loading-free score of each PC is positive (deterministic)
    for j in range(k):
        i = int(np.argmax(np.abs(scores[:, j])))
        if scores[i, j] < 0:
            scores[:, j] *= -1
    return scores, (s[:k] ** 2 / total if total > 0 else np.zeros(k))


def eta_sq_rows(y, P, k):
    """Eta squared (between-group share of variance) of y for every row of a code matrix P."""
    y = np.asarray(y, float)
    sst = float(((y - y.mean()) ** 2).sum())
    if sst <= 0:
        return np.zeros(P.shape[0])
    total = np.zeros(P.shape[0])
    for j in range(k):
        m = P == j
        cnt = m.sum(1)
        sums = (m * y).sum(1)
        total += np.where(cnt > 0, sums ** 2 / np.maximum(cnt, 1), 0)
    ssb = total - y.sum() ** 2 / len(y)
    return ssb / sst


def gram(Y):
    """Gram matrix of the feature-centered complete Y (samples x samples): Euclidean PERMANOVA
    needs only this."""
    Yc = Y - Y.mean(1, keepdims=True)
    return Yc.T @ Yc


def permanova_rows(K, P, k):
    """One-factor PERMANOVA on Euclidean distances, from the Gram matrix K, for every row of a
    code matrix P. -> (R2, pseudo-F) arrays. SS_A = sum_g 1'K_g1 / n_g (K is centered)."""
    n = K.shape[0]
    sst = float(np.trace(K))
    ssa = np.zeros(P.shape[0])
    for j in range(k):
        m = (P == j).astype(float)               # perms x n
        cnt = m.sum(1)
        quad = np.einsum("pi,ij,pj->p", m, K, m)
        ssa += np.where(cnt > 0, quad / np.maximum(cnt, 1), 0)
    ssw = sst - ssa
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (ssa / (k - 1)) / (ssw / (n - k))
    return ssa / sst if sst > 0 else np.zeros_like(ssa), F


def permanova_numeric_rows(K, Xp):
    """PERMANOVA with one numeric predictor per row of Xp (perms x n): regression SS / total."""
    n = K.shape[0]
    sst = float(np.trace(K))
    Xc = Xp - Xp.mean(1, keepdims=True)
    den = (Xc ** 2).sum(1)
    ssa = np.einsum("pi,ij,pj->p", Xc, K, Xc) / np.where(den > 0, den, 1)
    ssw = sst - ssa
    with np.errstate(divide="ignore", invalid="ignore"):
        F = ssa / (ssw / (n - 2))
    return ssa / sst if sst > 0 else np.zeros_like(ssa), F


def row_modified_z(A):
    """Modified z within each row (NaN ignored), with the mean-absolute-deviation fallback."""
    med = np.nanmedian(A, axis=1, keepdims=True)
    dev = np.abs(A - med)
    mad = np.nanmedian(dev, axis=1, keepdims=True)
    mean_ad = np.nanmean(dev, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(mad > 0, 0.6745 * (A - med) / np.where(mad > 0, mad, 1),
                     np.where(mean_ad > 0, (A - med) / (1.253314 * np.where(mean_ad > 0, mean_ad, 1)), 0.0))
    return np.where(np.isfinite(A), z, np.nan)


def ols_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc = x - x.mean()
    d = float(xc @ xc)
    return float(xc @ (y - y.mean())) / d if d > 0 else float("nan")
