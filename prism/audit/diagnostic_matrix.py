"""NaN and floor-tied values in diagnostics (Build Request v4, section 3).

PCA, PERMANOVA, clustering, correlations, PVCA and the outlier distance need a complete matrix
or a stated policy. **No imputation, no zero filling.** Every finding that uses a diagnostic
matrix reports the policy, the number and share of features used and the bias warning.

| policy             | rule |
|--------------------|------|
| all_observed       | only when Y has no NaN |
| complete_features  | q = 100: features observed in every sample (every sample kept). q < 100: samples whose
|                    | missing rate is above `diag_sample_missing_max` are dropped, then features observed in at
|                    | least q% of all samples and with no NaN in the kept samples are used |
| exclude_floor      | complete_features(q) without the floor-tied features (A4 item 3) |
| floor_censored     | floor-tied cells set to NaN in this diagnostic copy only, then complete_features(q) |
| pairwise           | correlations only: Pearson on pairwise-complete observations, shared features per pair |

The default for PCA, PERMANOVA, clustering, PVCA and outlier distance is complete_features(100);
if it keeps fewer than `diag_min_features` features or `diag_min_share` of the features
(heuristic), q = 75 and then 50 are tried and the q used is reported.
"""

from __future__ import annotations

import numpy as np

BIAS_WARNING = "the retained features are biased toward higher-abundance, more consistently detected features"
POLICIES = ("all_observed", "complete_features", "exclude_floor", "floor_censored", "pairwise")
Q_LADDER = (100, 75, 50)


class Diag:
    """A diagnostic matrix: Y restricted to rows (features) and cols (samples)."""

    def __init__(self, Y, rows, cols, policy, q, p_total, n_total, fallback=None, below_minimum=False):
        self.Y, self.rows, self.cols, self.policy, self.q = Y, rows, cols, policy, q
        self.p_total, self.n_total = p_total, n_total
        self.fallback = fallback or []
        self.below_minimum = below_minimum

    @property
    def n_features(self):
        return int(len(self.rows))

    @property
    def samples_dropped(self):
        return self.n_total - len(self.cols)

    def report(self, sample_ids=None):
        out = {"policy": self.policy, "q": self.q, "n_features_used": self.n_features, "of": int(self.p_total),
               "share": (self.n_features / self.p_total) if self.p_total else None,
               "n_samples_used": int(len(self.cols)), "n_samples_dropped": int(self.samples_dropped),
               "bias_warning": BIAS_WARNING}
        if sample_ids is not None and self.samples_dropped:
            keep = set(self.cols.tolist())
            out["samples_dropped"] = [s for j, s in enumerate(sample_ids) if j not in keep]
        if self.fallback:
            out["fallback"] = self.fallback
        if self.below_minimum:
            out["below_minimum"] = True
        return out


def floor_cells(X, min_obs):
    """(floor-tied features, floor-tied cells) from the raw matrix X (A4 item 3)."""
    from .a04_missingness import floor_ties
    tied, _, _, S = floor_ties(X, min_obs)
    return tied, S


def diagnostic_matrix(Y, policy="complete_features", q=100, floor_features=None, floor_cell_mask=None,
                      sample_missing_max=0.5):
    """-> Diag. Never fills a value: rows or samples are dropped, or (floor_censored) floor cells
    become NaN in this copy only."""
    if policy not in POLICIES or policy == "pairwise":
        raise ValueError(f"diagnostic_matrix: policy '{policy}' (pairwise is for correlations: use pairwise_corr).")
    p, n = Y.shape
    Z = Y
    if policy == "all_observed":
        if np.isnan(Y).any():
            raise ValueError("all_observed is used only when Y has no NaN.")
        return Diag(Y, np.arange(p), np.arange(n), policy, 100, p, n)
    if policy == "floor_censored":
        if floor_cell_mask is None:
            raise ValueError("floor_censored needs the floor-tied cells.")
        Z = np.where(floor_cell_mask, np.nan, Y)
    ok = np.isfinite(Z)
    cols = np.arange(n)
    if q < 100:
        cols = np.where(1 - ok.mean(0) <= sample_missing_max)[0]
    observed_share = ok.mean(1) * 100
    keep = (observed_share >= q - 1e-9) & ok[:, cols].all(1)
    if policy == "exclude_floor":
        if floor_features is None:
            raise ValueError("exclude_floor needs the floor-tied features.")
        keep &= ~np.asarray(floor_features, bool)
    rows = np.where(keep)[0]
    return Diag(Z[np.ix_(rows, cols)], rows, cols, policy, q, p, n)


def enough(d, min_features, min_share):
    return d.n_features >= min_features and d.n_features >= min_share * d.p_total


def default_matrix(Y, params, policy="complete_features", **kw):
    """complete_features(100), falling back to q = 75 then 50 when too few features remain."""
    tried = []
    best = None
    for q in Q_LADDER:
        d = diagnostic_matrix(Y, policy, q, sample_missing_max=params.get("diag_sample_missing_max", 0.5), **kw)
        tried.append({"q": q, "n_features": d.n_features, "n_samples": int(len(d.cols))})
        if best is None or d.n_features > best.n_features:
            best = d
        if enough(d, params.get("diag_min_features", 20), params.get("diag_min_share", 0.10)):
            d.fallback = tried if q != 100 else []
            return d
    best.fallback, best.below_minimum = tried, True
    return best


def pairwise_corr(Y):
    """Pearson r per sample pair on pairwise-complete features; -> (R, N shared)."""
    ok = np.isfinite(Y)
    n = Y.shape[1]
    if ok.all() and Y.shape[0] >= 3:
        with np.errstate(invalid="ignore", divide="ignore"):
            R = np.corrcoef(Y.T)
        return np.where(np.isfinite(R), R, np.nan), np.full((n, n), Y.shape[0], dtype=int)
    R = np.full((n, n), np.nan)
    N = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i, n):
            m = ok[:, i] & ok[:, j]
            N[i, j] = N[j, i] = int(m.sum())
            if m.sum() >= 3:
                a, b = Y[m, i], Y[m, j]
                sa, sb = a.std(), b.std()
                if sa > 0 and sb > 0:
                    R[i, j] = R[j, i] = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
    return R, N


def pairwise_report(N):
    iu = np.triu_indices(N.shape[0], 1)
    v = N[iu]
    return {"policy": "pairwise", "shared_features_per_pair": {"min": int(v.min()) if v.size else 0,
                                                               "median": float(np.median(v)) if v.size else 0,
                                                               "max": int(v.max()) if v.size else 0},
            "bias_warning": BIAS_WARNING}


def variants(Y, X, params):
    """The sensitivity variants: base (with fallback), exclude_floor, floor_censored and the q ladder."""
    tied, S = floor_cells(X, params["floor_min_observed"])
    base = default_matrix(Y, params)
    out = [("base", base)]
    smm = params.get("diag_sample_missing_max", 0.5)
    if tied.any():
        out.append(("exclude_floor", diagnostic_matrix(Y, "exclude_floor", base.q, floor_features=tied,
                                                       sample_missing_max=smm)))
        out.append(("floor_censored", diagnostic_matrix(Y, "floor_censored", base.q, floor_cell_mask=S,
                                                        sample_missing_max=smm)))
    for q in Q_LADDER:
        if q != base.q:
            out.append((f"q{q}", diagnostic_matrix(Y, "complete_features", q, sample_missing_max=smm)))
    return out, int(tied.sum())


def sensitivity_block(results, describe):
    """results: [(variant, Diag, conclusion)] where conclusion is hashable or None (not computable).
    -> the block; `depends` is True when two computable conclusions differ. No winner is picked."""
    rows = []
    for name, d, concl in results:
        rows.append({"variant": name, "policy": d.policy, "q": d.q, "n_features": d.n_features,
                     "n_samples": int(len(d.cols)), "conclusion": describe(concl) if concl is not None else None,
                     "computable": concl is not None})
    concls = {c for _, _, c in results if c is not None}
    return {"variants": rows, "same_conclusion": len(concls) <= 1, "depends": len(concls) > 1}
