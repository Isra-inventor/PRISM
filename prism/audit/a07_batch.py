"""A7. Batch structure (v3 §6.3): batch candidates against design variables, PCA associations,
marginal PERMANOVA. Multi-factor models are out of scope."""

from __future__ import annotations

import numpy as np

from . import stats
from .finding import Finding, association, permanova, table

AUDIT_ID, CODE = "A7", "batch"


def _codes(values):
    levels = sorted({v for v in values if v}, key=str)
    return levels, [levels.index(v) if v else -1 for v in values]


def structure(b, g):
    """Contingency of two categorical variables over samples where both are present."""
    pairs = [(x, y) for x, y in zip(b.values, g.values) if x and y]
    lb, lg = sorted({x for x, _ in pairs}, key=str), sorted({y for _, y in pairs}, key=str)
    T = np.zeros((len(lb), len(lg)), dtype=int)
    for x, y in pairs:
        T[lb.index(x), lg.index(y)] += 1
    n = T.sum()
    out = {"batch": b.name, "design": g.name, "n_used": int(n), "batch_levels": lb, "design_levels": lg,
           "table": T.tolist()}
    if len(lb) < 2 or len(lg) < 2:
        return dict(out, code=None, reason="fewer than 2 levels")
    exp = T.sum(1, keepdims=True) * T.sum(0, keepdims=True) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.nansum((T - exp) ** 2 / np.where(exp > 0, exp, np.nan)))
    out["cramers_v"] = float(np.sqrt(chi2 / (n * (min(T.shape) - 1))))
    g_in_b = all((T[:, j] > 0).sum() == 1 for j in range(T.shape[1]))     # each design level in one batch
    b_in_g = all((T[i] > 0).sum() == 1 for i in range(T.shape[0]))
    D = [np.ones(n)]
    xb = [lb.index(x) for x, _ in pairs]
    xg = [lg.index(y) for _, y in pairs]
    D += [np.array([1.0 if v == k else 0.0 for v in xb]) for k in range(1, len(lb))]
    D += [np.array([1.0 if v == k else 0.0 for v in xg]) for k in range(1, len(lg))]
    rank = int(np.linalg.matrix_rank(np.vstack(D).T))
    full = 1 + (len(lb) - 1) + (len(lg) - 1)
    out["design_rank"], out["full_rank"] = rank, full
    if g_in_b and b_in_g:
        code = "confounded"
    elif g_in_b:
        code = "nested"
        out["nesting"] = f"'{g.name}' is constant within '{b.name}'"
    elif b_in_g:
        code = "nested"
        out["nesting"] = f"'{b.name}' is constant within '{g.name}'"
    elif rank < full:
        code = "confounded"
    elif (T > 0).all() and len(set(T.ravel().tolist())) == 1:
        code = "crossed_balanced"
    else:
        code = "partially_confounded"
    out["code"] = code
    return out


def run(unit, ctx, params, results):
    f = Finding(AUDIT_ID, CODE, unit).feeds("batch_correction", "missingness_handling")
    if unit.out_of_scope:
        return f.status("not_applicable", f"Tier 1 does not audit {unit.out_of_scope} ({unit.scale}).")
    B, G, skipped = ctx.variables(unit)
    M = f.measures
    M["variables"] = {"B": [v.describe() for v in B], "G": [v.describe() for v in G], "skipped": skipped}
    if not B:
        f.status("insufficient_metadata", "no batch candidate: no sample-metadata column has audit_kind batch or "
                                          "run_order, and no batch_variable override", needs=["batch variable"])
    # 1. structure of each B against each G
    struct = []
    for b in B:
        for g in G:
            if b.numeric or g.numeric:
                continue
            s = structure(b, g)
            struct.append(s)
            if s.get("code") == "confounded":
                f.indicate("batch_confounded", {k: s[k] for k in ("batch", "design", "cramers_v")}, batch=b.name,
                           design=g.name)
            elif s.get("code") == "nested":
                f.indicate("batch_nested", {k: s[k] for k in ("batch", "design", "nesting")}, text=s["nesting"])
    M["structure"] = struct

    # 2. PCA of centered Y on complete features
    Y = stats.complete_features(unit.Y)
    n = unit.n
    subj = ctx.subject_codes(unit)
    variables = [v for v in B + G if v.role != "subject"] + [v for v in G if v.role == "subject"]
    if Y.shape[0] < 2 or n < 3:
        M["pca"] = {"status": "insufficient_data", "reason": "fewer than 2 complete features or 3 samples"}
        f.method("structure codes", n_used=n)
        return f
    k = min(params["max_pcs"], n - 1)
    scores, var_exp = stats.pca(Y, k)
    rows = []
    for j in range(k):
        for v in variables:
            if v.role == "subject":
                r = association(scores[:, j], v, None, params, f"A7|{unit.dataset_id}|{unit.unit_id}|PC{j + 1}|{v.name}",
                                guard_subjects=False, categorical="eta_squared")
            else:
                r = association(scores[:, j], v, subj, params, f"A7|{unit.dataset_id}|{unit.unit_id}|PC{j + 1}|{v.name}",
                                categorical="eta_squared")
            r["pc"] = j + 1
            rows.append(r)
    # 3. PERMANOVA (marginal), Euclidean distances on Y
    K = stats.gram(Y)
    for v in variables:
        r = permanova(K, v, None if v.role == "subject" else subj, params, f"A7|{unit.dataset_id}|{unit.unit_id}|PERMANOVA|{v.name}")
        r["pc"] = None
        rows.append(r)
    rows, n_tests = table(rows)
    M["pca"] = {"n_features_complete": int(Y.shape[0]), "of": unit.p, "k": k,
                "variance_explained": [float(x) for x in var_exp]}
    M["associations"] = {"n_tests": n_tests, "rows": rows}
    for r in rows:
        if r.get("q") is None or r["q"] > 0.05:
            continue
        if r["test"] == "permanova":
            f.indicate("permanova_association", {k_: r.get(k_) for k_ in ("variable", "r2", "value", "p", "q", "n_used")},
                       variable=r["variable"], r2=round(100 * r["r2"], 1), F=round(r["value"], 2), q=round(r["q"], 4))
        else:
            f.indicate("pc_association", {k_: r.get(k_) for k_ in ("variable", "pc", "statistic", "value", "p", "q")},
                       pc=r["pc"], pct=round(100 * var_exp[r["pc"] - 1], 1), variable=r["variable"],
                       stat="eta²" if r["statistic"] == "eta_squared" else "rho", value=round(r["value"], 3),
                       q=round(r["q"], 4))
    f.plot["pca"] = {"variance_explained": [float(x) for x in var_exp],
                     "samples": [{"sample": s, "scores": [float(x) for x in scores[i]],
                                  "values": {v.name: v.values[i] for v in B + G}} for i, s in enumerate(unit.samples)]}
    f.plot["crosstabs"] = [{k_: s[k_] for k_ in ("batch", "design", "batch_levels", "design_levels", "table")}
                           for s in struct]
    f.method("contingency structure, PCA (SVD of feature-centered Y, complete features), eta squared / Spearman "
             "and marginal PERMANOVA with restricted permutations", n_used=n, transform=unit.transform,
             k=k, permutations=params["permutations"], seed=params["seed"])
    return f
