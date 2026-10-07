"""A4. Missingness: rate, pattern, floor values (v3 §6.3).

Declared-vs-observed (item 6) is built from these measures in declared.py.
"""

from __future__ import annotations

import numpy as np

from . import stats
from .finding import Finding, association, table

AUDIT_ID, CODE = "A4", "missingness"


def floor_ties(X, min_obs):
    """Per feature with >= min_obs observed values: m_i = minimum observed, t_i = samples exactly
    at m_i (Step 0 copies values unchanged, so exact equality is meaningful). Floor-tied: t_i >= 2.
    -> (tied bool[p], t int[p], n_obs int[p], S bool[p, n])."""
    ok = np.isfinite(X)
    n_obs = ok.sum(1)
    with np.errstate(invalid="ignore"):
        m = np.where(n_obs > 0, np.nanmin(np.where(ok, X, np.inf), axis=1), np.nan)
    at_min = ok & (X == m[:, None])
    t = at_min.sum(1)
    eligible = n_obs >= min_obs
    tied = eligible & (t >= 2)
    S = at_min & tied[:, None]
    return tied, t, n_obs, S


def run(unit, ctx, params, results):
    f = Finding(AUDIT_ID, CODE, unit).feeds("missingness_handling", "batch_correction", "transformation",
                                            "feature_filtering")
    if unit.out_of_scope:
        return f.status("not_applicable", f"Tier 1 does not audit {unit.out_of_scope} ({unit.scale}).")
    X = unit.X
    n_feat, n = X.shape
    if n == 0 or n_feat == 0:
        return f.status("insufficient_data", "no samples or features in this unit")
    R = ~np.isfinite(X)
    n_cells = X.size
    n_missing = int(R.sum())
    n_zero = int((X == 0).sum())
    feat_rate = R.mean(1)
    samp_rate = R.mean(0)
    M = f.measures
    M["n_features"], M["n_samples"], M["n_cells"] = n_feat, n, n_cells
    M["missing"] = {"n": n_missing, "rate": n_missing / n_cells,
                    "per_feature": stats.summary(feat_rate), "per_sample": stats.summary(samp_rate),
                    "features_with_any": int((feat_rate > 0).sum()), "features_all_missing": int((feat_rate == 1).sum()),
                    "samples_with_any": int((samp_rate > 0).sum())}
    M["zeros"] = {"n": n_zero, "rate": n_zero / n_cells}
    if n_missing:
        f.indicate("missing_values_present", {"n_missing": n_missing, "of": n_cells},
                   pct=round(100 * n_missing / n_cells, 2), n_missing=n_missing, n_cells=n_cells)
    if n_zero:
        f.indicate("zeros_present", {"n_zero": n_zero, "of": n_cells}, pct=round(100 * n_zero / n_cells, 2),
                   n_zero=n_zero, n_cells=n_cells)

    # 2. sentinels
    obs = X[np.isfinite(X)]
    vals, cnt = np.unique(obs, return_counts=True)
    order = np.lexsort((vals, -cnt))[:5]
    M["most_frequent_values"] = [{"value": float(vals[i]), "count": int(cnt[i]), "share": float(cnt[i]) / n_cells}
                                 for i in order]
    for x in M["most_frequent_values"]:
        if x["share"] >= params["sentinel_share"] and x["count"] > 1:
            f.indicate("sentinel_value", {"value": x["value"], "count": x["count"], "of": n_cells},
                       value=x["value"], pct=round(100 * x["share"], 2))

    # 3. floor ties
    tied, t, n_obs, S = floor_ties(X, params["floor_min_observed"])
    n_tied = int(tied.sum())
    ratio = t[tied] / n_obs[tied] if n_tied else np.array([])
    M["floor_ties"] = {"n_features": n_tied, "of": n_feat, "share": n_tied / n_feat,
                       "n_eligible": int((n_obs >= params["floor_min_observed"]).sum()),
                       "min_observed": params["floor_min_observed"],
                       "ties_over_observed": {"p50": float(np.median(ratio)) if n_tied else None,
                                              "p90": float(np.quantile(ratio, 0.9)) if n_tied else None,
                                              "max": float(ratio.max()) if n_tied else None},
                       "n_cells": int(S.sum())}
    strata = {}
    for i, s in enumerate(unit.strata):
        e = strata.setdefault(s, {"stratum": s, "n_features": 0, "n_floor_tied": 0, "n_missing": 0, "n_cells": 0})
        e["n_features"] += 1
        e["n_floor_tied"] += int(tied[i])
        e["n_missing"] += int(R[i].sum())
        e["n_cells"] += n
    for e in strata.values():
        e["floor_tied_share"] = e["n_floor_tied"] / e["n_features"]
        e["missing_rate"] = e["n_missing"] / e["n_cells"]
    M["by_stratum"] = {"source": unit.strata_source, "strata": [strata[k] for k in strata]}
    if n_tied:
        f.indicate("floor_ties_present", {"n_features": n_tied, "of": n_feat},
                   pct=round(100 * n_tied / n_feat, 2), n_features=n_tied, of=n_feat)

    # 4. abundance dependence (on Y over cells neither missing nor at the floor)
    Y = unit.Y
    keep = ~R & ~S
    with np.errstate(invalid="ignore"):
        mean_y = np.where(keep.sum(1) > 0, np.where(keep, Y, 0).sum(1) / np.maximum(keep.sum(1), 1), np.nan)
    floor_rate = S.mean(1)
    dep = {}
    for what, rate, active in (("missing", feat_rate, n_missing > 0), ("floor", floor_rate, n_tied > 0)):
        if not active:
            dep[what] = None
            continue
        rho, nu = stats.spearman(rate, mean_y)
        dep[what] = {"rho": rho, "n_features": nu}
        if rho == rho and rho <= params["abundance_rho_indicator"]:
            f.indicate("abundance_dependent_missingness", {"rho": rho, "n_features": nu, "rate": what},
                       what="missing" if what == "missing" else "floor", rho=round(rho, 3), n=nu)
    M["abundance_dependence"] = dep

    # 5. per-sample rates against B and G
    B, G, skipped = ctx.variables(unit)
    subj = ctx.subject_codes(unit)
    rows = []
    responses = []
    if n_missing:
        responses.append(("missing", samp_rate))
    if n_tied:
        responses.append(("floor", S.mean(0)))
    for what, resp in responses:
        for v in B + G:
            if v.role == "subject":
                continue
            r = association(resp, v, subj, params, f"A4|{unit.dataset_id}|{unit.unit_id}|{what}|{v.name}")
            r["response"] = f"per-sample {what} rate"
            rows.append(r)
    rows, n_tests = table(rows)
    M["associations"] = {"n_tests": n_tests, "rows": rows, "skipped_variables": skipped,
                         "variables": {"B": [v.name for v in B], "G": [v.name for v in G]}}
    for r in rows:
        if r.get("q") is not None and r["q"] <= 0.05:
            f.indicate("missingness_associated", {k: r[k] for k in ("variable", "response", "p", "q", "n_used")},
                       what=r["response"].split()[1], variable=r["variable"], p=round(r["p"], 4), q=round(r["q"], 4))
    batch_tables = []
    for v in B:
        if v.numeric:
            continue
        tbl = []
        for lev in v.levels():
            idx = [i for i, x in enumerate(v.values) if x == lev]
            tbl.append({"level": lev, "n": len(idx), "missing_rate": float(samp_rate[idx].mean()),
                        "floor_rate": float(S[:, idx].mean()) if n_tied else 0.0})
        batch_tables.append({"variable": v.name, "levels": tbl})
    M["per_batch"] = batch_tables

    f.plot.update(feature_missing_rate=stats.histogram(feat_rate, 20), feature_floor_rate=stats.histogram(floor_rate, 20),
                  sample_rates=[{"sample": s, "missing": float(samp_rate[j]), "floor": float(S[:, j].mean())}
                                for j, s in enumerate(unit.samples)])
    f.method("missingness, sentinels, floor ties, abundance dependence, restricted-permutation associations",
             n_used=n, transform=unit.transform, floor_min_observed=params["floor_min_observed"],
             sentinel_share=params["sentinel_share"], permutations=params["permutations"], seed=params["seed"])
    results.setdefault(unit.dataset_id, {}).setdefault(unit.unit_id, {})["floor"] = {"tied": tied, "S": S}
    return f
