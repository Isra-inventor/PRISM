"""A11. Outlier flags (v3 §6.3). Flags only: nothing is removed."""

from __future__ import annotations

import numpy as np

from . import stats
from .finding import Finding

AUDIT_ID, CODE = "A11", "outliers"
TOP_CELLS = 50


def run(unit, ctx, params, results):
    f = Finding(AUDIT_ID, CODE, unit).feeds("feature_filtering", "sample_exclusion")
    if unit.out_of_scope:
        return f.status("not_applicable", f"Tier 1 does not audit {unit.out_of_scope} ({unit.scale}).")
    cut = params["modified_z_cutoff"]
    X, Y = unit.X, unit.Y
    n, samples = unit.n, unit.samples
    M = f.measures
    if n < 3:
        return f.status("insufficient_data", "fewer than 3 samples")
    # 1. samples: distance to the median profile (complete features), PC1/PC2 scores, r with the median
    Yc = stats.complete_features(Y)
    flagged = set()
    sample_rows = [{"sample": s} for s in samples]
    if Yc.shape[0] >= 2:
        med = np.median(Yc, axis=1)
        d = np.sqrt(((Yc - med[:, None]) ** 2).sum(0))
        zd = stats.modified_z(d)
        r = [stats.pearson(Yc[:, j], med) for j in range(n)]
        k = min(2, n - 1)
        scores, ve = stats.pca(Yc, k)
        zpc = [stats.modified_z(scores[:, j]) for j in range(k)]
        for j, row in enumerate(sample_rows):
            row.update(distance=float(d[j]), z_distance=float(zd[j]), r_median=float(r[j]),
                       **{f"z_pc{i + 1}": float(zpc[i][j]) for i in range(k)})
            reasons = []
            if zd[j] > cut:
                reasons.append("distance")
            reasons += [f"pc{i + 1}" for i in range(k) if abs(zpc[i][j]) > cut]
            row["flags"] = reasons
            if reasons:
                flagged.add(samples[j])
        M["samples"] = {"n_features_complete": int(Yc.shape[0]), "cutoff": cut,
                        "flagged": [r_["sample"] for r_ in sample_rows if r_["flags"]], "per_sample": sample_rows}
        for r_ in sample_rows:
            if r_["flags"]:
                f.indicate("outlier_sample", {k_: r_.get(k_) for k_ in ("sample", "z_distance", "flags")},
                           sample=r_["sample"], by=", ".join(r_["flags"]), z=round(max(
                               [abs(r_.get("z_distance", 0))] + [abs(r_.get(f"z_pc{i + 1}", 0)) for i in range(k)]), 2),
                           cut=cut)
    else:
        M["samples"] = {"status": "insufficient_data", "reason": "fewer than 2 complete features"}

    # 2. cells: modified z within each feature on Y
    Z = stats.row_modified_z(Y)
    big = np.abs(np.nan_to_num(Z)) > cut
    per_feature = big.sum(1)
    per_sample = big.sum(0)
    M["cells"] = {"cutoff": cut, "n_flagged": int(big.sum()), "n_features_with_any": int((per_feature > 0).sum()),
                  "per_sample": {s: int(per_sample[j]) for j, s in enumerate(samples)},
                  "per_feature": stats.summary(per_feature)}
    ii, jj = np.where(big)
    order = np.lexsort((jj, ii, -np.abs(Z[ii, jj])))[:TOP_CELLS]
    M["cells"]["top"] = [{"feature": unit.feature_keys[ii[o]], "sample": samples[jj[o]], "value": float(X[ii[o], jj[o]]),
                          "z": float(Z[ii[o], jj[o]]), "stratum": unit.strata[ii[o]]} for o in order]
    # max / p99 on the linear scale, per feature and per stratum
    with np.errstate(invalid="ignore", divide="ignore"):
        fmax = np.nanmax(np.where(np.isfinite(X), X, -np.inf), axis=1)
        fp99 = np.nanpercentile(X, 99, axis=1)
        ratio = np.where(fp99 > 0, fmax / fp99, np.nan)
    M["max_over_p99_per_feature"] = stats.summary(ratio)
    strata = []
    for s in sorted(set(unit.strata), key=str):
        rows = np.array([i for i, x in enumerate(unit.strata) if x == s])
        sub = X[rows]
        obs = sub[np.isfinite(sub)]
        if not obs.size:
            continue
        flat = np.where(np.isfinite(sub), sub, -np.inf)
        i, j = np.unravel_index(int(np.argmax(flat)), sub.shape)
        p99 = float(np.percentile(obs, 99))
        z = Z[rows[i], j]
        e = {"stratum": s, "n_features": int(len(rows)), "max": float(sub[i, j]), "p99": p99,
             "max_over_p99": float(sub[i, j]) / p99 if p99 > 0 else None, "feature": unit.feature_keys[rows[i]],
             "sample": samples[j], "z": float(z) if np.isfinite(z) else None,
             "flagged": bool(np.isfinite(z) and abs(z) > cut)}
        strata.append(e)
        if e["flagged"]:
            f.indicate("outlier_max_cell", {k_: e[k_] for k_ in ("stratum", "max", "feature", "sample", "z")},
                       value=e["max"], stratum=s, feature=e["feature"], sample=e["sample"], z=round(e["z"], 1), cut=cut)
    M["by_stratum"] = {"source": unit.strata_source, "strata": strata}
    if big.sum():
        f.indicate("outlier_cells", {"n_cells": int(big.sum()), "n_features": int((per_feature > 0).sum())},
                   n_cells=int(big.sum()), n_features=int((per_feature > 0).sum()), cut=cut)

    # 3. do flagged samples cluster in run order?
    ro = next((v for v in ctx.variables(unit)[0] if v.numeric and ctx.col_kind.get(v.name) == "run_order"), None)
    if ro is not None and len(flagged) >= 2:
        x = ro.as_float()
        idx = [j for j, s in enumerate(samples) if s in flagged and np.isfinite(x[j])]
        ranks = stats.rankdata(x[np.isfinite(x)])
        pos = {j: r_ for j, r_ in zip(np.where(np.isfinite(x))[0], ranks)}
        obs = _spread([pos[j] for j in idx])
        rng = np.random.default_rng([params["seed"], 11])
        allr = np.array(list(pos.values()))
        null = np.array([_spread(rng.choice(allr, len(idx), replace=False)) for _ in range(params["permutations"])])
        p = (1 + int((null <= obs + 1e-12).sum())) / (1 + len(null))
        M["run_order_clustering"] = {"variable": ro.name, "flagged_positions": [float(pos[j]) for j in idx],
                                     "mean_pairwise_distance": obs, "p": p, "p_min_attainable": 1 / (1 + len(null)),
                                     "n_permutations": len(null)}
    else:
        M["run_order_clustering"] = None
    f.plot["sample_distance"] = [{"sample": r_["sample"], "z": r_.get("z_distance")} for r_ in sample_rows]
    f.method("modified z of distance to the median profile, PC1/PC2 and cells within features",
             n_used=n, transform=unit.transform, cutoff=cut)
    return f


def _spread(pos):
    pos = np.asarray(pos, float)
    if len(pos) < 2:
        return 0.0
    d = np.abs(pos[:, None] - pos[None, :])
    return float(d[np.triu_indices(len(pos), 1)].mean())
