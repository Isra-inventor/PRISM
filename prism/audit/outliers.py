"""S4 / A11: outlier flags. Ported from the v4 build. Flags only: nothing is removed.

 1. samples, on the diagnostic matrix (v4 section 3): Euclidean distance to the median profile,
    PC1 and PC2 scores, each turned into a modified z (median and MAD); |z| > 3.5 (Iglewicz and
    Hoaglin; heuristic) flags the sample. The sensitivity block recomputes the flagged set under every
    diagnostic policy and reports when it changes.
 2. cells: the modified z within each protein; how many cells, in how many proteins, exceed 3.5;
    the 50 most extreme cells; the largest value of the block against its 99th percentile.
 3. when two or more samples are flagged and a run order exists: do they sit close together in the
    run (a permutation test on their mean distance in run-order ranks)?
"""

from __future__ import annotations

import numpy as np

from . import bridge, stats
from . import diagnostic as diagnostic_matrix
from .finding import Finding

TOP_CELLS = 50


def flag_samples(Yc, samples, cut):
    """Distance to the median profile, PC1 / PC2 scores and r with the median on a complete matrix."""
    n = Yc.shape[1]
    med = np.median(Yc, axis=1)
    d = np.sqrt(((Yc - med[:, None]) ** 2).sum(0))
    zd = stats.modified_z(d)
    r = [stats.pearson(Yc[:, j], med) for j in range(n)]
    k = min(2, n - 1)
    scores, _ = stats.pca(Yc, k)
    zpc = [stats.modified_z(scores[:, j]) for j in range(k)]
    rows = []
    for j in range(n):
        row = {"sample": samples[j], "distance": float(d[j]), "z_distance": float(zd[j]), "r_median": float(r[j])}
        row.update({f"z_pc{i + 1}": float(zpc[i][j]) for i in range(k)})
        reasons = (["distance"] if zd[j] > cut else []) + [f"pc{i + 1}" for i in range(k) if abs(zpc[i][j]) > cut]
        row["flags"] = reasons
        rows.append(row)
    return rows, k


def sensitivity(Y, X, samples, params, cut):
    vs, n_tied = diagnostic_matrix.variants(Y, X, params)
    res = []
    for name, d in vs:
        cols = d.cols.tolist()
        if d.n_features < 2 or len(cols) < 3:
            res.append((name, d, None))
            continue
        rows, _ = flag_samples(d.Y, [samples[j] for j in cols], cut)
        res.append((name, d, tuple(sorted(r["sample"] for r in rows if r["flags"]))))
    block = diagnostic_matrix.sensitivity_block(res, lambda c: {"flagged_samples": list(c)})
    block["n_floor_tied_proteins"] = n_tied
    block["conclusion_definition"] = "the set of flagged samples"
    return block


def _spread(pos):
    pos = np.asarray(pos, float)
    if len(pos) < 2:
        return 0.0
    d = np.abs(pos[:, None] - pos[None, :])
    return float(d[np.triu_indices(len(pos), 1)].mean())


def run(audit) -> Finding:
    f = Finding("A11", "Outliers", step="S4", feeds=["S3 feature filtering", "S4 sample exclusion"])
    params = audit.params
    cut = params["modified_z_cutoff"]

    def block(unit, f):
        X, Y, n, samples = unit.X, unit.Y, unit.n, unit.samples
        M = f.measures
        if n < 3:
            f.set_status("insufficient_data", "fewer than 3 samples")
            return
        dm = diagnostic_matrix.default_matrix(Y, params)
        M["diagnostic"] = dm.report(samples)
        flagged, sample_rows = set(), []
        cols = dm.cols.tolist()
        if dm.n_features >= 2 and len(cols) >= 3:
            sample_rows, k = flag_samples(dm.Y, [samples[j] for j in cols], cut)
            flagged = {r_["sample"] for r_ in sample_rows if r_["flags"]}
            M["samples"] = {"n_proteins_complete": dm.n_features, "cutoff": cut, "policy": dm.policy, "q": dm.q,
                            "flagged": sorted(flagged), "per_sample": sample_rows}
            for r_ in sample_rows:
                if r_["flags"]:
                    z = max([abs(r_.get("z_distance", 0))] + [abs(r_.get(f"z_pc{i + 1}", 0)) for i in range(k)])
                    f.indicate("outlier_sample", {k_: r_.get(k_) for k_ in ("sample", "z_distance", "flags")},
                               sample=r_["sample"], by=", ".join(r_["flags"]), z=round(z, 2), cut=cut)
            sens = sensitivity(Y, X, samples, params, cut)
            M["sensitivity"] = sens
            if sens["depends"]:
                f.indicate("diagnostic_conclusion_depends_on_policy", {"variants": sens["variants"]},
                           what="the set of flagged samples", n=sum(1 for v in sens["variants"] if v["computable"]))
            if not flagged:
                f.say("ok", f"no sample is an outlier by distance or PC1 / PC2 (modified z cut-off {cut}; "
                            f"{dm.n_features} complete proteins, policy {dm.policy}, q {dm.q}).")
        else:
            M["samples"] = {"status": "insufficient_data", "reason": "fewer than 2 proteins in the diagnostic matrix"}

        Z = stats.row_modified_z(Y)
        big = np.abs(np.nan_to_num(Z)) > cut
        per_feature, per_sample = big.sum(1), big.sum(0)
        M["cells"] = {"cutoff": cut, "n_flagged": int(big.sum()), "n_proteins_with_any": int((per_feature > 0).sum()),
                      "per_sample": {s: int(per_sample[j]) for j, s in enumerate(samples)}}
        ii, jj = np.where(big)
        order = np.lexsort((jj, ii, -np.abs(Z[ii, jj])))[:TOP_CELLS]
        M["cells"]["top"] = [{"protein": unit.feature_keys[ii[o]], "sample": samples[jj[o]],
                              "value": float(X[ii[o], jj[o]]), "z": float(Z[ii[o], jj[o]])} for o in order]
        if big.sum():
            f.indicate("outlier_cells", {"n_cells": int(big.sum())}, n_cells=int(big.sum()),
                       n_features=int((per_feature > 0).sum()), cut=cut)

        B, _, _ = bridge.variables(audit, samples)
        ro = next((v for v in B if v.numeric and "run_order" in v.source), None)
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
            M["run_order_clustering"] = {"variable": ro.name, "mean_pairwise_distance": obs, "p": p,
                                         "p_min_attainable": 1 / (1 + len(null))}
        else:
            M["run_order_clustering"] = None
        f.set_method("modified z of distance to the median profile, PC1 / PC2 and cells within proteins",
                     n_used=n, transform=unit.transform, cutoff=cut)

    return bridge.per_block(audit, f, block)
