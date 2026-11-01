"""S10 + G4 / A7: batch structure. Ported from the v4 build; statistics unchanged.

Batch candidates (B): sample variables with audit_kind batch or run_order. Design variables (G):
subject, time, group, sample_type, covariate. On the audit set (study samples, unflagged proteins):

 1. structure of every batch candidate against every design variable: the contingency table,
    Cramer's V, and a code: confounded (each batch holds one design level and the reverse, or the
    design matrix is rank-deficient), nested (one is constant within the other), crossed_balanced,
    partially_confounded. Also batch candidates against each other (rule R9).
 2. v4 section 4.4: samples per batch (a level with fewer than 25 samples is reported; the figure is
    a heuristic quoted in Cuklina et al. 2021 from other work, unverified), batch coverage (share of
    proteins with >= 1 and >= 2 observed values in every level), the relationship-stratified sample
    correlation (Pearson r on the diagnostic copy, pairwise-complete: same subject / different subject,
    same batch / different batch; batch_effect_on_correlation = median r same batch - median r
    different batch, within same-subject and different-subject pairs), control-protein trends against
    run order for proteins you mark as controls.
 3. PCA (SVD of the protein-centred diagnostic matrix, v4 section 3 policy): each of the first PCs
    against every variable (eta squared or Spearman, restricted permutations), and marginal PERMANOVA
    on Euclidean distances; BH over all tests.
 4. The sensitivity block: the strongest batch association recomputed under every diagnostic policy
    (base, exclude_floor, floor_censored, q = 100 / 75 / 50). If they disagree:
    diagnostic_conclusion_depends_on_policy. No variant is preferred.
"""

from __future__ import annotations

import numpy as np

from . import bridge, stats
from . import diagnostic as diagnostic_matrix
from .bridge import association, permanova, table
from .finding import Finding


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


def run(audit) -> Finding:
    f = Finding("A7", "Batch structure", step="S10", feeds=["S2 batch correction", "G7 missingness handling",
                                                            "S10 confounding and design"])
    params = audit.params

    def block(unit, f):
        B, G, skipped = bridge.variables(audit, unit.samples)
        M = f.measures
        M["variables"] = {"B": [v.describe() for v in B], "G": [v.describe() for v in G], "skipped": skipped}
        if not B:
            f.set_status("insufficient_metadata", "no batch candidate: no sample information column has audit_kind "
                                                  "batch or run_order", needs=["batch variable"])
        struct = []
        for b in B:
            for g in G:
                if b.numeric or g.numeric:
                    continue
                s = structure(b, g)
                struct.append(s)
                if s.get("code") == "confounded":
                    f.indicate("batch_confounded", {k: s.get(k) for k in ("batch", "design", "cramers_v")},
                               batch=b.name, design=g.name)
                elif s.get("code") == "nested":
                    f.indicate("batch_nested", {k: s[k] for k in ("batch", "design", "nesting")}, text=s["nesting"])
        M["structure"] = struct
        cat_b = [b for b in B if not b.numeric]
        M["structure_batch_pairs"] = [structure(cat_b[i], cat_b[j]) for i in range(len(cat_b))
                                      for j in range(i + 1, len(cat_b))]
        M["samples_per_batch"], M["batch_coverage"] = batch_sizes(f, unit, B, params)
        M["relationship_correlation"] = relationship_correlation(audit, unit, B)
        M["control_features"] = control_features(audit, unit, B)

        dm = diagnostic_matrix.default_matrix(unit.Y, params)
        M["diagnostic"] = dm.report(unit.samples)
        Y = dm.Y
        cols = dm.cols.tolist()
        n = len(cols)
        subj_all = bridge.subject_codes(audit, unit.samples)
        subj = [subj_all[j] for j in cols] if subj_all else None
        variables = [v for v in B + G if v.role != "subject"] + [v for v in G if v.role == "subject"]
        variables = [v.subset(cols) for v in variables] if n != unit.n else variables
        if Y.shape[0] < 2 or n < 3:
            M["pca"] = {"status": "insufficient_data", "reason": "fewer than 2 proteins in the diagnostic matrix or "
                                                                 "3 samples"}
            f.set_method("structure codes", n_used=n)
            return
        k = min(params["max_pcs"], n - 1)
        scores, var_exp = stats.pca(Y, k)
        rows = []
        for j in range(k):
            for v in variables:
                key = f"A7|{unit.block}|PC{j + 1}|{v.name}"
                if v.role == "subject":
                    r = association(scores[:, j], v, None, params, key, guard_subjects=False, categorical="eta_squared")
                else:
                    r = association(scores[:, j], v, subj, params, key, categorical="eta_squared")
                r["pc"] = j + 1
                rows.append(r)
        K = stats.gram(Y)
        for v in variables:
            r = permanova(K, v, None if v.role == "subject" else subj, params, f"A7|{unit.block}|PERMANOVA|{v.name}")
            r["pc"] = None
            rows.append(r)
        rows, n_tests = table(rows)
        M["pca"] = {"n_proteins_complete": int(Y.shape[0]), "of": unit.p, "k": k, "policy": dm.policy, "q": dm.q,
                    "variance_explained": [float(x) for x in var_exp]}
        M["associations"] = {"n_tests": n_tests, "rows": rows}
        for r in rows:
            if r.get("q") is None or r["q"] > 0.05:
                continue
            if r["test"] == "permanova":
                f.indicate("permanova_association", {k_: r.get(k_) for k_ in ("variable", "r2", "value", "p", "q")},
                           variable=r["variable"], r2=round(100 * r["r2"], 1), F=round(r["value"], 2),
                           q=round(r["q"], 4))
            else:
                f.indicate("pc_association", {k_: r.get(k_) for k_ in ("variable", "pc", "statistic", "value", "p", "q")},
                           pc=r["pc"], pct=round(100 * var_exp[r["pc"] - 1], 1), variable=r["variable"],
                           stat="eta squared" if r["statistic"] == "eta_squared" else "rho", value=round(r["value"], 3),
                           q=round(r["q"], 4))
        f.plot["pca"] = {"variance_explained": [float(x) for x in var_exp],
                         "samples": [{"sample": unit.samples[j], "scores": [float(x) for x in scores[i]],
                                      "values": {v.name: v.values[j] for v in B + G}} for i, j in enumerate(cols)]}
        if B:
            sens = sensitivity(unit, params, B, subj_all)
            M["sensitivity"] = sens
            if sens["depends"]:
                f.indicate("diagnostic_conclusion_depends_on_policy", {"variants": sens["variants"]},
                           what="the strongest batch association", n=sum(1 for v in sens["variants"] if v["computable"]))
        if not f.sentences:
            f.say("ok", f"no batch or design variable is associated with the first {k} PCs or the PERMANOVA "
                        f"(BH q > 0.05; {dm.n_features} complete proteins, policy {dm.policy}, q {dm.q}; "
                        f"{diagnostic_matrix.BIAS_WARNING}).")
        f.set_method("contingency structure, PCA (SVD of protein-centred Y on the diagnostic matrix), eta squared / "
                     "Spearman and marginal PERMANOVA with restricted permutations", n_used=n, transform=unit.transform,
                     k=k, permutations=params["permutations"], seed=params["seed"], diagnostic_policy=dm.policy, q=dm.q)

    return bridge.per_block(audit, f, block)


def _conclusion(d, B, subj_all, params, key):
    """The strongest batch association (PERMANOVA R2 over the batch candidates) and the set of batch
    candidates with p <= 0.05 under one diagnostic matrix; None when not computable."""
    cols = d.cols.tolist()
    if d.n_features < 2 or len(cols) < 3:
        return None
    K = stats.gram(d.Y)
    subj = [subj_all[j] for j in cols] if subj_all else None
    best, sig = None, []
    for b in B:
        r = permanova(K, b.subset(cols), subj, params, f"{key}|{b.name}")
        if r.get("status") != "computed":
            continue
        if best is None or r["r2"] > best[1]:
            best = (b.name, r["r2"])
        if r["p"] <= 0.05:
            sig.append(b.name)
    if best is None:
        return None
    return (best[0], tuple(sorted(sig)))


def sensitivity(unit, params, B, subj_all):
    key = f"A7sens|{unit.block}"
    vs, n_tied = diagnostic_matrix.variants(unit.Y, unit.X, params)
    res = [(name, d, _conclusion(d, B, subj_all, params, key)) for name, d in vs]
    block = diagnostic_matrix.sensitivity_block(
        res, lambda c: {"strongest_batch_variable": c[0], "batch_variables_p_le_0_05": list(c[1])})
    block["n_floor_tied_proteins"] = n_tied
    block["conclusion_definition"] = ("the batch candidate with the largest marginal PERMANOVA R2, and the batch "
                                      "candidates with permutation p <= 0.05")
    return block


def batch_sizes(f, unit, B, params):
    sizes, cover = [], []
    R = ~np.isfinite(unit.X)
    for b in B:
        if b.numeric:
            continue
        levels = b.levels()
        counts = {lev: sum(1 for x in b.values if x == lev) for lev in levels}
        small = [lev for lev, c in counts.items() if c < params["batch_min_samples"]]
        sizes.append({"variable": b.name, "counts": counts, "min": min(counts.values()) if counts else None,
                      "median": float(np.median(list(counts.values()))) if counts else None,
                      "below_suggestion": small, "suggestion": params["batch_min_samples"],
                      "suggestion_evidence": "heuristic: quoted in Cuklina et al. 2021 as a suggestion from other work; "
                                             "unverified"})
        if small:
            f.indicate("small_batches", {"variable": b.name, "levels": small}, variable=b.name, n=len(small),
                       cut=params["batch_min_samples"])
        c1, c2 = np.ones(unit.p, bool), np.ones(unit.p, bool)
        for lev in levels:
            ix = [j for j, x in enumerate(b.values) if x == lev]
            obs = (~R[:, ix]).sum(1)
            c1 &= obs >= 1
            c2 &= obs >= 2
        cover.append({"variable": b.name, "share_observed_ge1_every_level": float(c1.mean()) if unit.p else None,
                      "share_observed_ge2_every_level": float(c2.mean()) if unit.p else None})
    return sizes, cover


def _median(v):
    return float(np.median(v)) if len(v) else None


def relationship_correlation(audit, unit, B):
    """Pearson r on Y (pairwise-complete) for all sample pairs, by pair class. proBatch labels pairs by
    biospecimen replicate and batch; PRISM designs have subjects, so subject is the stand-in."""
    R, N = diagnostic_matrix.pairwise_corr(unit.Y)
    n = unit.n
    subj = bridge.subject_codes(audit, unit.samples)
    tech = next((v for k, v in bridge.column_values(audit, unit.samples).values() if k == "technical_replicate"), None)
    iu = np.triu_indices(n, 1)
    r = R[iu]
    ok = np.isfinite(r)
    out = dict(diagnostic_matrix.pairwise_report(N), n_pairs=int(ok.sum()),
               subject_stand_in="subject is used as the stand-in for proBatch's biospecimen replicate")
    same_subj = np.array([bool(subj and subj[i] and subj[i] == subj[j]) for i, j in zip(*iu)]) if subj else None
    if same_subj is not None:
        out["same_subject_median_r"] = _median(r[ok & same_subj])
        out["different_subject_median_r"] = _median(r[ok & ~same_subj])
    if tech:
        same_t = np.array([bool(tech[i] and tech[i] == tech[j]) for i, j in zip(*iu)])
        out["technical_replicate"] = {"same_median_r": _median(r[ok & same_t]),
                                      "different_median_r": _median(r[ok & ~same_t]),
                                      "n_same_pairs": int((ok & same_t).sum())}
    per_batch = []
    for b in B:
        if b.numeric:
            continue
        same_b = np.array([bool(b.values[i] and b.values[i] == b.values[j]) for i, j in zip(*iu)])
        e = {"variable": b.name, "same_batch_median_r": _median(r[ok & same_b]),
             "different_batch_median_r": _median(r[ok & ~same_b])}
        if e["same_batch_median_r"] is not None and e["different_batch_median_r"] is not None:
            e["batch_effect_on_correlation"] = e["same_batch_median_r"] - e["different_batch_median_r"]
        if same_subj is not None:
            for name, m in (("within_same_subject", same_subj), ("within_different_subject", ~same_subj)):
                a, c = _median(r[ok & m & same_b]), _median(r[ok & m & ~same_b])
                e[name] = {"same_batch_median_r": a, "different_batch_median_r": c,
                           "batch_effect_on_correlation": (a - c) if (a is not None and c is not None) else None}
        per_batch.append(e)
    out["by_batch"] = per_batch
    return out


def control_features(audit, unit, B):
    marked = [o for o in audit.overrides if o["kind"] == "control_feature"]
    if not marked:
        return {"status": "not_applicable", "reason": "no protein is marked as a control "
                                                      "(a.mark_control_protein(...))"}
    ro = next((b for b in B if b.numeric and "run_order" in b.source), None)
    idx = {k: i for i, k in enumerate(unit.feature_keys)}
    rows = []
    for o in marked:
        i = idx.get(o["feature"])
        if i is None:
            rows.append({"feature": o["feature"], "status": "not_found"})
            continue
        e = {"feature": o["feature"], "observed": int(np.isfinite(unit.Y[i]).sum())}
        if ro is None:
            e.update(status="insufficient_metadata", reason="no run order")
        else:
            x, y = ro.as_float(), unit.Y[i]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 4:
                e.update(status="computed", spearman_run_order=stats.spearman(x[ok], y[ok])[0],
                         ols_slope=stats.ols_slope(x[ok], y[ok]), n_used=int(ok.sum()))
            else:
                e["status"] = "insufficient_data"
        rows.append(e)
    return {"status": "computed", "features": rows, "run_order_variable": ro.name if ro else None}
