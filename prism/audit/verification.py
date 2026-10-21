"""Verification at C2 (Build Request v4, section 9): re-measurement with the same instrument.

Descriptive measures before and after (C1 vs C2, and S0 vs C2): batch association (PERMANOVA on
the diagnostic matrix, PVCA when the port is available), the relationship-stratified sample
correlation, control-feature trends, per-feature variability of QC samples, completeness.

Negative-control indicators: no batch clustering; same-batch r no longer above different-batch r.
Positive-control indicators: same-subject (or replicate) r above unrelated-sample r; peptide-
within-protein correlation when a peptide-to-protein map exists.

These measures check a chosen plan. They are never used to rank or select methods: nothing here
returns a method name, a ranking or a choice. Group-variance change is descriptive only.
"""

from __future__ import annotations

import numpy as np

from . import context, diagnostic_matrix, stats
from .a07_batch import control_features, relationship_correlation
from .finding import Finding, permanova

AUDIT_ID, CODE = "V", "verification"


def _state_units(st, did, meta, items, params):
    _, ctxs = context.build(st, items, params, [did], {did: meta})
    c = ctxs[0]
    return c, {u.unit_id: u for u in c.units}


def _measures(c, u, params):
    """The instrument: the same descriptive measures at any state."""
    out = {"state_id": c.state["state_id"], "label": c.state.get("label"), "checkpoint": c.state.get("checkpoint")}
    B, G, _ = c.variables(u)
    cat_b = [b for b in B if not b.numeric]
    dm = diagnostic_matrix.default_matrix(u.Y, params)
    out["diagnostic"] = dm.report()
    cols = dm.cols.tolist()
    subj_all = c.subject_codes(u)
    subj = [subj_all[j] for j in cols] if subj_all else None
    rows = []
    if dm.n_features >= 2 and len(cols) >= 3:
        K = stats.gram(dm.Y)
        for b in cat_b:
            r = permanova(K, b.subset(cols), subj, params, f"V|{c.dataset_id}|{u.unit_id}|{b.name}")
            rows.append({k: r.get(k) for k in ("variable", "status", "r2", "value", "p", "p_min_attainable", "n_used")})
    out["batch_association"] = rows
    out["relationship_correlation"] = relationship_correlation(u, c, B)
    out["control_features"] = control_features(u, c, B)
    qc = [j for j, s in enumerate(u.all_samples) if c.role(s) in ("qc", "pool")]
    if len(qc) >= 3:
        Q = u.X_all if c.state_scale in ("log2", "arcsinh") else context.diagnostic_copy(u.X_all)[0]
        Qs = Q[:, qc]
        with np.errstate(all="ignore"):
            sd = np.nanstd(Qs, axis=1, ddof=1)
        out["qc_variability"] = {"n_qc": len(qc), "sd_log2_summary": stats.summary(sd[np.isfinite(sd)]),
                                 "note": "SD of log2 values across QC injections (descriptive)"}
    else:
        out["qc_variability"] = {"status": "not_applicable", "reason": "fewer than 3 QC samples"}
    out["completeness"] = {"n_nan": int(np.isnan(u.X).sum()), "n_features": u.p, "n_samples": u.n}
    gv = []
    for g in G:
        if g.numeric or g.role in ("subject",):
            continue
        levels = g.levels()
        if len(levels) < 2:
            continue
        with np.errstate(all="ignore"):
            within = [np.nanmean(np.nanvar(u.Y[:, [j for j, x in enumerate(g.values) if x == lv]], axis=1, ddof=1))
                      for lv in levels if sum(1 for x in g.values if x == lv) >= 2]
        gv.append({"variable": g.name, "mean_within_group_variance": float(np.nanmean(within)) if within else None})
    out["group_variance"] = {"rows": gv, "note": "descriptive only; never a criterion"}
    try:
        from .. import ports
        if ports.enabled() and dm.n_features >= 3 and len(cols) >= 4 and cat_b:
            from ..ports.probatch import diagnostics
            labels = {b.name: [b.values[j] for j in cols] for b in cat_b}
            for g in G:
                if not g.numeric and g.role != "subject" and len(g.levels()) >= 2:
                    labels[g.name] = [g.values[j] for j in cols]
            fac = list(labels)[:3]
            out["pvca"] = diagnostics.pvca(dm.Y, labels, fac)
    except Exception as e:     # PVCA is optional; its failure is reported, never fatal
        out["pvca"] = {"status": "not_computed", "reason": str(e)[:200]}
    return out


def _indicators(after):
    """Negative and positive controls at one state (descriptive)."""
    neg, pos = {}, {}
    rows = after.get("batch_association") or []
    if rows:
        neg["no_batch_clustering"] = all((r.get("p") is None or r["p"] > 0.05) for r in rows)
    rc = after.get("relationship_correlation") or {}
    eff = [b.get("batch_effect_on_correlation") for b in rc.get("by_batch", []) if b.get("batch_effect_on_correlation")
           is not None]
    if eff:
        neg["same_batch_r_not_above_different_batch_r"] = all(e <= 0 for e in eff)
    if rc.get("same_subject_median_r") is not None and rc.get("different_subject_median_r") is not None:
        pos["same_subject_r_above_unrelated_r"] = rc["same_subject_median_r"] > rc["different_subject_median_r"]
    tr = rc.get("technical_replicate")
    if tr and tr.get("same_median_r") is not None and tr.get("different_median_r") is not None:
        pos["replicate_r_above_unrelated_r"] = tr["same_median_r"] > tr["different_median_r"]
    pos["peptide_within_protein"] = "not_applicable: no peptide-to-protein map"
    return neg, pos


def _compare(before, after):
    """Before/after table of the scalar measures (no judgement)."""
    def pick(m):
        out = {}
        for r in m.get("batch_association") or []:
            out[f"permanova_r2[{r['variable']}]"] = r.get("r2")
        rc = m.get("relationship_correlation") or {}
        for b in rc.get("by_batch", []):
            out[f"batch_effect_on_correlation[{b['variable']}]"] = b.get("batch_effect_on_correlation")
        out["same_subject_median_r"] = rc.get("same_subject_median_r")
        out["different_subject_median_r"] = rc.get("different_subject_median_r")
        out["n_nan"] = (m.get("completeness") or {}).get("n_nan")
        out["n_features"] = (m.get("completeness") or {}).get("n_features")
        qv = (m.get("qc_variability") or {}).get("sd_log2_summary")
        if qv:
            out["qc_sd_log2_median"] = qv.get("median")
        pv = (m.get("pvca") or {}).get("weights")
        if pv:
            for k, v in pv.items():
                out[f"pvca[{k}]"] = v
        return out
    a, b = pick(before), pick(after)
    return [{"measure": k, "before": a.get(k), "after": b.get(k)} for k in sorted(set(a) | set(b))]


def run_dataset(st, did, after_meta, items, params):
    """-> one Finding per unit of the dataset at the C2 state."""
    from ..executor import state as state_mod
    chain = [after_meta]
    m = after_meta
    while m.get("parent"):
        m = state_mod.meta(st, did, m["parent"])
        chain.append(m)
    root = chain[-1]
    c1 = next((x for x in chain[1:] if x.get("checkpoint") == "C1"), None)
    ca, ua = _state_units(st, did, after_meta, items, params)
    states = {"S0": _state_units(st, did, root, items, params)}
    if c1:
        states["C1"] = _state_units(st, did, c1, items, params)
    out = []
    for uid, u in ua.items():
        f = Finding(AUDIT_ID, CODE, u).feeds("plan_check")
        after = _measures(ca, u, params)
        M = f.measures
        M["after"] = after
        M["comparisons"] = {}
        for name, (cb, ub) in states.items():
            if uid not in ub:
                continue
            before = _measures(cb, ub[uid], params)
            M["comparisons"][f"{name}_vs_C2"] = {"before_state": before["state_id"], "table": _compare(before, after),
                                                  "nan_unchanged_except_removals":
                                                      before["completeness"]["n_nan"] >= after["completeness"]["n_nan"]}
        neg, pos = _indicators(after)
        M["negative_controls"], M["positive_controls"] = neg, pos
        M["use"] = ("These measures check the chosen plan. They are never used to rank or select methods; "
                    "group-variance change is descriptive only.")
        f.method("re-measurement at C2 with the same instrument: PERMANOVA on the diagnostic matrix, PVCA when "
                 "available, relationship-stratified correlation, control features, QC variability, completeness",
                 n_used=u.n)
        out.append(f)
    return out
