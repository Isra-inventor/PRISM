"""Verification at C2 (Build Request v4, section 9): re-measurement with the same instrument.
Ported from the v4 build.

Descriptive measures before and after (C1 vs C2, and S0 vs C2): batch association (PERMANOVA on
the diagnostic matrix, PVCA when the port is available), the relationship-stratified sample
correlation, control-protein trends, variability of the QC samples, completeness.

Negative-control indicators: no batch clustering; same-batch r no longer above different-batch r.
Positive-control indicators: same-subject (or replicate) r above unrelated-sample r; peptide-within-
protein correlation when a peptide-to-protein map exists (not in protein-level tables).

These measures check a chosen plan. They are never used to rank or select methods: nothing here
returns a method name, a ranking or a choice. Group-variance change is descriptive only.
"""

from __future__ import annotations

import numpy as np

from . import bridge, stats
from . import diagnostic as diagnostic_matrix
from .batch import control_features, relationship_correlation
from .bridge import permanova
from .finding import Finding
from .scale import diagnostic_y


def measures(audit, block: str) -> dict:
    """The instrument: the same descriptive measures at any state."""
    u = bridge.unit(audit, block)
    params = audit.params
    out = {"state_id": audit.state_id or "S0", "checkpoint": audit.checkpoint}
    B, G, _ = bridge.variables(audit, u.samples)
    cat_b = [b for b in B if not b.numeric]
    dm = diagnostic_matrix.default_matrix(u.Y, params)
    out["diagnostic"] = dm.report()
    cols = dm.cols.tolist()
    subj_all = bridge.subject_codes(audit, u.samples)
    subj = [subj_all[j] for j in cols] if subj_all else None
    rows = []
    if dm.n_features >= 2 and len(cols) >= 3:
        K = stats.gram(dm.Y)
        for b in cat_b:
            r = permanova(K, b.subset(cols), subj, params, f"V|{block}|{b.name}")
            rows.append({k: r.get(k) for k in ("variable", "status", "r2", "value", "p", "p_min_attainable", "n_used")})
    out["batch_association"] = rows
    out["relationship_correlation"] = relationship_correlation(audit, u, B)
    out["control_features"] = control_features(audit, u, B)
    full = audit.full_matrix(block)
    types = audit.schema.samples().set_index("sample")["sample_type"]
    qc = [c for c in full.columns if types.get(c) in ("qc", "pool")]
    if len(qc) >= 3:
        Q = full[qc].to_numpy(dtype=float)
        if not (audit.state is not None and audit.state.scale in ("log2", "arcsinh")):
            Q = diagnostic_y(Q, audit.results["scale"][block])[0]
        with np.errstate(all="ignore"):
            sd = np.nanstd(Q, axis=1, ddof=1)
        out["qc_variability"] = {"n_qc": len(qc), "sd_summary": stats.summary(sd[np.isfinite(sd)]),
                                 "note": "SD of the diagnostic-scale values across QC injections (descriptive)"}
    else:
        out["qc_variability"] = {"status": "not_applicable", "reason": "fewer than 3 QC samples"}
    out["completeness"] = {"n_nan": int(np.isnan(u.X).sum()), "n_proteins": u.p, "n_samples": u.n}
    gv = []
    for g in G:
        if g.numeric or g.role == "subject" or len(g.levels()) < 2:
            continue
        with np.errstate(all="ignore"):
            within = [np.nanmean(np.nanvar(u.Y[:, [j for j, x in enumerate(g.values) if x == lv]], axis=1, ddof=1))
                      for lv in g.levels() if sum(1 for x in g.values if x == lv) >= 2]
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
            out["pvca"] = diagnostics.pvca(dm.Y, labels, list(labels)[:3])
    except Exception as e:              # PVCA is optional; its failure is reported, never fatal
        out["pvca"] = {"status": "not_computed", "reason": str(e)[:200]}
    return out


def indicators(after: dict) -> tuple[dict, dict]:
    """Negative and positive controls at one state (descriptive)."""
    neg, pos = {}, {}
    rows = after.get("batch_association") or []
    if rows:
        neg["no_batch_clustering"] = all((r.get("p") is None or r["p"] > 0.05) for r in rows)
    rc = after.get("relationship_correlation") or {}
    eff = [b.get("batch_effect_on_correlation") for b in rc.get("by_batch", [])
           if b.get("batch_effect_on_correlation") is not None]
    if eff:
        neg["same_batch_r_not_above_different_batch_r"] = all(e <= 0 for e in eff)
    if rc.get("same_subject_median_r") is not None and rc.get("different_subject_median_r") is not None:
        pos["same_subject_r_above_unrelated_r"] = rc["same_subject_median_r"] > rc["different_subject_median_r"]
    tr = rc.get("technical_replicate")
    if tr and tr.get("same_median_r") is not None and tr.get("different_median_r") is not None:
        pos["replicate_r_above_unrelated_r"] = tr["same_median_r"] > tr["different_median_r"]
    pos["peptide_within_protein"] = "not_applicable: no peptide-to-protein map"
    return neg, pos


def compare(before: dict, after: dict) -> list[dict]:
    """Before / after table of the scalar measures (no judgement)."""
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
        out["n_proteins"] = (m.get("completeness") or {}).get("n_proteins")
        qv = (m.get("qc_variability") or {}).get("sd_summary")
        if qv:
            out["qc_sd_median"] = qv.get("median")
        for k, v in ((m.get("pvca") or {}).get("weights") or {}).items():
            out[f"pvca[{k}]"] = v
        return out
    a, b = pick(before), pick(after)
    return [{"measure": k, "before": a.get(k), "after": b.get(k)} for k in sorted(set(a) | set(b))]


def run(after_audit, before_audits: dict) -> Finding:
    """after_audit: the Audit at C2; before_audits: {"S0": Audit, "C1": Audit, ...}."""
    f = Finding("V", "Verification", step="C2", feeds=["the plan check (never method selection)"],
                method="re-measurement with the same instrument: PERMANOVA on the diagnostic matrix, PVCA when "
                       "available, relationship-stratified correlation, control proteins, QC variability, completeness")
    for bid in after_audit.schema.block_map():
        after = measures(after_audit, bid)
        m = {"after": after, "comparisons": {}}
        for name, a in before_audits.items():
            before = measures(a, bid)
            m["comparisons"][f"{name}_vs_C2"] = {"table": compare(before, after),
                                                 "nan_unchanged_except_removals":
                                                     before["completeness"]["n_nan"] >= after["completeness"]["n_nan"]}
        neg, pos = indicators(after)
        m["negative_controls"], m["positive_controls"] = neg, pos
        m["use"] = ("These measures check the chosen plan. They are never used to rank or select methods; "
                    "group-variance change is descriptive only.")
        f.measures[bid] = m
        f.say("ok", f"{bid}: negative controls {neg}; positive controls "
                    f"{ {k: v for k, v in pos.items() if not isinstance(v, str)} }.")
    return f
