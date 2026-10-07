"""A9. Tissue and sample-source heterogeneity (v3 §6.3), with the A7 machinery."""

from __future__ import annotations

from . import stats
from .context import Variable
from .finding import Finding, association, permanova, table

AUDIT_ID, CODE = "A9", "source_heterogeneity"


def run(unit, ctx, params, results):
    f = Finding(AUDIT_ID, CODE, unit).feeds("tissue_normalization", "stratification")
    if unit.out_of_scope:
        return f.status("not_applicable", f"Tier 1 does not audit {unit.out_of_scope} ({unit.scale}).")
    ov = {o["column"] for o in ctx.overrides if o["kind"] == "source_variable"}
    names = [c for c in ctx.M_columns if (ctx.col_kind.get(c) == "sample_type" or c in ov)
             and any((ctx.column(c) or {}).get(s) for s in unit.samples)]
    M = f.measures
    if not names:
        return f.status("insufficient_metadata", "no column has audit_kind sample_type and no source_variable override",
                        needs=["sample source"])
    subj = ctx.subject_codes(unit)
    Yc = stats.complete_features(unit.Y)
    K = stats.gram(Yc) if Yc.shape[0] >= 2 else None
    k = min(params["max_pcs"], unit.n - 1)
    scores, ve = stats.pca(Yc, k) if K is not None else (None, [])
    out, rows = [], []
    for c in names:
        vals = ctx.column(c)
        v = Variable(c, [vals.get(s, "") for s in unit.samples], "source", "categorical",
                     "override" if c in ov else "audit_kind sample_type")
        levels = {}
        for x in v.values:
            if x:
                levels[x] = levels.get(x, 0) + 1
        e = {"variable": c, "levels": levels}
        if subj:
            per = {}
            for x, s in zip(v.values, subj):
                if x and s:
                    per.setdefault(s, set()).add(x)
            e["constant_within_subject"] = all(len(xs) == 1 for xs in per.values()) if per else None
        if len(levels) <= 1:
            e["homogeneous"] = True
            f.indicate("single_source", {"variable": c, "level": next(iter(levels), None)}, variable=c,
                       level=next(iter(levels), "(empty)"))
        elif scores is not None:
            e["homogeneous"] = False
            for j in range(k):
                r = association(scores[:, j], v, subj, params, f"A9|{unit.dataset_id}|{unit.unit_id}|PC{j + 1}|{c}",
                                categorical="eta_squared")
                r["pc"] = j + 1
                rows.append(r)
            r = permanova(K, v, subj, params, f"A9|{unit.dataset_id}|{unit.unit_id}|PERMANOVA|{c}")
            r["pc"] = None
            rows.append(r)
        out.append(e)
    rows, n_tests = table(rows)
    M["variables"] = out
    M["associations"] = {"n_tests": n_tests, "rows": rows, "variance_explained": [float(x) for x in ve]}
    for r in rows:
        if r.get("q") is not None and r["q"] <= 0.05:
            f.indicate("source_association", {k_: r.get(k_) for k_ in ("variable", "test", "pc", "value", "q")},
                       variable=r["variable"], what="PERMANOVA" if r["test"] == "permanova" else f"PC{r['pc']}",
                       q=round(r["q"], 4))
    f.method("level counts; eta squared per PC and marginal PERMANOVA with restricted permutations", n_used=unit.n,
             transform=unit.transform)
    return f
