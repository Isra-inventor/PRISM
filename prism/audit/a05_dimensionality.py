"""A5. Dimensionality, sparsity, effective sample size (v3 §6.3). Numbers only: Tier 2 decides
what they mean."""

from __future__ import annotations

import numpy as np

from . import stats
from .a04_missingness import floor_ties
from .context import is_num
from .finding import Finding


def _key(v):
    return (0, float(v), "") if is_num(v) else (1, 0.0, v)

AUDIT_ID, CODE = "A5", "dimensionality"


def run(unit, ctx, params, results):
    f = Finding(AUDIT_ID, CODE, unit).feeds("feature_filtering", "method_stability")
    if unit.out_of_scope:
        return f.status("not_applicable", f"Tier 1 does not audit {unit.out_of_scope} ({unit.scale}).")
    X = unit.X
    p, n = X.shape
    M = f.measures
    subj = ctx.subject_codes(unit)
    n_subj = len({s for s in subj if s}) if subj else None
    M["n"], M["p"], M["n_over_p"] = n, p, n / p if p else None
    M["n_subjects"] = n_subj
    M["n_subjects_over_p"] = n_subj / p if (n_subj and p) else None
    if p and n / p < 1:
        f.indicate("few_samples_per_feature", {"n": n, "p": p}, n=n, p=p, ratio=round(n / p, 4))

    # 2. cell counts
    B, G, _ = ctx.variables(unit)
    cells = []
    for v in B + G:
        if v.numeric or v.role == "subject":
            continue
        counts = {}
        for x in v.values:
            if x:
                counts[x] = counts.get(x, 0) + 1
        if counts:
            counts = {k: counts[k] for k in sorted(counts, key=_key)}
            lev = min(counts, key=lambda k: (counts[k], k))
            cells.append({"variable": v.name, "role": v.role, "counts": counts, "smallest": {"level": lev, "n": counts[lev]}})
    M["cells"] = cells
    if cells:
        sm = min(cells, key=lambda c: (c["smallest"]["n"], c["variable"]))
        M["smallest_cell"] = {"variable": sm["variable"], "level": sm["smallest"]["level"], "n": sm["smallest"]["n"]}
        f.indicate("small_cell", M["smallest_cell"], size=sm["smallest"]["n"], variable=sm["variable"],
                   level=sm["smallest"]["level"])
    else:
        M["smallest_cell"] = None

    # 3. sparsity and detection
    R = ~np.isfinite(X)
    prev = results.get(unit.dataset_id, {}).get(unit.unit_id, {}).get("floor")
    S = prev["S"] if prev else floor_ties(X, params["floor_min_observed"])[3]
    zero = (X == 0) & ~S
    sparse = R | zero | S
    M["sparsity"] = {"share_missing_zero_or_floor": float(sparse.mean()), "share_missing": float(R.mean()),
                     "share_zero": float(zero.mean()), "share_floor": float(S.mean())}
    det = 1 - (R.mean(1) + S.mean(1))
    M["detection_rate"] = dict(stats.summary(det), **stats.quantiles(det, (0.1, 0.9)))
    f.plot["detection_rate"] = stats.histogram(det, 20)

    # 4. Kish design effect with the median ICC from A8
    icc = results.get(unit.dataset_id, {}).get(unit.unit_id, {}).get("median_icc")
    if n_subj and icc is not None:
        m = n / n_subj
        rho = max(0.0, icc)
        deff = 1 + (m - 1) * rho
        M["design_effect"] = {"m": m, "median_icc": icc, "rho_used": rho, "deff": deff, "n_eff": n / deff,
                              "note": "median ICC truncated at 0"}
        f.indicate("effective_sample_size", M["design_effect"], n_subjects=n_subj, icc=round(icc, 3),
                   n_eff=round(n / deff, 1), n=n, deff=round(deff, 3))
    else:
        M["design_effect"] = {"status": "insufficient_data",
                              "reason": "needs a subject and the A8 ICC (at least 3 subjects with 2 or more samples)"}
        if not n_subj:
            f.status("computed", needs=["subject"])
    f.method("n, p, cell counts, sparsity, Kish design effect", n_used=n, transform=unit.transform)
    return f
