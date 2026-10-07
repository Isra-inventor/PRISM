"""A6. Technical and platform noise, QC indicators (v3 §6.3).

Sample roles (qc / blank / pool / study) come from overrides.json: Step 0 records only
is_study_sample. Non-study samples without a role make items 1 and 2 insufficient_metadata.
"""

from __future__ import annotations

import re

import numpy as np

from . import stats
from .a04_missingness import floor_ties
from .context import is_num
from .finding import Finding

AUDIT_ID, CODE = "A6", "noise_qc"
QC_NAME = re.compile(r"(^|[^a-z])(qc|cv|rsd|lod|loq|signal.?to.?noise|s/?n|colcheck|rowcheck)([^a-z]|$)", re.I)


def lowess(x, y, frac=0.75):
    """Locally linear fit with tricube weights (deterministic, no robustness iterations)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    k = max(3, int(np.ceil(frac * n)))
    fit = np.empty(n)
    for i in range(n):
        d = np.abs(x - x[i])
        h = np.sort(d)[min(k, n) - 1] or 1.0
        w = np.clip(1 - (d / h) ** 3, 0, None) ** 3
        W = w.sum()
        mx, my = (w * x).sum() / W, (w * y).sum() / W
        sxx = (w * (x - mx) ** 2).sum()
        b = (w * (x - mx) * (y - my)).sum() / sxx if sxx > 0 else 0.0
        fit[i] = my + b * (x[i] - mx)
    return fit


def run(unit, ctx, params, results):
    f = Finding(AUDIT_ID, CODE, unit).feeds("normalization", "drift_correction", "feature_filtering")
    if unit.out_of_scope:
        return f.status("not_applicable", f"Tier 1 does not audit {unit.out_of_scope} ({unit.scale}).")
    M = f.measures
    all_s = unit.all_samples
    roles = {s: ctx.role(s) for s in all_s if s not in ctx.excluded_by_override}
    count = {}
    for r in roles.values():
        count[r] = count.get(r, 0) + 1
    M["roles"] = {"counts": count, "source": "overrides.json (sample_role); Step 0 records only is_study_sample"}
    Xa = unit.X_all
    col = {s: j for j, s in enumerate(all_s)}

    def cols(role):
        return [col[s] for s, r in roles.items() if r == role]

    unassigned = cols("unassigned")
    study = cols("study")
    qc = cols("qc") + cols("pool")
    blanks = cols("blank")
    # 1-2. QC / pool RSD
    if unassigned:
        M["qc_rsd"] = {"status": "insufficient_metadata", "reason": f"{len(unassigned)} non-study sample(s) have no role"}
        f.status("computed", needs=["sample roles"])
        f.indicate("roles_needed", {"n": len(unassigned)}, n=len(unassigned))
    elif len(qc) >= 3:
        Q = Xa[:, qc]
        with np.errstate(invalid="ignore", divide="ignore"):
            rsd = np.nanstd(Q, axis=1, ddof=1) / np.nanmean(Q, axis=1) * 100
        rsd = np.where(np.isfinite(Q).sum(1) >= 3, rsd, np.nan)
        lv = params["rsd_levels"]
        ok = rsd[np.isfinite(rsd)]
        M["qc_rsd"] = {"n_qc": len(qc), "summary": stats.summary(rsd), "levels": lv,
                       "share_above": {str(x): float((ok > x).mean()) if ok.size else None for x in lv}}
        f.plot["qc_rsd"] = stats.histogram(rsd, 30)
        if ok.size:
            f.indicate("qc_rsd", M["qc_rsd"], n_qc=len(qc), median=round(float(np.median(ok)), 1),
                       hi=lv[-1], pct=round(100 * float((ok > lv[-1]).mean()), 1))
    else:
        M["qc_rsd"] = {"status": "insufficient_data", "reason": f"{len(qc)} QC/pool sample(s); needs at least 3"}
    # 3. blanks
    if blanks and study:
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.nanmean(Xa[:, blanks], axis=1) / np.nanmean(Xa[:, study], axis=1)
        M["blank_ratio"] = {"n_blanks": len(blanks), "summary": stats.summary(ratio)}
    else:
        M["blank_ratio"] = None

    # 4. run order
    B, G, _ = ctx.variables(unit)
    ro_col = next((c for c in ctx.M_columns if ctx.col_kind.get(c) == "run_order"), None)
    Y = unit.Y
    med_y = np.nanmedian(Y, axis=0)
    if ro_col:
        rv = ctx.column(ro_col)
        x = np.array([float(rv.get(s)) if is_num(rv.get(s)) else np.nan for s in unit.samples])
        ok = np.isfinite(x) & np.isfinite(med_y)
        drift = {"variable": ro_col, "n_used": int(ok.sum())}
        if ok.sum() >= 4:
            rho, _ = stats.spearman(x[ok], med_y[ok])
            order = np.argsort(x[ok], kind="mergesort")
            fit = lowess(x[ok][order], med_y[ok][order])
            drift.update(spearman_median_y=rho, ols_slope=stats.ols_slope(x[ok], med_y[ok]),
                         lowess_range=float(fit.max() - fit.min()),
                         lowess=[[float(a), float(b)] for a, b in zip(x[ok][order], fit)])
            f.plot["run_order_points"] = [[float(x[j]), float(med_y[j]), unit.samples[j]] for j in np.where(ok)[0]]
            Yc = stats.complete_features(Y)
            if Yc.shape[0] >= 2:
                sc, _ = stats.pca(Yc, 1)
                drift["spearman_pc1"] = stats.spearman(x[ok], sc[ok, 0])[0]
            if abs(rho) >= 0.3:
                f.indicate("run_order_drift", {"rho": rho, "variable": ro_col}, variable=ro_col, rho=round(rho, 3))
        # QC per-feature trend
        if len(qc) >= 4 and not unassigned:
            qx = np.array([float(rv.get(all_s[j])) if is_num(rv.get(all_s[j])) else np.nan for j in qc])
            if np.isfinite(qx).sum() >= 4:
                Q = unit.diagnostic(Xa)[:, qc]
                rhos = np.array([stats.spearman(qx, Q[i])[0] for i in range(Q.shape[0])])
                drift["qc_feature_trend"] = {"summary": stats.summary(rhos),
                                             "share_abs_rho_gt_0_5": float((np.abs(rhos[np.isfinite(rhos)]) > 0.5).mean())}
        M["run_order"] = drift
    else:
        M["run_order"] = None
        f.status("computed", needs=["run order"])

    # 5. per-sample quality
    Yc = stats.complete_features(Y)
    prof = np.median(Yc, axis=1) if Yc.shape[0] else None
    tied, _, _, S = floor_ties(unit.X, params["floor_min_observed"])
    det = (np.isfinite(unit.X) & ~S).sum(0)
    M["per_sample"] = [{"sample": s, "median_y": float(med_y[j]), "n_detected": int(det[j]),
                        "r_median_profile": stats.pearson(Yc[:, j], prof) if prof is not None and Yc.shape[0] >= 3 else None}
                       for j, s in enumerate(unit.samples)]

    # 6. rows marked as suspect in Step 0
    sus_cols = [a for a in ctx.schema.get("feature_annotations", []) if a.get("marks_rows_as_suspect") and a.get("keep", True)]
    if sus_cols:
        flagged = np.zeros(unit.p, dtype=bool)
        for a in sus_cols:
            vals = set(a.get("flagged_values") or [])
            flagged |= np.array([(row.get(a["column"]) or "") in vals for row in unit.F])
        R = ~np.isfinite(unit.X)

        def grp(m):
            if not m.any():
                return None
            return {"n_features": int(m.sum()), "median_y": float(np.nanmedian(Y[m])), "missing_rate": float(R[m].mean()),
                    "floor_rate": float(S[m].mean())}
        M["suspect_rows"] = {"columns": [a["column"] for a in sus_cols], "flagged": grp(flagged), "unflagged": grp(~flagged)}
        if flagged.any():
            f.indicate("suspect_rows", M["suspect_rows"]["flagged"], n=int(flagged.sum()),
                       cols=", ".join(a["column"] for a in sus_cols))
    else:
        M["suspect_rows"] = None

    # 7. kept vendor per-feature QC columns: distributions only
    vendor = []
    for a in ctx.schema.get("feature_annotations", []):
        c = a.get("column")
        if not a.get("keep", True) or not c or not (QC_NAME.search(c) or a.get("family") in ("qc", "quality")):
            continue
        vals = [row.get(c, "") for row in unit.F]
        nums = np.array([float(v) for v in vals if is_num(v)])
        if nums.size:
            vendor.append({"column": c, "summary": stats.summary(nums)})
        else:
            lv = {}
            for v in vals:
                if v:
                    lv[v] = lv.get(v, 0) + 1
            vendor.append({"column": c, "levels": dict(sorted(lv.items(), key=lambda kv: -kv[1])[:20])})
    M["vendor_qc_columns"] = vendor
    f.method("QC RSD, blank ratio, run-order drift (Spearman, LOWESS, PC1), per-sample quality, suspect rows",
             n_used=unit.n, transform=unit.transform, rsd_levels=params["rsd_levels"])
    return f
