"""G4 / A3: distribution shape, the variance-mean relationship, and how sample medians differ.
No normality tests. Runs on the audit set; the diagnostic copy is log2 only for linear data (A2).

 1. Does the spread grow with the level? Per protein, log SD against log mean on the values as they
    are, OLS slope b with a seeded bootstrap CI over proteins. b near 1 means SD proportional to the
    mean (multiplicative noise, typical of raw intensities): evidence for a log transform (G5).
    b near 0 means the spread does not depend on the level.
 2. On the diagnostic copy: Spearman rho between protein mean and SD (|rho| >= 0.3 is reported).
 3. Skewness per protein, on the values and on the diagnostic copy.
 4. Relative log expression (RLE): each value minus its protein's median; per sample, the median and
    IQR of those differences. Sample medians that differ point to normalization (S1).
 5. The S1 decision node median_shift_technical: does the spread of sample medians follow a technical
    variable (batch, run order) or a design variable (group, time)? Restricted permutations, BH.
"""

from __future__ import annotations

import numpy as np

from . import stats
from .finding import Finding
from .scale import skew_rows


def _med(x):
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else None


def _share(x):
    x = x[np.isfinite(x)]
    return float((np.abs(x) > 1).mean()) if x.size else None


def run(audit) -> Finding:
    p = audit.params
    f = Finding("A3", "Distribution and variance-mean", step="G4", feeds=["G5 log transform", "S1 normalization"],
                method="OLS of log SD on log mean (seeded bootstrap CI), Spearman on the diagnostic copy, skewness, "
                       "relative log expression, restricted-permutation associations (BH)")
    variables = audit.variables()
    subj = variables.get("subject", (None, None))[1]
    for bid in audit.schema.block_map():
        Xd = audit.audit_matrix(bid)
        X = Xd.to_numpy(dtype=float)
        Yd, info = audit.diagnostic(bid)
        Y = Yd.to_numpy()
        m = {"diagnostic_copy": info}
        with np.errstate(invalid="ignore"):
            mx, sx = np.nanmean(X, axis=1), np.nanstd(X, axis=1, ddof=1)
            my, sy = np.nanmean(Y, axis=1), np.nanstd(Y, axis=1, ddof=1)

        ok = np.isfinite(mx) & np.isfinite(sx) & (mx > 0) & (sx > 0)
        if ok.sum() >= 3:
            lm, ls = np.log(mx[ok]), np.log(sx[ok])
            b = stats.ols_slope(lm, ls)
            rng = np.random.default_rng([p["seed"], 3])
            boots = np.array([stats.ols_slope(lm[i], ls[i])
                              for i in (rng.integers(0, len(lm), len(lm)) for _ in range(p["bootstrap"]))])
            boots = boots[np.isfinite(boots)]
            ci = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
            m["mean_sd"] = {"slope": b, "ci95": ci, "n_proteins": int(ok.sum()), "bootstrap": p["bootstrap"]}
            audit.results.setdefault("a3_slope", {})[bid] = {"slope": b, "ci95": ci}
            if ci[0] > 0.5:
                f.say("note", f"{bid}: the spread grows with the level (slope of log SD on log mean b = {b:.2f}, 95% CI "
                              f"{ci[0]:.2f} to {ci[1]:.2f}): consistent with multiplicative noise, as in raw intensities.")
            else:
                f.say("ok", f"{bid}: slope of log SD on log mean b = {b:.2f} (95% CI {ci[0]:.2f} to {ci[1]:.2f}).")
        else:
            m["mean_sd"] = {"status": "insufficient_data", "reason": "fewer than 3 proteins with mean > 0 and SD > 0"}

        rho, n_used = stats.spearman(my, sy)
        m["mean_sd_diagnostic"] = {"spearman_rho": rho, "n_proteins": n_used}
        if rho == rho and abs(rho) >= 0.3:
            f.say("note", f"{bid}: on the diagnostic copy ({info['transform']}), protein SD still follows the mean "
                          f"(Spearman rho {rho:.2f}).")

        kx, ky = skew_rows(X), skew_rows(Y)
        m["skewness"] = {"values": {"median": _med(kx), "share_abs_gt_1": _share(kx)},
                         "diagnostic": {"median": _med(ky), "share_abs_gt_1": _share(ky)}}

        with np.errstate(invalid="ignore"):
            resid = Y - np.nanmedian(Y, axis=1, keepdims=True)
        n = Y.shape[1]
        smed = np.array([np.nanmedian(resid[:, j]) if np.isfinite(resid[:, j]).any() else np.nan for j in range(n)])
        siqr = np.array([np.subtract(*np.nanpercentile(resid[:, j], [75, 25])) if np.isfinite(resid[:, j]).any()
                         else np.nan for j in range(n)])
        mad_med = float(np.nanmedian(np.abs(smed - np.nanmedian(smed)))) if np.isfinite(smed).any() else None
        m["rle"] = {"per_sample": [{"sample": s, "median": float(smed[j]), "iqr": float(siqr[j])}
                                   for j, s in enumerate(Xd.columns)],
                    "mad_of_sample_medians": mad_med}

        rows = []
        subject = None if subj is None else subj.reindex(Xd.columns).fillna("").astype(str).tolist()
        for name, (kind, series) in variables.items():
            if kind not in ("batch", "run_order", "group", "timepoint", "covariate"):
                continue
            vals = series.reindex(Xd.columns).fillna("").astype(str).str.strip().tolist()
            numeric = kind in ("run_order", "covariate") and _numeric(vals)
            r = stats.association(smed, vals, numeric, subject, p, f"A3|{bid}|median|{name}")
            rows.append(dict(r, variable=name, audit_kind=kind, technical=kind in ("batch", "run_order")))
        for r, q in zip(rows, stats.bh([r.get("p") if r["status"] == "computed" else None for r in rows])):
            if q is not None:
                r["q"] = q
        differ = mad_med is not None and mad_med > p["sample_median_spread_mad"]
        m["median_shift"] = {"mad_of_sample_medians": mad_med, "cut": p["sample_median_spread_mad"],
                             "medians_differ": differ, "associations": rows, "evidence_level": "heuristic"}
        tech = [r["variable"] for r in rows if r.get("q", 1) <= 0.05 and r["technical"]]
        design = [r["variable"] for r in rows if r.get("q", 1) <= 0.05 and not r["technical"]]
        m["median_shift"]["follows_technical"], m["median_shift"]["follows_design"] = tech, design
        if differ:
            f.say("note", f"{bid}: sample medians differ (MAD {mad_med:.3f} on the diagnostic scale, above "
                          f"{p['sample_median_spread_mad']}, heuristic)."
                          + (f" The shift follows technical variable(s) {', '.join(tech)}." if tech else "")
                          + (f" It also follows design variable(s) {', '.join(design)}: normalization could remove "
                             "biology." if design else ""))
        else:
            f.say("ok", f"{bid}: sample medians are close (MAD {mad_med:.3f} on the diagnostic scale)."
                  if mad_med is not None else f"{bid}: sample medians could not be computed.")
        f.measures[bid] = m
    return f


def _numeric(vals: list[str]) -> bool:
    v = [x for x in vals if x != ""]
    try:
        [float(x) for x in v]
        return bool(v)
    except ValueError:
        return False
