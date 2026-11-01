"""G4 / A6: technical noise, QC samples and per-sample quality.

Sample types come from step 7 (study, qc, blank, pool, calibrator, reference). Proteins: unflagged,
except in item 6, which compares flagged with unflagged proteins.

 1. QC repeatability: per protein, the RSD (SD / mean x 100) over the QC and pool samples (>= 3),
    on linear values; shares above 20% and 30% (heuristic levels). On log-scale data the RSD is not
    meaningful, so the SD on the log scale is reported instead.
 2. Blanks: per protein, mean blank signal / mean study signal (linear values only).
 3. Run order (when a run_order variable exists): each study sample's median (diagnostic scale)
    against run order: Spearman rho, OLS slope, the range of a LOWESS fit, and rho of PC1 with run
    order (PC1 on the diagnostic matrix, policy reported). |rho| >= 0.3 (heuristic) is reported as a
    possible drift. With >= 4 QC samples: per protein, Spearman rho of QC values against run order;
    the share with |rho| > 0.5.
 4. Per study sample: median, number of detected values (neither missing nor at a floor), Pearson r
    with the median profile over complete proteins (a low r marks an unusual sample).
 5. Vendor per-protein QC columns (names like CV, LoD, QC, ColCheck): their distributions only.
 6. Flagged proteins (step 6) against the others: count, median, missing rate.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import diagnostic as dmx
from . import stats
from .finding import Finding
from .missingness import floor_cells
from .scale import diagnostic_y

QC_NAME = re.compile(r"(^|[^a-z])(qc|cvs?|rsd|lodb?|loq|signal.?to.?noise|s/?n|colcheck|rowcheck)([^a-z]|$)", re.I)


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


def _num(series: pd.Series, names) -> np.ndarray:
    return pd.to_numeric(series.reindex(names), errors="coerce").to_numpy(dtype=float)


def run(audit) -> Finding:
    p = audit.params
    s = audit.schema
    f = Finding("A6", "Noise, QC and sample quality", step="G4",
                feeds=["S1 normalization", "G6 drift correction", "S3 feature filtering", "S4 sample exclusion"],
                method="QC RSD, blank ratio, run-order drift (Spearman, OLS, LOWESS, PC1), per-sample quality, vendor "
                       "QC columns, flagged against unflagged proteins")
    types = s.samples().set_index("sample")["sample_type"]
    flagged_ids = set(s.feature_metadata().query("flagged")["feature"])
    variables = audit.variables()
    run_order = next(((n, v) for n, (k, v) in variables.items() if k == "run_order"), None)
    for bid in s.block_map():
        cls = audit.results["scale"][bid]
        linear = cls == "continuous_linear"
        full = audit.full_matrix(bid)
        A = full.loc[~full.index.isin(flagged_ids)]
        m = {"sample_types": types.reindex(A.columns).value_counts().to_dict()}
        qc = [c for c in A.columns if types.get(c) in ("qc", "pool")]
        study = [c for c in A.columns if types.get(c) == "study"]
        blanks = [c for c in A.columns if types.get(c) == "blank"]

        # 1. QC repeatability
        if len(qc) >= 3:
            Q = A[qc].to_numpy(dtype=float)
            enough = np.isfinite(Q).sum(1) >= 3
            if linear:
                with np.errstate(invalid="ignore", divide="ignore"):
                    rsd = np.nanstd(Q, axis=1, ddof=1) / np.nanmean(Q, axis=1) * 100
                rsd = np.where(enough, rsd, np.nan)
                ok = rsd[np.isfinite(rsd)]
                lv = p["rsd_levels"]
                m["qc_rsd"] = {"n_qc": len(qc), "summary": stats.summary(rsd), "levels": lv,
                               "share_above": {str(x): float((ok > x).mean()) if ok.size else None for x in lv}}
                if ok.size:
                    f.say("note" if (ok > lv[-1]).mean() > 0.5 else "ok",
                          f"{bid}: QC repeatability over {len(qc)} QC / pool samples: median RSD {np.median(ok):.1f}%; "
                          f"{100 * (ok > lv[0]).mean():.1f}% of proteins above {lv[0]}%, "
                          f"{100 * (ok > lv[-1]).mean():.1f}% above {lv[-1]}% (heuristic levels).")
            else:
                with np.errstate(invalid="ignore"):
                    sd = np.where(enough, np.nanstd(Q, axis=1, ddof=1), np.nan)
                m["qc_sd_log_scale"] = {"n_qc": len(qc), "summary": stats.summary(sd)}
                f.say("ok", f"{bid}: values are on a log-like scale ({cls}), so QC repeatability is the SD on that scale "
                            f"(median {np.nanmedian(sd):.3f} over {len(qc)} QC / pool samples), not an RSD.")
        else:
            m["qc_rsd"] = {"status": "insufficient_data", "reason": f"{len(qc)} QC / pool sample(s); needs at least 3"}
            f.say("not_applicable", f"{bid}: {len(qc)} QC / pool sample(s): QC repeatability needs at least 3.")

        # 2. blanks
        if blanks and study and linear:
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.nanmean(A[blanks].to_numpy(float), axis=1) / np.nanmean(A[study].to_numpy(float), axis=1)
            m["blank_ratio"] = {"n_blanks": len(blanks), "summary": stats.summary(ratio)}
            audit.results.setdefault("blank_ratio", {})[bid] = pd.Series(ratio, index=A.index)

        # 3. run order
        Yd, info = audit.diagnostic(bid)
        Y = Yd.to_numpy()
        med_y = np.nanmedian(Y, axis=0)
        if run_order:
            name, series = run_order
            x = _num(series, Yd.columns)
            ok = np.isfinite(x) & np.isfinite(med_y)
            drift = {"variable": name, "n_used": int(ok.sum())}
            if ok.sum() >= 4:
                rho, _ = stats.spearman(x[ok], med_y[ok])
                order = np.argsort(x[ok], kind="mergesort")
                fit = lowess(x[ok][order], med_y[ok][order])
                drift.update(spearman_median=rho, ols_slope=stats.ols_slope(x[ok], med_y[ok]),
                             lowess_range=float(fit.max() - fit.min()))
                dm = dmx.default_matrix(Y, p)
                drift["diagnostic_matrix"] = dm.report(list(Yd.columns))
                if dm.n_features >= 2 and len(dm.cols) >= 4:
                    sc, _ = stats.pca(dm.Y, 1)
                    okc = ok[dm.cols]
                    drift["spearman_pc1"] = stats.spearman(x[dm.cols][okc], sc[okc, 0])[0]
                if abs(rho) >= p["run_order_rho_indicator"]:
                    f.say("note", f"{bid}: sample medians change with '{name}' (Spearman rho {rho:.2f} over "
                                  f"{int(ok.sum())} study samples; LOWESS range {drift['lowess_range']:.3f}): a "
                                  "plausible sign of drift over the run.")
                else:
                    f.say("ok", f"{bid}: no clear trend of sample medians with '{name}' (Spearman rho {rho:.2f}).")
            qx = _num(series, qc) if qc else np.array([])
            if len(qc) >= 4 and np.isfinite(qx).sum() >= 4:
                Qy = diagnostic_y(A[qc].to_numpy(dtype=float), cls)[0]
                rhos = np.array([stats.spearman(qx, Qy[i])[0] for i in range(Qy.shape[0])])
                fin = rhos[np.isfinite(rhos)]
                share = float((np.abs(fin) > 0.5).mean()) if fin.size else None
                drift["qc_protein_trend"] = {"summary": stats.summary(rhos), "share_abs_rho_gt_0_5": share}
                if share is not None and share >= p["qc_drift_share"]:
                    f.say("note", f"{bid}: in the QC samples, {100 * share:.1f}% of proteins follow run order "
                                  "(|Spearman rho| > 0.5): a plausible sign of signal drift.")
            m["run_order"] = drift
        else:
            m["run_order"] = None
            f.say("not_applicable", f"{bid}: no run-order variable in the sample information, so drift over the run "
                                    "cannot be assessed.")

        # 4. per study sample
        Yc = stats.complete_features(Y)
        prof = np.median(Yc, axis=1) if Yc.shape[0] else None
        X = audit.audit_matrix(bid).to_numpy(dtype=float)
        _, S = floor_cells(X, p["floor_min_observed"])
        det = (np.isfinite(X) & ~S).sum(0)
        m["per_sample"] = [{"sample": c, "median": float(med_y[j]), "n_detected": int(det[j]),
                            "r_with_median_profile": stats.pearson(Yc[:, j], prof)
                            if prof is not None and Yc.shape[0] >= 3 else None}
                           for j, c in enumerate(Yd.columns)]
        rs = [r["r_with_median_profile"] for r in m["per_sample"] if r["r_with_median_profile"] is not None]
        if rs:
            worst = min(m["per_sample"], key=lambda r: r["r_with_median_profile"] if r["r_with_median_profile"]
                        is not None else 2)
            f.say("ok", f"{bid}: correlation of each study sample with the median profile ({Yc.shape[0]} complete "
                        f"proteins): median r {np.median(rs):.3f}, lowest {worst['r_with_median_profile']:.3f} "
                        f"({worst['sample']}).")

        # 6. flagged proteins against the others
        if flagged_ids & set(full.index):
            Fl = full.loc[full.index.isin(flagged_ids), study]
            Un = full.loc[~full.index.isin(flagged_ids), study]
            fy, uy = diagnostic_y(Fl.to_numpy(float), cls)[0], diagnostic_y(Un.to_numpy(float), cls)[0]
            m["flagged_proteins"] = {
                "flagged": {"n": int(Fl.shape[0]), "median": float(np.nanmedian(fy)) if np.isfinite(fy).any() else None,
                            "missing_rate": float(np.isnan(fy).mean())},
                "unflagged": {"n": int(Un.shape[0]), "median": float(np.nanmedian(uy)),
                              "missing_rate": float(np.isnan(uy).mean())}}
            fl, un = m["flagged_proteins"]["flagged"], m["flagged_proteins"]["unflagged"]
            f.say("ok", f"{bid}: {fl['n']} flagged proteins (left out of the audit set): median {fl['median']:.3g} vs "
                        f"{un['median']:.3g}, missing rate {100 * fl['missing_rate']:.1f}% vs "
                        f"{100 * un['missing_rate']:.1f}% for the others.")
        f.measures[bid] = m

    # 5. vendor per-protein QC columns
    vendor = []
    for g in s.groups_with("feature_annotation"):
        for c in g.columns:
            if not (QC_NAME.search(c) or (g.family or "").lower() in ("qc", "quality")):
                continue
            vals = pd.to_numeric(s.table.data[c].str.strip(), errors="coerce")
            if vals.notna().any():
                vendor.append({"column": c, "label": g.label, "summary": stats.summary(vals.dropna().to_numpy())})
            else:
                vendor.append({"column": c, "label": g.label,
                               "levels": s.table.data[c].str.strip().value_counts().head(20).to_dict()})
    f.measures["vendor_qc_columns"] = vendor
    if vendor:
        f.say("ok", f"Vendor per-protein QC columns found: {', '.join(v['column'] for v in vendor)} (distributions in "
                    "the measures).")
    return f
