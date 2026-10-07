"""A3. Distribution shape and variance-mean relationship (v3 §6.3). No normality tests."""

from __future__ import annotations

import numpy as np

from . import stats
from .context import skew_rows
from .finding import Finding

AUDIT_ID, CODE = "A3", "distribution"
MAX_POINTS = 3000


def run(unit, ctx, params, results):
    f = Finding(AUDIT_ID, CODE, unit).feeds("transformation", "normalization")
    if unit.out_of_scope:
        return f.status("not_applicable", f"Tier 1 does not audit {unit.out_of_scope} ({unit.scale}).")
    X, Y = unit.X, unit.Y
    M = f.measures
    with np.errstate(invalid="ignore"):
        mx = np.nanmean(X, axis=1)
        sx = np.nanstd(X, axis=1, ddof=1)
        my = np.nanmean(Y, axis=1)
        sy = np.nanstd(Y, axis=1, ddof=1)
    # 1. log SD on log mean on X, bootstrap CI over features
    ok = np.isfinite(mx) & np.isfinite(sx) & (mx > 0) & (sx > 0)
    if ok.sum() >= 3:
        lm, ls = np.log(mx[ok]), np.log(sx[ok])
        b = stats.ols_slope(lm, ls)
        rng = np.random.default_rng([params["seed"], 3])
        boots = []
        for _ in range(params["bootstrap"]):
            i = rng.integers(0, len(lm), len(lm))
            boots.append(stats.ols_slope(lm[i], ls[i]))
        boots = np.array(boots)
        boots = boots[np.isfinite(boots)]
        M["mean_sd_x"] = {"slope": b, "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                          "n_features": int(ok.sum()), "excluded": int((~ok).sum()), "bootstrap": params["bootstrap"]}
        if M["mean_sd_x"]["ci95"][0] > 0.5:
            f.indicate("sd_grows_with_mean", M["mean_sd_x"], b=round(b, 3), lo=round(M["mean_sd_x"]["ci95"][0], 3),
                       hi=round(M["mean_sd_x"]["ci95"][1], 3))
    else:
        M["mean_sd_x"] = {"status": "insufficient_data", "reason": "fewer than 3 features with mean > 0 and SD > 0"}
    # 2. on Y
    rho, nu = stats.spearman(my, sy)
    M["mean_sd_y"] = {"spearman_rho": rho, "n_features": nu}
    if rho == rho and abs(rho) >= 0.3:
        f.indicate("mean_sd_after_transform", {"rho": rho, "n_features": nu}, rho=round(rho, 3),
                   transform=unit.transform["transform"])
    # 3. skewness
    kx, ky = skew_rows(X), skew_rows(Y)
    M["skewness"] = {"x": {"median": _med(kx), "share_abs_gt_1": _share(kx)},
                     "y": {"median": _med(ky), "share_abs_gt_1": _share(ky)}}
    # 4. relative log expression on Y
    with np.errstate(invalid="ignore"):
        resid = Y - np.nanmedian(Y, axis=1, keepdims=True)
    smed = np.array([np.nanmedian(resid[:, j]) if np.isfinite(resid[:, j]).any() else np.nan for j in range(unit.n)])
    siqr = np.array([np.subtract(*np.nanpercentile(resid[:, j], [75, 25])) if np.isfinite(resid[:, j]).any() else np.nan
                     for j in range(unit.n)])
    mad_med = float(np.nanmedian(np.abs(smed - np.nanmedian(smed))))
    M["rle"] = {"sample_medians": stats.summary(smed), "sample_iqr": stats.summary(siqr),
                "mad_of_sample_medians": mad_med}
    f.plot["rle"] = [{"sample": s, "median": float(smed[j]), "iqr": float(siqr[j])} for j, s in enumerate(unit.samples)]
    idx = np.where(ok)[0]
    if len(idx) > MAX_POINTS:
        idx = idx[np.linspace(0, len(idx) - 1, MAX_POINTS).astype(int)]
    f.plot["mean_sd_x"] = [[float(mx[i]), float(sx[i])] for i in idx]
    f.method("OLS of log SD on log mean (seeded bootstrap CI), Spearman on Y, skewness, relative log expression",
             n_used=unit.n, transform=unit.transform, bootstrap=params["bootstrap"], seed=params["seed"])
    return f


def _med(x):
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else None


def _share(x):
    x = x[np.isfinite(x)]
    return float((np.abs(x) > 1).mean()) if x.size else None
