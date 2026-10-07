"""A2. Data type and measurement scale (v3 §6.3). Always computed: it decides whether the
other audits apply (count-like, proportion-like and compositional data are out of Tier 1)."""

from __future__ import annotations

import numpy as np

from . import stats
from .context import skew_rows
from .finding import Finding

AUDIT_ID, CODE = "A2", "scale"

CLASS_TEXT = {
    "count_like": "non-negative integers (count-like)",
    "proportion_like": "values in [0, 1] (proportion-like)",
    "compositional_like": "constant sample sums (compositional-like)",
    "continuous_signed": "continuous values with negatives (already centred or log-ratio-like)",
    "continuous_linear": "continuous, right-skewed values that log2 makes more symmetric (linear scale)",
    "continuous_symmetric_or_log_like": "continuous values that log2 does not make more symmetric (symmetric or "
                                        "already log-like)",
}


def median_scaling(X, cut):
    """MAD of log2 per-feature medians (positive medians only). Below cut: features share a common
    median, consistent with per-feature scaling."""
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=1)
    med = med[np.isfinite(med) & (med > 0)]
    if med.size < 3:
        return None
    l = np.log2(med)
    mad = float(np.median(np.abs(l - np.median(l))))
    return {"mad_log2_feature_medians": mad, "median_of_feature_medians": float(np.median(med)),
            "n_features": int(med.size), "cut": cut, "signature": mad < cut}


def run(unit, ctx, params, results):
    f = Finding(AUDIT_ID, CODE, unit).feeds("transformation")
    X = unit.X
    M = f.measures
    ev = dict(unit.scale_evidence)
    obs = X[np.isfinite(X)]
    if not obs.size:
        return f.status("insufficient_data", "no observed values")
    M["classification"] = unit.scale
    M["tests"] = {k: ev.get(k) for k in ("all_non_negative", "all_integer", "all_in_0_1", "negative_present")}
    M["cv_sample_sums"] = ev.get("cv_sample_sums")
    M["constant_sum_signature"] = bool(ev.get("all_non_negative") and ev.get("cv_sample_sums") is not None
                                       and ev["cv_sample_sums"] <= 0.001)
    ms = median_scaling(X, params["median_scaling_mad"])
    M["median_scaling"] = ms
    M["skewness"] = {"median_x": ev.get("median_skew_x"), "median_y": ev.get("median_skew_y")}
    pos = obs[obs > 0]
    p1, p99 = (float(np.percentile(pos, 1)), float(np.percentile(pos, 99))) if pos.size else (None, None)
    M["range"] = {"min": float(obs.min()), "p1": p1, "median": float(np.median(obs)), "p99": p99, "max": float(obs.max()),
                  "log10_p99_over_p1": float(np.log10(p99 / p1)) if p1 and p99 else None,
                  "max_over_p99": float(obs.max() / p99) if p99 else None}
    f.indicate("scale_class", {"classification": unit.scale, **M["tests"], "median_skew_x": ev.get("median_skew_x"),
                               "median_skew_y": ev.get("median_skew_y")}, text=CLASS_TEXT.get(unit.scale, unit.scale))
    if unit.out_of_scope:
        f.indicate("out_of_scope_type", {"classification": unit.scale}, what=unit.out_of_scope)
    if ms and ms["signature"]:
        f.indicate("median_scaling_signature", ms, mad=round(ms["mad_log2_feature_medians"], 3), cut=ms["cut"],
                   median=round(ms["median_of_feature_medians"], 3))
    results.setdefault(unit.dataset_id, {}).setdefault(unit.unit_id, {})["scale"] = {
        "classification": unit.scale, "median_scaling": bool(ms and ms["signature"]), "negative": ev.get("negative_present")}
    f.plot["value_histogram_log10"] = stats.histogram(obs, 40, log10=True)
    f.plot["feature_skew_x"] = stats.histogram(skew_rows(X), 30)
    f.method("ordered classification from value tests, sample sums, per-feature skewness on X and Y "
             "(heuristic thresholds)", n_used=unit.n, transform=unit.transform,
             median_scaling_mad=params["median_scaling_mad"])
    return f
