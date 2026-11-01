"""G2 / A2: what kind of numbers are these? (measurement scale)

Runs on the audit set (study samples, unflagged proteins), block by block. The class is decided by
an ordered list of tests on the values; the first that applies wins:

    count_like          every value a non-negative whole number        (out of PRISM's scope)
    proportion_like     every value between 0 and 1                    (out of scope)
    compositional_like  every sample sums to the same total             (out of scope)
    continuous_signed   negative values present: already centred or on a log scale (e.g. Olink NPX)
    continuous_linear   right-skewed per protein, and log2 makes them more symmetric (raw intensities, RFU)
    continuous_symmetric_or_log_like   log2 does not make them more symmetric: symmetric or already log

"More symmetric" compares the median per-protein skewness on the values with the same after a
diagnostic log2 (log2(x), or log2(x + c) with c = half the smallest positive value when zeros
exist). The diagnostic copy is used only here and never written anywhere.

It also reports the median-scaling signature: when every protein's median is close to the same
value (MAD of log2 per-protein medians below 0.5, a heuristic), the proteins were probably scaled
one by one (e.g. each divided by its median).
"""

from __future__ import annotations

import numpy as np

from .finding import Finding

CLASS_TEXT = {
    "count_like": "non-negative whole numbers (count-like)",
    "proportion_like": "values between 0 and 1 (proportion-like)",
    "compositional_like": "every sample sums to the same total (compositional-like)",
    "continuous_signed": "continuous values with negatives (already centred or on a log scale)",
    "continuous_linear": "continuous, right-skewed values that log2 makes more symmetric (a linear scale)",
    "continuous_symmetric_or_log_like": "continuous values that log2 does not make more symmetric "
                                        "(symmetric, or already log-like)",
}
OUT_OF_SCOPE = ("count_like", "proportion_like", "compositional_like")


def skew_rows(A: np.ndarray) -> np.ndarray:
    """Skewness of every row, ignoring missing values (rows with < 3 values or no spread: NaN)."""
    ok = np.isfinite(A)
    n = ok.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(ok, A, 0).sum(1) / np.maximum(n, 1)
        d = np.where(ok, A - m[:, None], 0)
        m2 = (d ** 2).sum(1) / np.maximum(n, 1)
        m3 = (d ** 3).sum(1) / np.maximum(n, 1)
        s = m3 / m2 ** 1.5
    return np.where((n >= 3) & (m2 > 0), s, np.nan)


def diagnostic_log(X: np.ndarray) -> tuple[np.ndarray, dict]:
    """log2(X), or log2(X + c) with c = half the smallest positive value; signed data unchanged."""
    obs = X[np.isfinite(X)]
    if obs.size and (obs < 0).any():
        return X.copy(), {"transform": "none", "reason": "negative values present"}
    if obs.size and (obs > 0).all():
        return np.log2(X), {"transform": "log2"}
    pos = obs[obs > 0]
    c = 0.5 * float(pos.min()) if pos.size else 1.0
    return np.log2(X + c), {"transform": "log2(x + c)", "c": c}


def diagnostic_y(X: np.ndarray, scale_class: str) -> tuple[np.ndarray, dict]:
    """The copy diagnostics run on: log2 (see diagnostic_log) only when A2 found a linear scale;
    log-like and signed values are used as they are (taking log2 of log values would be a mistake)."""
    if scale_class == "continuous_linear":
        return diagnostic_log(X)
    return X.copy(), {"transform": "none", "reason": f"A2 class {scale_class}: used as it is"}


def classify(X: np.ndarray, params: dict) -> tuple[str, dict]:
    obs = X[np.isfinite(X)]
    ev = {"n_observed": int(obs.size)}
    if not obs.size:
        return "unknown", ev
    ev["all_non_negative"] = bool((obs >= 0).all())
    ev["all_integer"] = bool(np.all(obs == np.round(obs)))
    ev["all_in_0_1"] = bool(((obs >= 0) & (obs <= 1)).all())
    sums = np.nansum(X, axis=0)
    ev["cv_sample_sums"] = float(np.std(sums) / abs(np.mean(sums))) if np.mean(sums) else None
    ev["negative_present"] = bool((obs < 0).any())
    sx = skew_rows(X)
    ev["median_skew_values"] = float(np.nanmedian(sx)) if np.isfinite(sx).any() else None
    ev["median_skew_after_log2"] = None
    if ev["all_non_negative"]:
        sy = skew_rows(diagnostic_log(X)[0])
        ev["median_skew_after_log2"] = float(np.nanmedian(sy)) if np.isfinite(sy).any() else None
    if ev["all_non_negative"] and ev["all_integer"]:
        return "count_like", ev
    if ev["all_in_0_1"]:
        return "proportion_like", ev
    if ev["all_non_negative"] and ev["cv_sample_sums"] is not None and ev["cv_sample_sums"] <= params["compositional_cv"]:
        return "compositional_like", ev
    if ev["negative_present"]:
        return "continuous_signed", ev
    sx_, sy_ = ev["median_skew_values"], ev["median_skew_after_log2"]
    if sx_ is not None and sy_ is not None and abs(sy_) < abs(sx_):
        return "continuous_linear", ev
    return "continuous_symmetric_or_log_like", ev


def median_scaling(X: np.ndarray, cut: float) -> dict | None:
    """MAD of log2 per-protein medians (positive medians only); below cut: a shared median."""
    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # proteins with no value give NaN, as intended
        med = np.nanmedian(X, axis=1)
    med = med[np.isfinite(med) & (med > 0)]
    if med.size < 3:
        return None
    lm = np.log2(med)
    mad = float(np.median(np.abs(lm - np.median(lm))))
    return {"mad_log2_protein_medians": round(mad, 4), "median_of_protein_medians": float(np.median(med)),
            "n_proteins": int(med.size), "cut": cut, "signature": mad < cut}


def run(audit) -> Finding:
    f = Finding("A2", "Measurement scale", step="G2", feeds=["G5 log transform", "A13 processing state"],
                method="ordered tests on the values (audit set); per-protein skewness on the values and after a "
                       "diagnostic log2; thresholds marked heuristic in parameters.yaml")
    p = audit.params
    for bid in audit.schema.block_map():
        X = audit.audit_matrix(bid).to_numpy(dtype=float)
        cls, ev = classify(X, p)
        ms = median_scaling(X, p["median_scaling_mad"])
        obs = X[np.isfinite(X)]
        pos = obs[obs > 0]
        rng = {}
        if obs.size:
            rng = {"min": float(obs.min()), "median": float(np.median(obs)), "max": float(obs.max())}
            if pos.size:
                p1, p99 = np.percentile(pos, [1, 99])
                rng["log10_p99_over_p1"] = round(float(np.log10(p99 / p1)), 3)
        f.measures[bid] = {"class": cls, "tests": ev, "range": rng, "median_scaling": ms}
        audit.results.setdefault("scale", {})[bid] = cls
        sx, sy = ev.get("median_skew_values"), ev.get("median_skew_after_log2")
        skew = f" (median per-protein skewness {sx:.2f} on the values" + (f", {sy:.2f} after log2)" if sy is not None
                                                                           else ")") if sx is not None else ""
        f.say("ok", f"{bid}: {CLASS_TEXT.get(cls, cls)}{skew}. Values from {rng.get('min', 0):.3g} to "
                    f"{rng.get('max', 0):.3g}, median {rng.get('median', 0):.3g}.")
        if cls in OUT_OF_SCOPE:
            f.say("note", f"{bid}: {cls.replace('_', '-')} data are outside what PRISM's later steps handle; their "
                          "audits will be not applicable.")
        if ms and ms["signature"]:
            f.say("note", f"{bid}: every protein has nearly the same median (MAD of log2 medians "
                          f"{ms['mad_log2_protein_medians']:.3g}, below {ms['cut']}): consistent with proteins "
                          "scaled one by one before upload.")
    return f
