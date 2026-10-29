"""A12. Missingness profile (Build Request v4, section 5): detection and description, never imputation.

Hard rule: nothing here writes a filled value anywhere (state, diagnostic copy or plot). Floor-tied
values count as observed numbers unless a variant says otherwise.

    12.1 cell encoding        observed / nan / zero / floor_tied, by assay and stratum; the zeros question
    12.2 rates and structure  per feature, sample, batch, group, time; pattern summaries
    12.3 batch-wide absence   features with no observed value in a batch level with >= 3 samples
    12.4 detection curve      zero-truncated binomial logit P(detect) = b0 + b1 * ybar (descriptive)
    12.5 design association   per-sample NaN rate vs B and G; per-feature NaN indicator vs batch (BH)
    12.6 feature classes      batch_wide_absent > low_abundance_pattern > scattered; complete
    12.7 pattern summary      pattern_class and three-valued indicators; the wording rule
    12.8 feasible set         imputation_candidates from imputation_catalog.yaml (no ranking)

The words that name a missingness mechanism are never used as a diagnosis: the pattern is
quantified and plausible mechanisms are assessed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import stats
from .a04_missingness import floor_ties
from .finding import Finding, association, table

AUDIT_ID, CODE = "A12", "missingness_profile"
WORDING = "Pattern quantified, plausible mechanisms assessed."
MOLENBERGHS = ("Models in which values are missing at random and models in which they are not can fit the observed "
               "data equally well (Molenberghs, Beunckens, Sotto and Kenward 2008), so the observed data cannot settle "
               "the mechanism.")
LIMITATION = ("PRISM's detection curve is a simplified reference implementation: ybar is the mean of the detected "
              "values only, so it is biased upward for features with censoring. The published method (Li and Smyth "
              "2023, 'Neither random nor censored') uses a more careful likelihood with moderated variances. Read b1 "
              "as a descriptive measure.")
PATTERN_RULES = [
    "none: the matrix has no missing values",
    "mixed: intensity dependence = yes and batch association = yes",
    "mostly_intensity_dependent: intensity dependence = yes, batch association not yes",
    "batch_associated: batch association = yes, intensity dependence not yes",
    "insufficient_data: otherwise (neither indicator is yes, or neither could be assessed)",
    "intensity dependence: yes when the 95% CI of b1 excludes 0; no when it lies inside +-a12_b1_negligible; "
    "indeterminate otherwise (including a CI that straddles a threshold) or when the curve is not computed",
    "batch association: yes when any feature is BH-significant (q <= 0.05) or the batch-wide absence share is above "
    "a12_batch_absence_share; no when tested and neither holds; indeterminate without a batch candidate",
    "floor pattern: yes when the floor-tied feature share reaches floor_tie_share_indicator",
]


# ------------------------------------------------------------------ 12.4 the detection curve


def _expit(x):
    return 0.5 * (1 + np.tanh(0.5 * x))


def ztb_negll(beta, x, d, n):
    """Negative log-likelihood (and gradient) of the zero-truncated binomial with logit p = b0 + b1 x."""
    eta = beta[0] + beta[1] * x
    p = _expit(eta)
    log1mp = -np.logaddexp(0, eta)                # log(1 - p)
    logp = -np.logaddexp(0, -eta)                 # log p
    qn = np.exp(n * log1mp)                       # (1 - p)^n
    log_trunc = np.log(-np.expm1(n * log1mp))     # log(1 - (1 - p)^n)
    ll = d * logp + (n - d) * log1mp - log_trunc
    g_eta = d - n * p - n * p * qn / np.maximum(1 - qn, 1e-300)
    return -float(ll.sum()), -np.array([g_eta.sum(), (g_eta * x).sum()])


def fit_detection(ybar, d, n):
    from scipy.optimize import minimize
    mu = float(np.mean(ybar))
    xc = ybar - mu
    p0 = np.clip(d.mean() / n, 0.05, 0.95)
    res = minimize(ztb_negll, np.array([np.log(p0 / (1 - p0)), 0.0]), args=(xc, d, n), jac=True, method="BFGS")
    c0, b1 = float(res.x[0]), float(res.x[1])
    return c0 - b1 * mu, b1, bool(res.success)


def detection_curve(Y, params):
    """-> dict with status computed / insufficient_data / not_applicable."""
    ok = np.isfinite(Y)
    n = Y.shape[1]
    d = ok.sum(1)
    if not (~ok).any():
        return {"status": "not_applicable", "reason": "no missing values: the floor-based abundance check (A4 item 4) "
                                                      "is the substitute"}
    partial = int(((d > 0) & (d < n)).sum())
    if partial < params["a12_min_partial_features"]:
        return {"status": "insufficient_data", "reason": f"{partial} feature(s) with 0 < detected < n; needs "
                                                         f"{params['a12_min_partial_features']}", "n_partial": partial}
    use = d >= 1
    with np.errstate(invalid="ignore"):
        ybar = np.where(use, np.where(ok, Y, 0).sum(1) / np.maximum(d, 1), np.nan)[use]
    du = d[use].astype(float)
    b0, b1, conv = fit_detection(ybar, du, n)
    rng = np.random.default_rng([params["seed"], 12])
    boots = []
    for _ in range(params["a12_bootstrap"]):
        i = rng.integers(0, len(ybar), len(ybar))
        boots.append(fit_detection(ybar[i], du[i], n)[1])
    boots = np.array(boots)
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    q10, q90 = float(np.quantile(ybar, 0.1)), float(np.quantile(ybar, 0.9))
    p10, p90 = float(_expit(b0 + b1 * q10)), float(_expit(b0 + b1 * q90))
    grid = np.linspace(float(ybar.min()), float(ybar.max()), 40)
    bins = np.quantile(ybar, np.linspace(0, 1, 21))
    pts = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (ybar >= a) & (ybar <= b)
        if m.any():
            pts.append([float(ybar[m].mean()), float((du[m] / n).mean()), int(m.sum())])
    return {"status": "computed", "model": "zero-truncated binomial, logit P = b0 + b1 * ybar, maximum likelihood "
                                           "(scipy.optimize BFGS), features with d >= 1",
            "b0": b0, "b1": b1, "b1_ci95": [lo, hi], "converged": conv, "bootstrap": params["a12_bootstrap"],
            "n_features_used": int(use.sum()), "n_partial": partial, "n_samples": n,
            "ybar_p10": q10, "ybar_p90": q90, "p_detect_at_p10": p10, "p_detect_at_p90": p90,
            "plateau_missing_share": 1 - p90, "limitation": LIMITATION,
            "_plot": {"binned": pts, "curve": [[float(x), float(_expit(b0 + b1 * x))] for x in grid]}}


# ------------------------------------------------------------------ 12.5 per-feature NaN indicator vs batch


def nan_batch_tests(R, var, subj, params, key):
    """Chi-square statistic of the 2 x k (NaN / observed by batch level) table per feature, with a
    restricted-permutation p (covers small cells), BH over features. R: NaN indicator (p x n)."""
    present = var.present()
    lev = var.levels()
    if len(lev) < 2:
        return None
    idx = np.where(present)[0]
    codes = np.array([lev.index(var.values[j]) for j in idx])
    Rs = R[:, idx].astype(float)
    m_tot = Rs.sum(1)
    test = (m_tot > 0) & (m_tot < len(idx))
    if not test.any():
        return {"variable": var.name, "n_tested": 0, "n_significant": 0}
    Rt = Rs[test]
    k = len(lev)
    n = len(idx)

    def chi2(cd):
        oh = np.zeros((n, k))
        oh[np.arange(n), cd] = 1
        nl = oh.sum(0)
        m = Rt @ oh
        E = Rt.sum(1, keepdims=True) * nl[None, :] / n
        E2 = (n - Rt.sum(1, keepdims=True)) * nl[None, :] / n
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nansum((m - E) ** 2 * (1 / np.where(E > 0, E, np.nan) + 1 / np.where(E2 > 0, E2, np.nan)), axis=1)

    obs = chi2(codes)
    sub = None if subj is None else [subj[j] for j in idx]
    perm = stats.Permuter(codes, sub, params["a12_feature_permutations"], 0, params["seed"], key)
    P = perm.matrix()
    ge = np.zeros(len(obs))
    for row in P:
        ge += chi2(row) >= obs - 1e-12 * np.maximum(1.0, np.abs(obs))
    pv = (1 + ge) / (1 + len(P))
    qv = stats.bh(pv)
    q = np.array([x if x is not None else np.nan for x in qv], float)
    sig = q <= 0.05
    return {"variable": var.name, "levels": lev, "n_tested": int(test.sum()), "n_significant": int(np.nansum(sig)),
            "statistic": "chi-square (2 x k)", "p_min_attainable": 1.0 / (1 + len(P)), **perm.describe(),
            "_sig_mask": (test, sig)}


# ------------------------------------------------------------------ the audit


def _catalog():
    import yaml
    return yaml.safe_load((Path(__file__).resolve().parent / "imputation_catalog.yaml").read_text(encoding="utf-8"))["families"]


def _fits(cond, profile):
    if cond.get("always"):
        return True
    if "pattern_class_in" in cond:
        return profile["pattern_class"] in cond["pattern_class_in"]
    if "scattered_majority" in cond:
        return bool(profile["scattered_majority"]) == bool(cond["scattered_majority"])
    if "batch_wide_absent_any" in cond:
        return bool(profile["batch_wide_absent_any"]) == bool(cond["batch_wide_absent_any"])
    if "batch_association_in" in cond:
        return profile["batch_association"] in cond["batch_association_in"]
    return False


def candidates(profile):
    keys = ("id", "kind", "family", "methods", "evidence", "evidence_level", "caveats", "tolerates_batch_wide_absence",
            "notes")
    return [dict({k: e.get(k) for k in keys}, fits_profile=_fits(e.get("fits_when", {}), profile),
                 fits_when=e.get("fits_when")) for e in _catalog()]


def _rate_table(var, rate_s):
    rows = []
    for lev in var.levels():
        ix = [j for j, x in enumerate(var.values) if x == lev]
        rows.append({"level": lev, "n": len(ix), "missing_rate": float(rate_s[ix].mean())})
    return {"variable": var.name, "role": var.role, "levels": rows}


def run(unit, ctx, params, results):
    f = Finding(AUDIT_ID, CODE, unit).feeds("missingness_handling", "batch_correction", "imputation_hand_off")
    if unit.out_of_scope:
        return f.status("not_applicable", f"Tier 1 does not audit {unit.out_of_scope} ({unit.scale}).")
    X, Y = unit.X, unit.Y
    p, n = X.shape
    if p == 0 or n == 0:
        return f.status("insufficient_data", "no samples or features in this unit")
    M = f.measures
    R = np.isnan(X)
    tied, _, _, S = floor_ties(X, params["floor_min_observed"])
    zero = (X == 0) & ~R
    floor_cells = S & ~zero
    observed = ~R & ~zero & ~floor_cells

    # 12.1 cell encoding
    enc = {"observed": int(observed.sum()), "nan": int(R.sum()), "zero": int(zero.sum()),
           "floor_tied": int(floor_cells.sum()), "n_cells": int(X.size), "n_floor_tied_features": int(tied.sum())}
    strata = {}
    for i, s in enumerate(unit.strata):
        e = strata.setdefault(s, {"stratum": s, "observed": 0, "nan": 0, "zero": 0, "floor_tied": 0, "n_features": 0})
        e["n_features"] += 1
        e["observed"] += int(observed[i].sum())
        e["nan"] += int(R[i].sum())
        e["zero"] += int(zero[i].sum())
        e["floor_tied"] += int(floor_cells[i].sum())
    M["cell_encoding"] = dict(enc, by_stratum=list(strata.values()), stratum_source=unit.strata_source,
                              note="floor-tied values are observed numbers; they are counted apart, not converted")
    questions = []
    zero_answer = next((o.get("value") for o in reversed(ctx.overrides) if o.get("kind") == "zeros_are_missing"
                        and o.get("assay") in (None, "", unit.unit_id)), None)
    if enc["zero"] and unit.scale != "count_like":
        M["zeros_answer"] = zero_answer
        if zero_answer is None:
            questions.append({"kind": "zeros_meaning", "question_id": f"zeros:{ctx.dataset_id}:{unit.unit_id}",
                              "dataset": ctx.dataset_id, "assay": unit.unit_id, "n_zero": enc["zero"],
                              "text": "Do zeros represent missing values?", "options": ["yes", "no"],
                              "consequence": "yes: the plan proposes the op recode_zero_as_missing (applied only after "
                                             "approval); no: zeros stay observed values"})
    M["questions"] = questions

    # 12.2 rates and structure
    rate_f, rate_s = R.mean(1), R.mean(0)
    B, G, skipped = ctx.variables(unit)
    M["rates"] = {"overall": float(R.mean()), "per_feature": stats.summary(rate_f), "per_sample": stats.summary(rate_s),
                  "by_batch": [_rate_table(v, rate_s) for v in B if not v.numeric],
                  "by_group": [_rate_table(v, rate_s) for v in G if not v.numeric and v.role not in ("subject", "time")],
                  "by_time": [_rate_table(v, rate_s) for v in G if v.role == "time" and not v.numeric],
                  "pattern": {"share_no_nan": float((rate_f == 0).mean()), "share_le_10": float((rate_f <= 0.10).mean()),
                              "share_le_25": float((rate_f <= 0.25).mean()), "share_le_50": float((rate_f <= 0.50).mean())}}

    # 12.3 batch-wide absence and batch coverage
    bwa_any = np.zeros(p, bool)
    absence = []
    for v in B:
        if v.numeric:
            continue
        levs = v.levels()
        per, cov1, cov2 = [], np.ones(p, bool), np.ones(p, bool)
        absent = np.zeros(p, bool)
        for lev in levs:
            ix = [j for j, x in enumerate(v.values) if x == lev]
            obs_n = (~R[:, ix]).sum(1)
            cov1 &= obs_n >= 1
            cov2 &= obs_n >= 2
            if len(ix) >= params["a12_min_level_n"]:
                a = obs_n == 0
                absent |= a
                per.append({"level": lev, "n_samples": len(ix), "n_features_absent": int(a.sum())})
            else:
                per.append({"level": lev, "n_samples": len(ix), "n_features_absent": None,
                            "reason": f"fewer than {params['a12_min_level_n']} samples"})
        bwa_any |= absent
        absence.append({"variable": v.name, "n_features": int(absent.sum()), "share": float(absent.mean()),
                        "features": [unit.feature_keys[i] for i in np.where(absent)[0][:500]],
                        "per_level": per, "coverage_100_share": float(cov1.mean()), "coverage_2_share": float(cov2.mean())})
    M["batch_wide_absence"] = {"variables": absence, "n_features_any": int(bwa_any.sum()),
                               "share_any": float(bwa_any.mean())} if B else \
        {"status": "insufficient_metadata", "reason": "no batch candidate"}

    # 12.4 detection curve
    dc = detection_curve(Y, params)
    plot = dc.pop("_plot", None)
    if plot:
        f.plot["detection_curve"] = plot
    M["detection_curve"] = dc
    if dc["status"] == "not_applicable":
        M["detection_curve"]["substitute"] = "A4 item 4 (floor rate against abundance)"

    # 12.5 association with design
    subj = ctx.subject_codes(unit)
    rows = []
    if R.any():
        for v in B + G:
            if v.role == "subject":
                continue
            r = association(rate_s, v, subj, params, f"A12|{unit.dataset_id}|{unit.unit_id}|nan|{v.name}")
            r["response"] = "per-sample NaN rate"
            rows.append(r)
    rows, n_tests = table(rows)
    feat_tests, sig_any = [], np.zeros(p, bool)
    for v in B:
        if v.numeric:
            continue
        t = nan_batch_tests(R, v, subj, params, f"A12feat|{unit.dataset_id}|{unit.unit_id}|{v.name}")
        if t is None:
            continue
        mask = t.pop("_sig_mask", None)
        if mask is not None:
            tested, sig = mask
            sig_any[np.where(tested)[0][sig]] = True
        feat_tests.append(t)
    M["design_association"] = {"per_sample": {"n_tests": n_tests, "rows": rows}, "per_feature": feat_tests,
                               "n_features_significant_any": int(sig_any.sum())}

    # 12.6 feature classes (descriptive, heuristic)
    with np.errstate(invalid="ignore"):
        mean_obs = np.where((~R).sum(1) > 0, np.where(~R, Y, 0).sum(1) / np.maximum((~R).sum(1), 1), np.nan)
    finite = np.isfinite(mean_obs)
    cut = np.quantile(mean_obs[finite], params["a12_low_abundance_fraction"]) if finite.any() else np.nan
    anynan = rate_f > 0
    cls = np.array(["complete"] * p, dtype=object)
    cls[anynan] = "scattered"
    low = anynan & finite & (mean_obs <= cut)
    cls[low] = "low_abundance_pattern"
    cls[anynan & bwa_any] = "batch_wide_absent"
    counts = {c: int((cls == c).sum()) for c in ("batch_wide_absent", "low_abundance_pattern", "scattered", "complete")}
    M["feature_classes"] = {"counts": counts, "precedence": ["batch_wide_absent", "low_abundance_pattern", "scattered"],
                            "evidence_level": "heuristic",
                            "note": "the labels describe a pattern; they are never called a mechanism"}

    # 12.7 pattern summary
    if dc["status"] == "computed":
        lo, hi = dc["b1_ci95"]
        band = params["a12_b1_negligible"]
        intensity = "yes" if (lo > 0 or hi < 0) else ("no" if (lo >= -band and hi <= band) else "indeterminate")
    else:
        intensity = "indeterminate"
    if not B or all(v.numeric for v in B):
        batch = "indeterminate"
    else:
        share = M["batch_wide_absence"]["share_any"]
        batch = "yes" if (sig_any.any() or share > params["a12_batch_absence_share"]) else "no"
    floor = "yes" if (tied.mean() if tied.size else 0) >= params["floor_tie_share_indicator"] else "no"
    if not R.any():
        pclass = "none"
    elif intensity == "yes" and batch == "yes":
        pclass = "mixed"
    elif intensity == "yes":
        pclass = "mostly_intensity_dependent"
    elif batch == "yes":
        pclass = "batch_associated"
    else:
        pclass = "insufficient_data"
    M["pattern"] = {"pattern_class": pclass, "indicators": {"intensity_dependence": intensity,
                                                            "batch_association": batch, "floor_pattern": floor},
                    "rules": PATTERN_RULES, "rules_evidence_level": "heuristic", "wording": WORDING,
                    "mechanism_note": MOLENBERGHS}
    f.indicate("missingness_pattern", {"pattern_class": pclass, "intensity_dependence": intensity,
                                       "batch_association": batch, "floor_pattern": floor},
               pclass=pclass.replace("_", " "), intensity=intensity, batch=batch, floor=floor)

    # 12.8 the feasible set handed forward (never ranked, never applied)
    nan_feats = max(int(anynan.sum()), 1)
    profile = {"pattern_class": pclass, "batch_association": batch, "batch_wide_absent_any": bool(bwa_any.any()),
               "scattered_majority": counts["scattered"] / nan_feats >= 0.5 if anynan.any() else False}
    M["imputation_candidates"] = {"hand_off": "hand-off to the method-selection stage; nothing is imputed",
                                  "ranked": False, "profile": profile,
                                  "entries": candidates(profile) if R.any() else []}
    if dc["status"] == "insufficient_data" and R.any():
        f.status("partial", "the detection curve needs more features with partial detection; the rest is computed")
    f.method("cell encoding, rates, batch-wide absence, zero-truncated binomial detection curve, restricted-"
             "permutation associations (BH over features), descriptive feature classes", n_used=n,
             transform=unit.transform, permutations=params["permutations"], seed=params["seed"])
    return f
