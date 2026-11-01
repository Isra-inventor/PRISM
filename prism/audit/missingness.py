"""G7 missingness, part 1: describe what is missing and where. Never fills anything.

Hard rule (Build Request v4, section 5): nothing here writes a filled value anywhere. Missing (NaN)
and zero are counted apart; zeros are never turned into missing here (if you said in step 7 that
zeros mean "not detected", the plan will later propose the op recode_zero_as_missing, applied
only after approval). Floor-tied values are numbers in the file and count as observed.

A4 missingness (audit set: study samples, unflagged proteins), per block:
  1. how much is missing, per protein and per sample; zeros apart
  2. floor ties: proteins with >= 2 samples exactly at their minimum value (>= 5 observed values);
     how many cells sit at that floor
  3. abundance dependence: Spearman rho between a protein's missing rate (and floor rate) and its
     mean log2 value over cells that are neither missing nor at the floor. Strongly negative
     (rho <= -0.1, heuristic): low-abundance proteins miss more, consistent with a detection limit.

A12 missingness profile, part 1:
  5.1 cell encoding: every cell is exactly one of missing, zero, floor-tied, observed (counts)
  5.2 rates by batch, by group, by time point; share of proteins with no / <= 10% / <= 25% / <= 50%
      missing (reported, none chosen)
  5.3 batch-wide absence: proteins with no observed value in at least one batch level that has
      >= 3 samples. This pattern breaks both imputation and ComBat, so it is counted per batch.
      Also batch coverage: share of proteins with >= 1 (and >= 2) observed values in every level.
A12 part 2:
  5.4 detection curve: logit P(detected) = b0 + b1 * mean log2 value, zero-truncated binomial, 95% CI
      of b1 from a seeded bootstrap over proteins. A simplified reference method (see LIMITATION).
  5.5 the missing pattern against the design: each sample's missing (and floor) rate against every
      batch / group / time / run-order variable (restricted permutations, BH); per protein, its
      missing indicator against each batch variable (chi-square, permutation p, BH over proteins)
  5.6 protein classes: batch_wide_absent > low_abundance_pattern > scattered; complete (heuristic labels)
  5.7 pattern summary: none | mostly_intensity_dependent | batch_associated | mixed | insufficient_data,
      with three-valued indicators (yes / no / indeterminate). Never "MCAR", "MAR", "MNAR".
  5.8 imputation_candidates from imputation_catalog.yaml: a feasible set for the later method-selection
      stage, never ranked, never applied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .finding import Finding
from .processing_state import floor_ties


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float | None, int]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return None, int(ok.sum())
    rx, ry = pd.Series(x[ok]).rank(), pd.Series(y[ok]).rank()
    return float(np.corrcoef(rx, ry)[0, 1]), int(ok.sum())


def _summary(v: np.ndarray) -> dict:
    return {"min": float(v.min()), "median": float(np.median(v)), "p90": float(np.quantile(v, 0.9)),
            "max": float(v.max())} if v.size else {}


def floor_cells(X: np.ndarray, min_obs: int) -> tuple[np.ndarray, np.ndarray]:
    """(tied per protein, S: cells sitting at the floor of a tied protein)."""
    tied, _ = floor_ties(X, min_obs)
    with np.errstate(invalid="ignore"):
        m = np.nanmin(np.where(np.isfinite(X), X, np.inf), axis=1)
    S = np.isfinite(X) & (X == m[:, None]) & tied[:, None]
    return tied, S


def run_a4(audit) -> Finding:
    p = audit.params
    f = Finding("A4", "Missingness", step="G7", feeds=["G7 missingness profile", "S2 batch correction",
                                                       "G5 log transform", "S3 feature filtering"],
                method="exact counts on the audit set; floor ties on proteins with >= "
                       f"{p['floor_min_observed']} observed values; Spearman rho on log2 means")
    for bid, g in audit.schema.block_map().items():
        X = audit.audit_matrix(bid).to_numpy(dtype=float)
        n_p, n_s = X.shape
        R = np.isnan(X)
        n_cells, n_miss, n_zero = X.size, int(R.sum()), int((X == 0).sum())
        prate, srate = R.mean(1), R.mean(0)
        m = {"n_proteins": n_p, "n_samples": n_s, "n_cells": n_cells,
             "missing": {"n": n_miss, "rate": n_miss / n_cells if n_cells else 0, "per_protein": _summary(prate),
                         "per_sample": _summary(srate), "proteins_with_any": int((prate > 0).sum()),
                         "proteins_all_missing": int((prate == 1).sum()), "samples_with_any": int((srate > 0).sum())},
             "zeros": {"n": n_zero, "rate": n_zero / n_cells if n_cells else 0, "zero_means": g.zero_means}}
        if n_miss:
            f.say("note", f"{bid}: {100 * n_miss / n_cells:.2f}% of cells are missing ({n_miss} of {n_cells}); "
                          f"{m['missing']['proteins_with_any']} of {n_p} proteins miss at least one value.")
        else:
            f.say("ok", f"{bid}: no value is missing.")
        if n_zero:
            f.say("note", f"{bid}: {100 * n_zero / n_cells:.2f}% of cells are exactly zero ({n_zero}); zeros are counted "
                          f"apart from missing values. You said zeros mean '{g.zero_means}'"
                          + (": the plan will propose recoding them as missing (after your approval)."
                             if g.zero_means == "not_detected" else "."))

        tied, S = floor_cells(X, p["floor_min_observed"])
        n_tied = int(tied.sum())
        eligible = int((np.isfinite(X).sum(1) >= p["floor_min_observed"]).sum())
        m["floor_ties"] = {"n_proteins": n_tied, "n_eligible": eligible, "share": n_tied / eligible if eligible else 0,
                           "n_cells": int(S.sum())}
        if n_tied and n_tied / max(eligible, 1) >= p["floor_tie_share_indicator"]:
            f.say("note", f"{bid}: {100 * n_tied / eligible:.1f}% of proteins ({n_tied} of {eligible}) have two or more "
                          f"samples at their minimum value ({int(S.sum())} cells): consistent with values set to a floor "
                          "(for example a detection limit or an imputed minimum).")
        elif n_tied:
            f.say("ok", f"{bid}: {n_tied} of {eligible} proteins have floor ties (below the "
                        f"{100 * p['floor_tie_share_indicator']:.0f}% indicator).")

        Y = audit.diagnostic(bid)[0].to_numpy()
        keep = ~R & ~S
        with np.errstate(invalid="ignore"):
            mean_y = np.where(keep.sum(1) > 0, np.where(keep, Y, 0).sum(1) / np.maximum(keep.sum(1), 1), np.nan)
        dep = {}
        for what, rate, active in (("missing", prate, n_miss > 0), ("floor", S.mean(1), n_tied > 0)):
            if not active:
                continue
            rho, n_used = spearman(rate, mean_y)
            dep[what] = {"rho": rho, "n_proteins": n_used}
            if rho is not None and rho <= p["abundance_rho_indicator"]:
                f.say("note", f"{bid}: lower-abundance proteins have more {what} values (Spearman rho {rho:.3f} over "
                              f"{n_used} proteins), consistent with a detection limit.")
        m["abundance_dependence"] = dep
        f.measures[bid] = m
        audit.results.setdefault("floor", {})[bid] = S
    return f


def _levels(values: pd.Series, min_n: int) -> dict:
    out = {}
    for lev, idx in values.groupby(values).groups.items():
        out[lev] = list(idx)
    return {k: v for k, v in out.items() if k not in ("", None)}


def run_a12_part1(audit) -> Finding:
    p = audit.params
    f = Finding("A12", "Missingness profile", step="G7", feeds=["S2 batch correction", "S3 feature filtering",
                                                               "the later imputation decision"],
                method="exact counts on the audit set, per block; batch / group / time from your sample "
                       f"information; batch levels need >= {p['a12_min_level_n']} samples for batch-wide absence")
    variables = audit.variables()
    for bid in audit.schema.block_map():
        Xd = audit.audit_matrix(bid)
        X = Xd.to_numpy(dtype=float)
        R = np.isnan(X)
        S = audit.results.get("floor", {}).get(bid)
        if S is None:
            _, S = floor_cells(X, p["floor_min_observed"])
        n_cells = X.size
        Z = X == 0
        # every cell gets exactly one class, in this order: missing, zero, floor-tied, observed
        enc = {"observed": int((~R & ~Z & ~S).sum()), "missing": int(R.sum()), "zero": int(Z.sum()),
               "floor_tied": int((S & ~Z).sum()), "n_cells": n_cells}
        m = {"cell_encoding": enc}
        f.say("ok", f"{bid}: cells: {enc['observed']} observed, {enc['missing']} missing, {enc['zero']} zero, "
                    f"{enc['floor_tied']} at a protein's floor (of {n_cells}).")

        prate = R.mean(1)
        m["protein_missing_shares"] = {k: round(float(v), 4) for k, v in (
            ("no_missing", (prate == 0).mean()), ("at_most_10pct", (prate <= 0.10).mean()),
            ("at_most_25pct", (prate <= 0.25).mean()), ("at_most_50pct", (prate <= 0.50).mean()))}

        by = {}
        for name, (kind, values) in variables.items():
            vals = values.reindex(Xd.columns).fillna("").astype(str).str.strip()
            rows = []
            for lev, cols in _levels(vals, 1).items():
                j = [Xd.columns.get_loc(c) for c in cols]
                rows.append({"level": lev, "n_samples": len(j), "missing_rate": round(float(R[:, j].mean()), 4)})
            if rows and kind in ("batch", "group", "timepoint"):
                by[name] = {"audit_kind": kind, "levels": rows}
        m["rates_by_variable"] = by

        absence = {}
        for name, (kind, values) in variables.items():
            if kind != "batch":
                continue
            vals = values.reindex(Xd.columns).fillna("").astype(str).str.strip()
            levels = {lev: [Xd.columns.get_loc(c) for c in cols] for lev, cols in _levels(vals, 1).items()}
            if len(levels) < 2:
                continue
            big = {lev: j for lev, j in levels.items() if len(j) >= p["a12_min_level_n"]}
            obs_per_level = {lev: (~R[:, j]).sum(1) for lev, j in levels.items()}
            absent = np.zeros(X.shape[0], bool)
            per_level = {}
            for lev, j in big.items():
                a = obs_per_level[lev] == 0
                per_level[lev] = int(a.sum())
                absent |= a
            cov1 = np.all([obs_per_level[lev] >= 1 for lev in levels], axis=0)
            cov2 = np.all([obs_per_level[lev] >= 2 for lev in levels], axis=0)
            absence[name] = {"n_proteins_batch_wide_absent": int(absent.sum()), "share": round(float(absent.mean()), 4),
                             "per_level": per_level, "levels_too_small": sorted(set(levels) - set(big)),
                             "coverage_at_least_1_every_level": round(float(cov1.mean()), 4),
                             "coverage_at_least_2_every_level": round(float(cov2.mean()), 4)}
            if absent.any():
                f.say("note", f"{bid}: {int(absent.sum())} protein(s) ({100 * absent.mean():.1f}%) have no observed value "
                              f"in at least one level of '{name}' (per level: {per_level}). This pattern breaks both "
                              "imputation across batches and ComBat.")
            f.say("ok", f"{bid}: {100 * cov1.mean():.1f}% of proteins have at least one observed value in every level of "
                        f"'{name}' ({100 * cov2.mean():.1f}% at least two).")
        if not any(k == "batch" for k, _ in variables.values()):
            f.say("not_applicable", f"{bid}: no batch variable in the sample information, so batch-wide absence "
                                    "cannot be assessed.")
        m["batch_wide_absence"] = absence
        f.measures[bid] = m
    return f


# ====================================================================== A12 part 2 (sections 5.4-5.8)

from pathlib import Path   # noqa: E402

import yaml   # noqa: E402

from . import stats   # noqa: E402

WORDING = "Pattern quantified, plausible mechanisms assessed."
MOLENBERGHS = ("Models in which values are missing at random and models in which they are not can fit the observed "
               "data equally well (Molenberghs, Beunckens, Sotto and Kenward 2008), so the observed data cannot settle "
               "the mechanism.")
LIMITATION = ("PRISM's detection curve is a simplified reference implementation: ybar is the mean of the detected "
              "values only, so it is biased upward for proteins with censoring. The published method (Li and Smyth "
              "2023, 'Neither random nor censored') uses a more careful likelihood with moderated variances. Read b1 "
              "as a descriptive measure.")
PATTERN_RULES = [
    "none: no value is missing",
    "mixed: intensity dependence = yes and batch association = yes",
    "mostly_intensity_dependent: intensity dependence = yes, batch association not yes",
    "batch_associated: batch association = yes, intensity dependence not yes",
    "insufficient_data: otherwise (neither indicator is yes, or neither could be assessed)",
    "intensity dependence: yes when the 95% CI of b1 excludes 0 (and is not inside +-a12_b1_negligible); no when it "
    "lies inside +-a12_b1_negligible; indeterminate otherwise or when the curve is not computed",
    "batch association: yes when any protein is BH-significant (q <= 0.05) or the batch-wide absence share is above "
    "a12_batch_absence_share; no when tested and neither holds; indeterminate without a batch variable",
    "floor pattern: yes when the floor-tied protein share reaches floor_tie_share_indicator",
]


# ---------------------------------------------------------------- 5.4 the detection curve

def _expit(x):
    return 0.5 * (1 + np.tanh(0.5 * x))


def ztb_negll(beta, x, d, n):
    """Negative log-likelihood (and gradient) of the zero-truncated binomial with logit p = b0 + b1 x:
    protein i is detected in d_i of n samples, and only proteins with d_i >= 1 are in the data."""
    eta = beta[0] + beta[1] * x
    p = _expit(eta)
    log1mp = -np.logaddexp(0, eta)                # log(1 - p)
    logp = -np.logaddexp(0, -eta)                 # log p
    qn = np.exp(n * log1mp)                       # (1 - p)^n: the chance of never being detected
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


def detection_curve(Y: np.ndarray, p: dict) -> dict:
    ok = np.isfinite(Y)
    n = Y.shape[1]
    d = ok.sum(1)
    if not (~ok).any():
        return {"status": "not_applicable", "reason": "no missing values: the floor-based abundance check (A4) "
                                                      "is the substitute"}
    partial = int(((d > 0) & (d < n)).sum())
    if partial < p["a12_min_partial_features"]:
        return {"status": "insufficient_data", "n_partial": partial,
                "reason": f"{partial} protein(s) detected in some but not all samples; the curve needs "
                          f"{p['a12_min_partial_features']}"}
    use = d >= 1
    with np.errstate(invalid="ignore"):
        ybar = (np.where(ok, Y, 0).sum(1) / np.maximum(d, 1))[use]
    du = d[use].astype(float)
    b0, b1, conv = fit_detection(ybar, du, n)
    rng = np.random.default_rng([p["seed"], 12])
    boots = np.array([fit_detection(ybar[i], du[i], n)[1]
                      for i in (rng.integers(0, len(ybar), len(ybar)) for _ in range(p["a12_bootstrap"]))])
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    q10, q90 = float(np.quantile(ybar, 0.1)), float(np.quantile(ybar, 0.9))
    p10, p90 = float(_expit(b0 + b1 * q10)), float(_expit(b0 + b1 * q90))
    return {"status": "computed", "model": "zero-truncated binomial, logit P(detected) = b0 + b1 * ybar (mean "
                                           "observed log2 value), maximum likelihood, proteins detected at least once",
            "b0": b0, "b1": b1, "b1_ci95": [lo, hi], "converged": conv, "bootstrap": p["a12_bootstrap"],
            "n_proteins_used": int(use.sum()), "n_partial": partial, "n_samples": n,
            "ybar_p10": q10, "ybar_p90": q90, "p_detect_at_p10": p10, "p_detect_at_p90": p90,
            "plateau_missing_share": 1 - p90, "limitation": LIMITATION}


# ---------------------------------------------------------------- 5.5 the missing pattern against the design

def nan_batch_tests(R: np.ndarray, values: list[str], subject, p: dict, key: str) -> dict | None:
    """Per protein: chi-square of the 2 x k table (missing / observed by batch level), p from restricted
    permutations of the batch labels (valid with small cells), BH over proteins."""
    idx = [j for j, v in enumerate(values) if v != ""]
    levels = sorted({values[j] for j in idx})
    if len(levels) < 2:
        return None
    codes = np.array([levels.index(values[j]) for j in idx])
    Rs = R[:, idx].astype(float)
    m_tot = Rs.sum(1)
    test = (m_tot > 0) & (m_tot < len(idx))
    if not test.any():
        return {"n_tested": 0, "n_significant": 0}
    Rt, k, n = Rs[test], len(levels), len(idx)

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
    sub = None if subject is None else [subject[j] for j in idx]
    perm = stats.Permuter(codes, sub, p["a12_feature_permutations"], 0, p["seed"], key)
    P = perm.matrix()
    ge = np.zeros(len(obs))
    for row in P:
        ge += chi2(row) >= obs - 1e-12 * np.maximum(1.0, np.abs(obs))
    pv = (1 + ge) / (1 + len(P))
    q = np.array([x if x is not None else np.nan for x in stats.bh(list(pv))], float)
    sig = q <= 0.05
    full = np.zeros(R.shape[0], bool)
    full[np.where(test)[0][sig]] = True
    return {"levels": levels, "n_tested": int(test.sum()), "n_significant": int(np.nansum(sig)),
            "statistic": "chi-square (2 x k)", "p_min_attainable": 1.0 / (1 + len(P)), **perm.describe(),
            "_significant": full}


def _numeric(values: pd.Series) -> bool:
    v = values.dropna().astype(str).str.strip()
    v = v[v != ""]
    return len(v) > 0 and pd.to_numeric(v, errors="coerce").notna().all()


# ---------------------------------------------------------------- 5.8 the hand-off

def _catalog() -> list[dict]:
    return yaml.safe_load((Path(__file__).parent / "imputation_catalog.yaml").read_text(encoding="utf-8"))["families"]


def _fits(cond: dict, profile: dict) -> bool:
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


def candidates(profile: dict) -> list[dict]:
    keys = ("id", "kind", "family", "methods", "evidence", "evidence_level", "caveats", "tolerates_batch_wide_absence",
            "notes")
    return [dict({k: e.get(k) for k in keys}, fits_profile=_fits(e.get("fits_when", {}), profile),
                 fits_when=e.get("fits_when")) for e in _catalog()]


def run_a12_part2(audit, f: Finding) -> Finding:
    """Adds 5.4-5.8 to the A12 finding, block by block."""
    p = audit.params
    variables = audit.variables()
    subj_series = variables.get("subject", (None, None))[1]
    for bid in audit.schema.block_map():
        Xd = audit.audit_matrix(bid)
        X = Xd.to_numpy(dtype=float)
        R = np.isnan(X)
        Y = audit.diagnostic(bid)[0].to_numpy()
        m = f.measures[bid]
        subject = None if subj_series is None else \
            subj_series.reindex(Xd.columns).fillna("").astype(str).str.strip().tolist()

        # 5.4
        dc = detection_curve(Y, p)
        m["detection_curve"] = dc
        if dc["status"] == "computed":
            lo, hi = dc["b1_ci95"]
            f.say("ok", f"{bid}: detection curve: each log2 unit of abundance changes the log-odds of being detected "
                        f"by b1 = {dc['b1']:.3f} (95% CI {lo:.3f} to {hi:.3f}; {dc['n_proteins_used']} proteins). "
                        f"Fitted chance of detection: {dc['p_detect_at_p10']:.2f} at the 10th and "
                        f"{dc['p_detect_at_p90']:.2f} at the 90th percentile of abundance.")
        elif R.any():
            f.say("note", f"{bid}: detection curve not computed: {dc['reason']}.")

        # 5.5 per-sample rates and per-protein indicators against the design
        rows = []
        responses = [("missing", R.mean(0))] if R.any() else []
        S = audit.results.get("floor", {}).get(bid)
        if S is not None and S.any():
            responses.append(("floor", S.mean(0)))
        for what, resp in responses:
            for name, (kind, series) in variables.items():
                if kind in ("subject_id", "sample_type", "other"):
                    continue
                vals = series.reindex(Xd.columns).fillna("").astype(str).str.strip().tolist()
                numeric = kind in ("run_order", "covariate") and _numeric(series)
                r = stats.association(resp, vals, numeric, subject, p, f"A12|{bid}|{what}|{name}")
                rows.append(dict(r, variable=name, audit_kind=kind, response=f"per-sample {what} rate"))
        qs = stats.bh([r.get("p") if r["status"] == "computed" else None for r in rows])
        for r, q in zip(rows, qs):
            if q is not None:
                r["q"] = q
                if q <= 0.05:
                    f.say("note", f"{bid}: the {r['response']} differs with '{r['variable']}' "
                                  f"(permutation p {r['p']:.4f}, q {q:.4f}; {r['scheme']}).")
        per_protein, sig_any = [], np.zeros(X.shape[0], bool)
        for name, (kind, series) in variables.items():
            if kind != "batch" or not R.any():
                continue
            vals = series.reindex(Xd.columns).fillna("").astype(str).str.strip().tolist()
            t = nan_batch_tests(R, vals, subject, p, f"A12feat|{bid}|{name}")
            if t is None:
                continue
            sig = t.pop("_significant", None)
            if sig is not None:
                sig_any |= sig
            per_protein.append(dict(t, variable=name))
            if t.get("n_significant"):
                f.say("note", f"{bid}: {t['n_significant']} of {t['n_tested']} proteins have a missing pattern that "
                              f"follows '{name}' (BH q <= 0.05).")
        m["design_association"] = {"per_sample": rows, "per_protein": per_protein,
                                   "n_proteins_significant_any": int(sig_any.sum())}

        # 5.6 protein classes (descriptive, heuristic)
        bwa_any = np.zeros(X.shape[0], bool)
        for name, (kind, series) in variables.items():
            if kind != "batch":
                continue
            vals = series.reindex(Xd.columns).fillna("").astype(str).str.strip()
            for lev in set(vals) - {""}:
                j = np.where(vals.values == lev)[0]
                if len(j) >= p["a12_min_level_n"]:
                    bwa_any |= (~R[:, j]).sum(1) == 0
        with np.errstate(invalid="ignore"):
            mean_obs = np.where((~R).sum(1) > 0, np.where(~R, Y, 0).sum(1) / np.maximum((~R).sum(1), 1), np.nan)
        finite = np.isfinite(mean_obs)
        cut = np.quantile(mean_obs[finite], p["a12_low_abundance_fraction"]) if finite.any() else np.nan
        anynan = R.any(1)
        cls = np.array(["complete"] * X.shape[0], dtype=object)
        cls[anynan] = "scattered"
        cls[anynan & finite & (mean_obs <= cut)] = "low_abundance_pattern"
        cls[anynan & bwa_any] = "batch_wide_absent"
        counts = {c: int((cls == c).sum()) for c in ("batch_wide_absent", "low_abundance_pattern", "scattered",
                                                     "complete")}
        m["protein_classes"] = {"counts": counts, "precedence": ["batch_wide_absent", "low_abundance_pattern",
                                                                 "scattered"], "evidence_level": "heuristic",
                                "note": "the labels describe a pattern; they are never called a mechanism"}
        audit.results.setdefault("protein_class", {})[bid] = pd.Series(cls, index=Xd.index)
        if anynan.any():
            f.say("ok", f"{bid}: proteins with missing values: {counts['batch_wide_absent']} absent from a whole batch, "
                        f"{counts['low_abundance_pattern']} in the lowest third of abundance, {counts['scattered']} "
                        f"scattered ({counts['complete']} complete). Descriptive labels, not mechanisms.")

        # 5.7 pattern summary
        if dc["status"] == "computed":
            lo, hi = dc["b1_ci95"]
            band = p["a12_b1_negligible"]
            intensity = "no" if (lo >= -band and hi <= band) else ("yes" if (lo > 0 or hi < 0) else "indeterminate")
        else:
            intensity = "indeterminate"
        has_batch = any(k == "batch" for k, _ in variables.values())
        batch = "indeterminate" if not has_batch else \
            ("yes" if (sig_any.any() or bwa_any.mean() > p["a12_batch_absence_share"]) else "no")
        floor = "yes" if m.get("cell_encoding", {}).get("floor_tied", 0) and \
            (audit.findings.get("A4") and audit.findings["A4"].measures[bid]["floor_ties"]["share"]
             >= p["floor_tie_share_indicator"]) else "no"
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
        m["pattern"] = {"pattern_class": pclass, "indicators": {"intensity_dependence": intensity,
                                                                "batch_association": batch, "floor_pattern": floor},
                        "rules": PATTERN_RULES, "rules_evidence_level": "heuristic", "wording": WORDING,
                        "mechanism_note": MOLENBERGHS}
        f.say("ok", f"{bid}: missingness pattern: {pclass.replace('_', ' ')} (intensity dependence {intensity}, batch "
                    f"association {batch}, floor pattern {floor}). {WORDING}")

        # 5.8 the feasible set handed forward (never ranked, never applied)
        n_nan = max(int(anynan.sum()), 1)
        profile = {"pattern_class": pclass, "batch_association": batch, "batch_wide_absent_any": bool(bwa_any.any()),
                   "scattered_majority": counts["scattered"] / n_nan >= 0.5 if anynan.any() else False}
        m["imputation_candidates"] = {"hand_off": "hand-off to the method-selection stage; nothing is imputed",
                                      "ranked": False, "profile": profile,
                                      "entries": candidates(profile) if R.any() else []}
    if any(st != "not_applicable" for st, _ in f.sentences):
        f.say("ok", MOLENBERGHS)
    return f
