"""The finding format (v3 §6.2) and the templated indicator sentences.

Indicator text only comes from TEMPLATES, with numbers filled from the evidence. Wording:
"plausible indicator", "consistent with"; never "diagnosed" or "confirmed".
"""

from __future__ import annotations

import json
import math

import numpy as np

from . import stats
from .context import Variable

VERSION = "0.1"
PLOT_LIMIT = 1_000_000

TEMPLATES = {
    # A4
    "missing_values_present": "{pct}% of cells are missing ({n_missing} of {n_cells}).",
    "zeros_present": "{pct}% of cells are exactly zero ({n_zero} of {n_cells}); zeros are counted apart from missing values.",
    "sentinel_value": "The value {value} covers {pct}% of all cells: a plausible indicator of a placeholder or fill value.",
    "floor_ties_present": "{pct}% of features ({n_features} of {of}) have two or more samples at their minimum value, "
                          "consistent with values set to a floor (for example a detection limit or an imputed minimum).",
    "abundance_dependent_missingness": "Lower-abundance features have more {what} values (Spearman rho {rho} over "
                                       "{n} features), consistent with detection-limit censoring.",
    "declared_not_imputed_floor_ties": "Declared not imputed, but {pct}% of features have two or more samples at their "
                                       "minimum: a plausible indicator of imputation with a minimum or a floor value.",
    "declared_not_normalized_signature": "Declared not normalized, but the values show a {what}: consistent with "
                                         "normalization or scaling before upload.",
    "declared_not_logged_log_like": "Declared not log-transformed, but {what}: consistent with log-scale values.",
    "missingness_associated": "The per-sample {what} rate differs with '{variable}' (permutation p {p}, q {q}).",
    # A5
    "few_samples_per_feature": "There are {n} samples for {p} features (n/p = {ratio}).",
    "effective_sample_size": "With {n_subjects} subjects and a median ICC of {icc}, the effective sample size is about "
                             "{n_eff} of {n} samples (design effect {deff}).",
    "small_cell": "The smallest design cell has {size} sample(s) ('{variable}' = {level}).",
    # A8
    "repeated_measures": "{n_subjects} subjects with 1 to {max_size} samples each ({n_singletons} with a single sample): "
                         "samples of one subject are not independent.",
    "paired_design": "Every subject has exactly two samples: a paired design.",
    "irregular_time_grid": "Time spacing within subjects is irregular (gaps from {min_gap} to {max_gap} {unit}).",
    "high_icc": "{pct}% of features have ICC above {cut}: subject identity explains much of the variation, "
                "consistent with strong within-subject correlation.",
    "within_subject_correlation": "Samples of the same subject correlate more (mean r {same}) than samples of different "
                                  "subjects (mean r {diff}).",
    "variable_constant_within_subject": "'{variable}' is constant within every subject: subjects are nested in "
                                        "'{variable}'.",
    # A7
    "batch_confounded": "'{batch}' and '{design}' are confounded in this design: their effects cannot be separated.",
    "batch_nested": "{text}: the two effects cannot be fully separated.",
    "pc_association": "PC{pc} ({pct}% of variance) is associated with '{variable}' ({stat} {value}, q {q}).",
    "permanova_association": "'{variable}' accounts for {r2}% of the between-sample distance (PERMANOVA pseudo-F {F}, "
                             "q {q}; marginal).",
    # A11
    "outlier_sample": "Sample '{sample}' is a plausible outlier by {by} (modified z {z}, cut-off {cut}).",
    "outlier_cells": "{n_cells} cell(s) in {n_features} feature(s) have a within-feature modified z beyond {cut}.",
    "outlier_max_cell": "The largest value of stratum '{stratum}', {value} ({feature}, sample {sample}), is a plausible "
                        "outlier cell (modified z {z}, cut-off {cut}).",
    # A1
    "dimension_mismatch": "Block {block} is {file} in the file but {schema} in the schema.",
    "duplicate_feature_keys": "{n} feature key(s) occur more than once.",
    "duplicate_sample_ids": "{n} sample ID(s) occur more than once.",
    "samples_x_vs_m": "{a} sample(s) of the matrix are missing from the sample table and {b} sample(s) of the table "
                      "are missing from the matrix.",
    "unparsable_cells": "{n} cell(s) could not be read as numbers and are treated as missing.",
    "infinite_values": "{n} cell(s) are infinite.",
    "constant_features": "{n} feature(s) are constant over their observed values.",
    "sparse_features": "{n} feature(s) have fewer than 3 observed values.",
    "duplicate_samples": "Samples {samples} have identical values: a plausible indicator of a duplicated column.",
    "duplicate_features": "{n} group(s) of features have identical values.",
    "near_duplicate_samples": "Samples '{a}' and '{b}' correlate at r = {r}: a plausible indicator of a near-duplicate.",
    # A6
    "roles_needed": "{n} sample(s) are not study samples and have no role (qc, blank, pool): set sample roles to "
                    "audit QC precision.",
    "qc_rsd": "Across {n_qc} QC/pool samples the median feature RSD is {median}%; {pct}% of features exceed {hi}%.",
    "run_order_drift": "The per-sample median changes with '{variable}' (Spearman rho {rho}): consistent with "
                       "run-order drift.",
    "suspect_rows": "{n} feature(s) are marked suspect in Step 0 ({cols}); flagged and unflagged features are compared.",
    # A9
    "single_source": "'{variable}' has a single level ({level}): homogeneous (single source).",
    "source_association": "'{variable}' is associated with the data ({what}, q {q}).",
    # A10
    "layer_overlap": "{a} and {b} share {n} sample(s) ({only_a} only in {a}, {only_b} only in {b}; Jaccard {j}).",
    # A2
    "scale_class": "The values look like {text}.",
    "out_of_scope_type": "Tier 1 does not audit {what}: the other audits report not_applicable.",
    "median_scaling_signature": "Per-feature medians are nearly equal (MAD of log2 medians {mad} below {cut}; median of "
                                "medians {median}), consistent with per-feature (median) scaling.",
    # A3
    "sd_grows_with_mean": "On the original scale SD grows with the mean (log-log slope {b}, 95% CI {lo} to {hi}); a "
                          "slope near 1 is consistent with multiplicative noise.",
    "mean_sd_after_transform": "After {transform}, feature mean and SD are still related (Spearman rho {rho}).",
}


def _fmt(v):
    if isinstance(v, float):
        if v != v:
            return "n/a"
        if abs(v) >= 100:
            return f"{v:,.0f}"
        return f"{v:.3g}"
    if isinstance(v, int) and abs(v) >= 10000:
        return f"{v:,}"
    return str(v)


def clean(obj):
    """JSON-safe: numpy scalars to Python, NaN/inf to None, floats rounded to 10 significant digits."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if not math.isfinite(x):
            return None
        return float(f"{x:.10g}")
    return obj


class Finding:
    def __init__(self, audit_id, code, unit=None, dataset=None):
        self.d = {"audit_id": audit_id, "audit": code,
                  "dataset": dataset or (unit.dataset_id if unit else None),
                  "assay": unit.unit_id if unit else None, "assay_label": unit.label if unit else None,
                  "status": "computed", "measures": {}, "indicators": [], "feeds": [], "needs": [],
                  "method": {"name": "", "params": {}, "n_used": None, "version": VERSION}, "plot_data": {}}

    @property
    def measures(self):
        return self.d["measures"]

    @property
    def plot(self):
        return self.d["plot_data"]

    def status(self, s, reason=None, needs=None):
        self.d["status"] = s
        if reason:
            self.d["status_reason"] = reason
        for n in needs or []:
            if n not in self.d["needs"]:
                self.d["needs"].append(n)
        return self

    def method(self, name, n_used=None, transform=None, **params):
        self.d["method"].update(name=name, params=params)
        if transform is not None:
            self.d["method"]["transform"] = transform
        if n_used is not None:
            self.d["method"]["n_used"] = n_used
        return self

    def feeds(self, *f):
        self.d["feeds"] = list(f)
        return self

    def indicate(self, code, evidence=None, **values):
        text = TEMPLATES[code].format(**{k: _fmt(v) for k, v in values.items()})
        self.d["indicators"].append({"code": code, "text": text, "evidence": evidence if evidence is not None else values})
        return self

    def to_json(self):
        out = clean(self.d)
        s = json.dumps(out, sort_keys=True)
        if len(s) > PLOT_LIMIT:
            out["plot_data"] = {"omitted": f"plot data larger than {PLOT_LIMIT} bytes"}
        return out


# ------------------------------------------------------------------ association tables


def _prepare(var, subject, ok):
    subj = None
    if subject is not None:
        subj = np.array([subject[i] for i in range(len(subject))])[ok]
        if any(s == "" for s in subj):
            subj = np.array([s if s else f"_sample{i}" for i, s in enumerate(subj)])
    return subj


def association(response, var, subject, params, key, guard_subjects=True, categorical="kruskal_wallis"):
    """One row of an association table: a per-sample response against a variable, Kruskal-Wallis
    (categorical) or |Spearman| (numeric), with restricted permutation p (v3 §6.1)."""
    response = np.asarray(response, float)
    vals = var.values
    ok = np.array([str(v).strip() != "" for v in vals]) & np.isfinite(response)
    row = {"variable": var.name, "role": var.role, "test": categorical if not var.numeric else "spearman",
           "n_used": int(ok.sum())}
    subj = None
    if subject is not None:
        subj = np.array([subject[i] for i in range(len(subject))])[ok]
        if guard_subjects and len({s for s in subj if s}) < 3:
            return dict(row, status="insufficient_data", reason="fewer than 3 subjects")
        if any(s == "" for s in subj):
            subj = np.array([s if s else f"_sample{i}" for i, s in enumerate(subj)])
    y = response[ok]
    if var.numeric:
        x = var.as_float()[ok]
        if len(y) < 4 or len(set(x.tolist())) < 2:
            return dict(row, status="insufficient_data", reason="fewer than 4 samples or one value")
        ry, rx = stats.rankdata(y), stats.rankdata(x)
        obs = stats.pearson(rx, ry)
        if obs != obs:
            return dict(row, status="insufficient_data", reason="constant response")
        perm = stats.Permuter(rx, subj, params["permutations"], params["exhaustive_max"], params["seed"], key)
        P = perm.matrix().astype(float)
        ryc = ry - ry.mean()
        Pc = P - P.mean(1, keepdims=True)
        denom = np.sqrt((Pc ** 2).sum(1) * float(ryc @ ryc))
        rho = np.where(denom > 0, Pc @ ryc / np.where(denom > 0, denom, 1), 0)
        p, pmin = perm.p_value(np.abs(rho), abs(obs))
        return dict(row, status="computed", statistic="rho", value=obs, p=p, p_min_attainable=pmin, **perm.describe())
    levels = sorted(set(vals[i] for i in np.where(ok)[0]), key=lambda v: (not _isnum(v), _num(v), str(v)))
    codes = np.array([levels.index(vals[i]) for i in np.where(ok)[0]])
    counts = np.bincount(codes, minlength=len(levels))
    row["levels"] = {str(l): int(c) for l, c in zip(levels, counts)}
    if len(levels) < 2:
        return dict(row, status="insufficient_data", reason="fewer than 2 levels")
    if counts.min() < 2:
        return dict(row, status="insufficient_data", reason="smallest cell below 2")
    perm = stats.Permuter(codes, subj, params["permutations"], params["exhaustive_max"], params["seed"], key)
    if perm.n_distinct < 2:
        return dict(row, status="insufficient_data", reason="no permutation is possible under the design")
    P = perm.matrix()
    if categorical == "eta_squared":
        obs = float(stats.eta_sq_rows(y, codes[None, :], len(levels))[0])
        H = stats.eta_sq_rows(y, P, len(levels))
        name = "eta_squared"
    else:
        ry = stats.rankdata(y)
        obs = stats.kruskal_h(ry, codes, len(levels))
        H = stats.kruskal_h_rows(ry, P, len(levels))
        name = "H"
    p, pmin = perm.p_value(H, obs)
    return dict(row, status="computed", statistic=name, value=obs, p=p, p_min_attainable=pmin, **perm.describe())


def guard(var, subject, ok):
    """None, or why the variable cannot be tested (v3 §6.1 guards)."""
    if subject is not None and len({subject[i] for i in np.where(ok)[0] if subject[i]}) < 3:
        return "fewer than 3 subjects"
    vals = [var.values[i] for i in np.where(ok)[0]]
    if var.numeric:
        return "fewer than 4 samples or one value" if len(vals) < 4 or len(set(vals)) < 2 else None
    levels = {}
    for v in vals:
        levels[v] = levels.get(v, 0) + 1
    if len(levels) < 2:
        return "fewer than 2 levels"
    if min(levels.values()) < 2:
        return "smallest cell below 2"
    return None


def permanova(K_full, var, subject, params, key):
    """PERMANOVA of one variable on Euclidean distances of Y (marginal), restricted permutations."""
    ok = np.array([str(v).strip() != "" for v in var.values])
    row = {"variable": var.name, "role": var.role, "test": "permanova", "n_used": int(ok.sum())}
    why = guard(var, subject, ok)
    if why:
        return dict(row, status="insufficient_data", reason=why)
    idx = np.where(ok)[0]
    K = K_full[np.ix_(idx, idx)]
    K = K - K.mean(0) - K.mean(1)[:, None] + K.mean()          # re-centre on the samples used
    subj = _prepare(var, subject, ok)
    if var.numeric:
        x = var.as_float()[ok]
        perm = stats.Permuter(x, subj, params["permutations"], params["exhaustive_max"], params["seed"], key)
        P = perm.matrix().astype(float)
        r2o, fo = stats.permanova_numeric_rows(K, x[None, :])
        r2, F = stats.permanova_numeric_rows(K, P)
    else:
        vals = [var.values[i] for i in idx]
        levels = sorted(set(vals), key=str)
        codes = np.array([levels.index(v) for v in vals])
        perm = stats.Permuter(codes, subj, params["permutations"], params["exhaustive_max"], params["seed"], key)
        if perm.n_distinct < 2:
            return dict(row, status="insufficient_data", reason="no permutation is possible under the design")
        P = perm.matrix()
        r2o, fo = stats.permanova_rows(K, codes[None, :], len(levels))
        r2, F = stats.permanova_rows(K, P, len(levels))
    p, pmin = perm.p_value(np.nan_to_num(F, nan=-1.0), float(fo[0]))
    return dict(row, status="computed", statistic="pseudo_F", value=float(fo[0]), r2=float(r2o[0]), p=p,
                p_min_attainable=pmin, **perm.describe())



def _isnum(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _num(v):
    return float(v) if _isnum(v) else 0.0


def table(rows):
    """Adds BH q-values over the computed rows; -> (rows, n_tests)."""
    ps = [r.get("p") if r.get("status") == "computed" else None for r in rows]
    qs = stats.bh(ps)
    for r, q in zip(rows, qs):
        if q is not None:
            r["q"] = q
    return rows, sum(1 for p in ps if p is not None)


def variable_list(vs):
    return [v.describe() for v in vs]


__all__ = ["Finding", "association", "table", "Variable", "TEMPLATES", "clean"]
