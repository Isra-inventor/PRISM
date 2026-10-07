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

    def method(self, name, n_used=None, **params):
        self.d["method"].update(name=name, params=params)
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


def association(response, var, subject, params, key, guard_subjects=True):
    """One row of an association table: a per-sample response against a variable, Kruskal-Wallis
    (categorical) or |Spearman| (numeric), with restricted permutation p (v3 §6.1)."""
    response = np.asarray(response, float)
    vals = var.values
    ok = np.array([str(v).strip() != "" for v in vals]) & np.isfinite(response)
    row = {"variable": var.name, "role": var.role, "test": "kruskal_wallis" if not var.numeric else "spearman",
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
    ry = stats.rankdata(y)
    obs = stats.kruskal_h(ry, codes, len(levels))
    perm = stats.Permuter(codes, subj, params["permutations"], params["exhaustive_max"], params["seed"], key)
    if perm.n_distinct < 2:
        return dict(row, status="insufficient_data", reason="no permutation is possible under the design")
    P = perm.matrix()
    H = stats.kruskal_h_rows(ry, P, len(levels))
    p, pmin = perm.p_value(H, obs)
    return dict(row, status="computed", statistic="H", value=obs, p=p, p_min_attainable=pmin, **perm.describe())


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
