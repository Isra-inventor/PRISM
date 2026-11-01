"""The bridge to the audits ported from the v4 build (A5, A7, A8, A11, G3, verification).

Those audits were written against a "unit" (one assay of one dataset) and a "context" (the merged
sample table of a session). The package replaced both with Audit and Schema; this module rebuilds the
few pieces the ported code reads, so its statistics stay exactly as they were:

    Unit             per block: X (audit set), Y (diagnostic copy), samples, feature keys, ...
    Variable         one sample variable with its role (batch, subject, time, group, covariate)
    variables(...)   B = batch candidates (audit_kind batch, run_order), G = design variables
                     (subject, time, group, sample_type, covariate), and the variables left out
    association, permanova, guard, table   the v3 section 6.1 tests on Variables
    TEMPLATES        the sentences of the ported audits (numbers filled in; careful wording)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import stats


def is_num(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _level_key(v):
    return (0, float(v), "") if is_num(v) else (1, 0.0, str(v))


class Variable:
    """A sample-level variable aligned to the unit's samples ('' = missing)."""

    def __init__(self, name, values, role, kind, source):
        self.name, self.values, self.role, self.kind, self.source = name, list(values), role, kind, source

    @property
    def numeric(self):
        return self.kind == "numeric"

    def present(self):
        return np.array([str(v).strip() != "" for v in self.values])

    def levels(self):
        return sorted({v for v in self.values if str(v).strip() != ""}, key=_level_key)

    def as_float(self):
        return np.array([float(v) if is_num(v) else np.nan for v in self.values])

    def subset(self, idx):
        return Variable(self.name, [self.values[j] for j in idx], self.role, self.kind, self.source)

    def describe(self):
        return {"name": self.name, "role": self.role, "kind": self.kind, "source": self.source,
                "n_levels": len(self.levels())}


@dataclass
class Unit:
    """One block of the audit set, as the ported audits expect it."""
    block: str
    X: np.ndarray                       # proteins x study samples, values as in the file
    Y: np.ndarray                       # the diagnostic copy (log2 only for linear data)
    samples: list
    feature_keys: list
    transform: dict
    scale: str
    strata: list = field(default_factory=list)
    strata_source: str = "block"

    @property
    def n(self):
        return self.X.shape[1]

    @property
    def p(self):
        return self.X.shape[0]


def unit(audit, block: str) -> Unit:
    Xd = audit.audit_matrix(block)
    Yd, info = audit.diagnostic(block)
    return Unit(block=block, X=Xd.to_numpy(dtype=float), Y=Yd.to_numpy(), samples=list(Xd.columns),
                feature_keys=[str(k) for k in Xd.index], transform=info, scale=audit.results["scale"][block],
                strata=[block] * Xd.shape[0])


def _vals(series, samples):
    return series.reindex(samples).fillna("").astype(str).str.strip().tolist()


def variables(audit, samples) -> tuple[list, list, list]:
    """B (batch candidates) and G (design variables) for these samples, and the ones left out."""
    P = audit.params
    B, G, skipped = [], [], []
    allv = audit.variables()

    def typed(name, vals, role, source, force=None):
        levels = {x for x in vals if x != ""}
        if force:
            kind = force
        elif levels and all(is_num(x) for x in levels) and len(levels) > 2:
            kind = "numeric"
        else:
            kind = "categorical"
        if kind == "categorical" and role != "subject" and not (2 <= len(levels) <= P["categorical_max_levels"]):
            skipped.append({"variable": name, "reason": f"{len(levels)} level(s): categorical variables need 2 to "
                                                        f"{P['categorical_max_levels']}"})
            return None
        return Variable(name, vals, role, kind, source)

    for name, (kind, series) in allv.items():
        if kind in ("batch", "run_order"):
            v = typed(name, _vals(series, samples), "batch", f"audit_kind {kind}",
                      "numeric" if kind == "run_order" else "categorical")
            if v:
                B.append(v)
    if "subject" in allv:
        G.append(Variable("subject", _vals(allv["subject"][1], samples), "subject", "categorical", "design"))
    if "time" in allv:
        vals = _vals(allv["time"][1], samples)
        nlev = len({x for x in vals if x})
        G.append(Variable("time", vals, "time", "categorical" if nlev <= P["time_ordered_max_levels"] else "numeric",
                          "design"))
    for name, (kind, series) in allv.items():
        if name in ("subject", "time") or kind in ("batch", "run_order", "subject_id", "timepoint", "other",
                                                  "technical_replicate"):
            continue
        if kind in ("group", "sample_type"):
            v = typed(name, _vals(series, samples), kind, f"audit_kind {kind}", "categorical")
        elif kind == "covariate":
            v = typed(name, _vals(series, samples), "covariate", "audit_kind covariate")
        else:
            continue
        if v:
            G.append(v)
    return B, G, skipped


OUT_OF_SCOPE = ("count_like", "proportion_like", "compositional_like")


def per_block(audit, f, fn):
    """Run fn(unit, sub_finding) for every block; measures go under the block id, sentences get the
    block id in front. Blocks whose data are out of scope (A2) are not_applicable."""
    from .finding import Finding
    for bid in audit.schema.block_map():
        u = unit(audit, bid)
        sub = Finding(f.check, f.title, step=f.step)
        if u.scale in OUT_OF_SCOPE:
            sub.say("not_applicable", f"{u.scale.replace('_', '-')} data are outside what PRISM audits.")
        else:
            fn(u, sub)
        f.measures[bid] = sub.measures
        for st, text in sub.sentences:
            f.say(st, f"{bid}: {text}")
        if sub.method:
            f.method = sub.method
    return f


def subject_codes(audit, samples):
    allv = audit.variables()
    return _vals(allv["subject"][1], samples) if "subject" in allv else None


def column_values(audit, samples) -> dict:
    """Every sample information column (with its audit_kind), as text per sample."""
    return {name: (kind, _vals(series, samples)) for name, (kind, series) in audit.variables().items()}



# ------------------------------------------------------------------ v3 section 6.1 tests on Variables


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



# ------------------------------------------------------------------ sentence templates (ported)

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
    # A12
    "missingness_pattern": "Missingness pattern: {pclass} (intensity dependence {intensity}, batch association "
                           "{batch}, floor pattern {floor}). Pattern quantified, plausible mechanisms assessed.",
    # A13
    "processing_state_mismatch": "Declared {item}: {declared}, but the evidence does not match (confirmed: {confirmed}).",
    # v4 section 3 (A7, A11)
    "diagnostic_conclusion_depends_on_policy": "{what} is not the same under every diagnostic policy ({n} variants "
                                               "computed): the conclusion depends on how missing and floor values "
                                               "are handled. No variant is preferred.",
    "small_batches": "{n} level(s) of '{variable}' have fewer than {cut} samples (a heuristic suggestion, unverified): "
                     "correction may overfit small batches.",
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


def fmt(v):
    if isinstance(v, float):
        if v != v:
            return "n/a"
        if abs(v) >= 100:
            return f"{v:,.0f}"
        return f"{v:.3g}"
    if isinstance(v, int) and abs(v) >= 10000:
        return f"{v:,}"
    return str(v)
