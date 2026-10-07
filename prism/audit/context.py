"""What one audit run sees (v3 §6.1): per dataset and audit unit, X (features x samples), the
diagnostic copy Y, F (feature table) and M (the unified sample table restricted to the
dataset's samples), plus the design (subject, time) and the variable sets B and G.

Read from the output folders and the merge decisions only; never from the uploads.
"""

from __future__ import annotations

import numpy as np

from ..io.loader import load_output
from ..session import merge

OUT_OF_SCOPE = {"count_like": "integer counts", "proportion_like": "values in [0, 1] (proportions)",
                "compositional_like": "constant sample sums (compositional)"}


def is_num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return False
    return x == x


# ------------------------------------------------------------------ scale class (A2 item 5)


def skew_rows(A):
    """Per-row sample skewness ignoring NaN (rows with < 3 values or SD 0 give NaN)."""
    out = np.full(A.shape[0], np.nan)
    ok = np.isfinite(A)
    n = ok.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(ok, A, 0).sum(1) / np.maximum(n, 1)
        d = np.where(ok, A - m[:, None], 0)
        m2 = (d ** 2).sum(1) / np.maximum(n, 1)
        m3 = (d ** 3).sum(1) / np.maximum(n, 1)
        s = m3 / m2 ** 1.5
    good = (n >= 3) & (m2 > 0)
    out[good] = s[good]
    return out


def scale_evidence(X):
    obs = X[np.isfinite(X)]
    ev = {"n_observed": int(obs.size)}
    if not obs.size:
        return "unknown", ev
    ev["all_non_negative"] = bool((obs >= 0).all())
    ev["all_integer"] = bool(np.all(obs == np.round(obs)))
    ev["all_in_0_1"] = bool(((obs >= 0) & (obs <= 1)).all())
    sums = np.nansum(X, axis=0)
    ev["cv_sample_sums"] = float(np.std(sums) / np.mean(sums)) if np.mean(sums) else None
    ev["negative_present"] = bool((obs < 0).any())
    sx = skew_rows(X)
    ev["median_skew_x"] = float(np.nanmedian(sx)) if np.isfinite(sx).any() else None
    if ev["all_non_negative"]:
        Y, rule = diagnostic_copy(X)
        sy = skew_rows(Y)
        ev["median_skew_y"] = float(np.nanmedian(sy)) if np.isfinite(sy).any() else None
    else:
        ev["median_skew_y"] = None
    if ev["all_non_negative"] and ev["all_integer"]:
        cls = "count_like"
    elif ev["all_in_0_1"]:
        cls = "proportion_like"
    elif ev["cv_sample_sums"] is not None and ev["cv_sample_sums"] <= 0.001:
        cls = "compositional_like"
    elif ev["negative_present"]:
        cls = "continuous_signed"
    elif ev["median_skew_y"] is not None and ev["median_skew_x"] is not None \
            and abs(ev["median_skew_y"]) < abs(ev["median_skew_x"]):
        cls = "continuous_linear"
    else:
        cls = "continuous_symmetric_or_log_like"
    return cls, ev


def diagnostic_copy(X):
    """Y = log2(X) if every observed X > 0, else log2(X + c), c = 0.5 x smallest positive value.
    Signed data are used as they are. Never written to any output."""
    obs = X[np.isfinite(X)]
    if obs.size and (obs < 0).any():
        return X.copy(), {"transform": "none", "reason": "negative values present", "c": None}
    if obs.size and (obs > 0).all():
        with np.errstate(divide="ignore"):
            return np.log2(X), {"transform": "log2", "c": None}
    pos = obs[obs > 0]
    c = 0.5 * float(pos.min()) if pos.size else 1.0
    return np.log2(X + c), {"transform": "log2(x + c)", "c": c}


# ------------------------------------------------------------------ variables


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

    def describe(self):
        return {"name": self.name, "role": self.role, "kind": self.kind, "source": self.source,
                "n_levels": len(self.levels())}


def _level_key(v):
    return (0, float(v), "") if is_num(v) else (1, 0.0, str(v))


# ------------------------------------------------------------------ units


class Unit:
    """One audit unit of one dataset: an assay (blocks concatenated when their feature keys are
    disjoint) or one block (blocks sharing feature keys are alternative measurements)."""

    def __init__(self, ds, assay, blocks, unit_id, label):
        self.ds, self.assay, self.unit_id, self.label = ds, assay, unit_id, label
        self.dataset_id = ds.dataset_id
        samples = blocks[0].sample_ids
        for b in blocks[1:]:
            if b.sample_ids != samples:   # samples-in-rows blocks share the sample axis; align by ID
                ix = [b.sample_ids.index(s) for s in samples]
                b.X, b.sample_ids = b.X[:, ix], samples
        self.all_samples = list(samples)
        self.feature_keys = [k for b in blocks for k in b.feature_keys]
        self.block_of = [b.block_id for b in blocks for _ in b.feature_keys]
        Xall = np.vstack([b.X for b in blocks]) if len(blocks) > 1 else blocks[0].X
        self.n_unparsable = sum(b.n_unparsable for b in blocks)
        keep = [i for i, s in enumerate(samples) if ds.in_audit(s)]
        self.excluded = [s for s in samples if not ds.in_audit(s) and s not in ds.excluded_by_override]
        self.excluded_by_override = [s for s in samples if s in ds.excluded_by_override]
        self.samples = [samples[i] for i in keep]
        self.X = Xall[:, keep]
        self.X_non_study = Xall[:, [i for i, s in enumerate(samples) if s in ds.non_study]]
        self.non_study_samples = [s for s in samples if s in ds.non_study]
        self.scale, self.scale_evidence = scale_evidence(self.X)
        self.out_of_scope = OUT_OF_SCOPE.get(self.scale)
        self._Y = None
        feats = ds.output.features_of(assay["assay_id"])
        self.F = [feats.get(k, {}) for k in self.feature_keys]
        self.strata, self.strata_source = self._strata(blocks)

    @property
    def Y(self):
        if self._Y is None:
            self._Y, self.y_rule = diagnostic_copy(self.X)
        return self._Y

    @property
    def transform(self):
        _ = self.Y
        return self.y_rule

    def _strata(self, blocks):
        der = [a for a in self.ds.schema.get("feature_annotations", [])
               if a.get("column") == "feature_class" and a.get("keep", True)]
        if der:
            labels = der[0].get("display_labels") or {}
            vals = [labels.get(f.get("feature_class", ""), f.get("feature_class", "")) or "(empty)" for f in self.F]
            return vals, "feature_class"
        labs = {b.block_id: b.label for b in blocks}
        return [labs[b] for b in self.block_of], "block"

    def var(self, v):
        """Values of a dataset variable for this unit's samples."""
        return [v[s] for s in self.samples]

    @property
    def n(self):
        return self.X.shape[1]

    @property
    def p(self):
        return self.X.shape[0]


class DatasetCtx:
    def __init__(self, st, d, merged, overrides, params):
        self.st, self.entry, self.params = st, d, params
        self.dataset_id = d["dataset_id"]
        self.output = load_output(st.output_dir(self.dataset_id))
        self.schema = self.output.schema
        view = next(v for v in merged["views"] if v.did == self.dataset_id)
        self.view = view
        # M: the unified table restricted to this dataset's samples, keyed by original sample ID
        header, rows = merge.unified_table(merged, only=self.dataset_id)
        key = f"sample_id@{self.dataset_id}"
        self.M = {r[header.index(key)]: dict(zip(header, r)) for r in rows}
        self.M_columns = header
        self.col_kind = {}
        for c in merged["columns"]:
            kinds = c["audit_kind"]
            if c["status"] in ("single", "merged", "taken"):
                self.col_kind[c["column"]] = kinds if isinstance(kinds, str) else next(iter(kinds.values()))
            elif c["status"] in ("kept_per_dataset", "conflict_open"):
                for did in c["datasets"]:
                    k = kinds.get(did) if isinstance(kinds, dict) else kinds
                    self.col_kind[f"{c['column']}@{did}"] = k
        ov = [o for o in overrides if o.get("dataset") in (None, "", self.dataset_id)]
        self.overrides = ov
        self.roles = {o["sample"]: o["role"] for o in ov if o["kind"] == "sample_role"}
        self.excluded_by_override = {o["sample"] for o in ov if o["kind"] == "exclude_from_audit"}
        self.non_study = {s for s in self.M if self.role(s) != "study"}
        self.subject = self._design_var(view.subject_col, "subject")
        self.time = self._design_var(view.time_col, "time")
        self.time_unit = view.time_unit
        self.units = self._units()

    def role(self, s):
        """qc / blank / pool / study from overrides; otherwise study, or 'unassigned' for a sample
        Step 0 marked as not a study sample (Step 0 records only is_study_sample)."""
        if s in self.roles:
            return self.roles[s]
        return "study" if (self.M.get(s) or {}).get("is_study_sample", "true") != "false" else "unassigned"

    def in_audit(self, s):
        return s not in self.excluded_by_override and s not in self.non_study

    def column(self, name):
        """{sample: value} of a column of M (a dataset column may carry an @dataset suffix)."""
        if name is None:
            return None
        for h in (f"{name}@{self.dataset_id}", name):
            if h in self.M_columns:
                return {s: (r.get(h) or "").strip() for s, r in self.M.items()}
        return None

    def _design_var(self, col, role):
        vals = self.column(col)
        if vals is None or not any(vals.values()):
            return None
        return {"column": col, "values": vals}

    def _units(self):
        out = []
        for a in self.output.assays:
            blocks = a["blocks"]
            if not blocks:
                continue
            keys = [set(b.feature_keys) for b in blocks]
            disjoint = all(not (keys[i] & keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys)))
            aid = a["meta"]["assay_id"]
            label = a["meta"].get("assay_label") or aid
            if disjoint:
                out.append(Unit(self, a["meta"], blocks, aid, label))
            else:
                for b in blocks:
                    out.append(Unit(self, a["meta"], [b], f"{aid}_{b.block_id}", f"{label} · {b.label}"))
        return out

    # -------------------------------------------------------------- variable sets

    def variables(self, unit, extra_batch=(), extra_design=()):
        """B (batch candidates) and G (design variables) for a unit, with the reasons for any
        column left out."""
        P = self.params
        B, G, skipped = [], [], []
        samples = unit.samples
        used = set()

        def vals_of(col):
            v = self.column(col)
            return [v.get(s, "") for s in samples] if v is not None else None

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

        ov_batch = {o["column"] for o in self.overrides if o["kind"] == "batch_variable"} | set(extra_batch)
        ov_design = {o["column"] for o in self.overrides if o["kind"] == "design_variable"} | set(extra_design)
        for col in self.M_columns:
            k = self.col_kind.get(col)
            if k in ("batch", "run_order") or col in ov_batch:
                vals = vals_of(col)
                v = typed(col, vals, "batch", "override" if col in ov_batch else "audit_kind " + str(k),
                          "numeric" if k == "run_order" else "categorical")
                if v:
                    B.append(v)
                used.add(col)
        if self.subject:
            vals = [self.subject["values"].get(s, "") for s in samples]
            G.append(Variable("subject", vals, "subject", "categorical", f"design ({self.subject['column']})"))
            used.add(self.subject["column"])
        if self.time:
            vals = [self.time["values"].get(s, "") for s in samples]
            nlev = len({x for x in vals if x})
            kind = "categorical" if nlev <= P["time_ordered_max_levels"] else "numeric"
            G.append(Variable("time", vals, "time", kind, f"design ({self.time['column']})"))
            used.add(self.time["column"])
        for col in self.M_columns:
            if col in used:
                continue
            k = self.col_kind.get(col)
            if col in ov_design:
                v = typed(col, vals_of(col), "design", "override")
            elif k in ("group", "sample_type"):
                v = typed(col, vals_of(col), k, f"audit_kind {k}", "categorical")
            elif k == "covariate":
                v = typed(col, vals_of(col), "covariate", "audit_kind covariate")
            else:
                continue
            if v:
                G.append(v)
        return B, G, skipped

    def subject_codes(self, unit):
        if not self.subject:
            return None
        return [self.subject["values"].get(s, "") for s in unit.samples]


def build(st, overrides, params, datasets=None):
    merged = merge.compute(st)
    out = []
    for d in st.confirmed():
        if datasets and d["dataset_id"] not in datasets:
            continue
        out.append(DatasetCtx(st, d, merged, overrides, params))
    return merged, out
