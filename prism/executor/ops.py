"""Phase 1 ops (Build Request v4, section 2.2). Each takes the child data (a copy), the resolved
parameters and the context, changes the data in place and returns details for the op record.
Ops read only their parameters and the data, so a replay from S0 is exact.
"""

from __future__ import annotations

import numpy as np

from . import OpSpec, REQUIRED, ExecutorError, register, stack, units, unstack
from ..audit.context import scale_evidence


def _check(name, ok, value, text):
    return {"name": name, "ok": bool(ok), "value": value, "text": text}


def _touch(blocks, keys=None):
    return {(b.assay_id, b.block_id): set(b.feature_keys if keys is None else [k for k in b.feature_keys if k in keys])
            for b in blocks}


def _merge_touch(*ts):
    out = {}
    for t in ts:
        for k, v in t.items():
            out.setdefault(k, set()).update(v)
    return out


# ------------------------------------------------------------------ remove_features / remove_samples


def _known_features(data, assay):
    return {k for b in data.blocks if not assay or b.assay_id == assay for k in b.feature_keys}


def pre_remove_features(data, p, ctx):
    ids = list(p["ids"] or [])
    unknown = sorted(set(ids) - _known_features(data, p.get("assay")))
    return [_check("ids_known", not unknown, unknown[:10], "every feature id exists in the state"),
            _check("ids_given", bool(ids), len(ids), "at least one feature id"),
            _check("reason_given", bool(str(p["reason"]).strip()), p["reason"], "a reason is recorded")]


def op_remove_features(data, p, ctx):
    ids = set(p["ids"])
    for b in data.blocks:
        if p.get("assay") and b.assay_id != p["assay"]:
            continue
        keep = [i for i, k in enumerate(b.feature_keys) if k not in ids]
        b.X = b.X[keep]
        b.feature_keys = [b.feature_keys[i] for i in keep]
    rows = data.features["rows"]
    data.features["rows"] = [r for r in rows if not (r.get("feature_key") in ids and
                                                     (not p.get("assay") or r.get("assay_id") == p["assay"]))]
    return {"removed": sorted(ids), "reason": p["reason"]}


register(OpSpec("remove_features", op_remove_features, ("G3", "S3"), {"ids": REQUIRED, "reason": REQUIRED},
                pre_remove_features, changes_shape=True, doc="Drop features (rows). Changes no value."))


def pre_remove_samples(data, p, ctx):
    ids = list(p["ids"] or [])
    have = {s for b in data.blocks for s in b.sample_ids}
    unknown = sorted(set(ids) - have)
    left = len(have - set(ids))
    return [_check("ids_known", not unknown, unknown[:10], "every sample id exists in the state"),
            _check("ids_given", bool(ids), len(ids), "at least one sample id"),
            _check("enough_left", left >= 3, left, "at least 3 samples remain"),
            _check("reason_given", bool(str(p["reason"]).strip()), p["reason"], "a reason is recorded")]


def op_remove_samples(data, p, ctx):
    ids = set(p["ids"])
    for b in data.blocks:
        keep = [j for j, s in enumerate(b.sample_ids) if s not in ids]
        b.X = b.X[:, keep]
        b.sample_ids = [b.sample_ids[j] for j in keep]
    data.samples["rows"] = [r for r in data.samples["rows"] if r.get("sample_id") not in ids]
    return {"removed": sorted(ids), "reason": p["reason"]}


register(OpSpec("remove_samples", op_remove_samples, ("S4",), {"ids": REQUIRED, "reason": REQUIRED},
                pre_remove_samples, changes_shape=True,
                doc="Drop samples (columns); each sample needs its own approval. Changes no value."))


# ------------------------------------------------------------------ recode_zero_as_missing (G7)


def pre_recode_zero(data, p, ctx):
    n0 = sum(int((b.X == 0).sum()) for g in units(data, p.get("assay")) for b in g)
    return [_check("zeros_present", n0 > 0, n0, "the assay has zero cells"),
            _check("answer_recorded", p["answer"] == "yes", p["answer"],
                   "the user answered 'zeros represent missing values' with yes")]


def op_recode_zero(data, p, ctx):
    n = 0
    t = {}
    for g in units(data, p.get("assay")):
        for b in g:
            z = b.X == 0
            n += int(z.sum())
            b.X = np.where(z, np.nan, b.X)
            t.update(_touch([b]))
    return {"cells_recoded": n, "_touched": t}


register(OpSpec("recode_zero_as_missing", op_recode_zero, ("G7",), {"answer": REQUIRED}, pre_recode_zero,
                nan_exception="zero_to_nan",
                doc="Documented NaN exception: exact zeros become NaN after the user says zeros mean missing. "
                    "No other value changes."))


# ------------------------------------------------------------------ log_transform (G5)

OFFSET_RULES = ("none", "half_min", "fixed", "proBatch_compat")


def _offset(X, rule, c):
    obs = X[np.isfinite(X)]
    if rule == "none":
        return 0.0
    if rule == "half_min":
        pos = obs[obs > 0]
        return 0.5 * float(pos.min()) if pos.size else 1.0
    if rule == "fixed":
        return float(c)
    return 1.0   # proBatch_compat: log2(x + 1), oracle tests only


def pre_log(data, p, ctx):
    out = [_check("offset_rule", p["offset_rule"] in OFFSET_RULES, p["offset_rule"],
                  f"offset rule is one of {', '.join(OFFSET_RULES)}"),
           _check("base", p["base"] in (2, 10, "e"), p["base"], "base is 2, 10 or e"),
           _check("not_already_log", data.scale not in ("log2", "arcsinh"), data.scale,
                  "the state is not already log or arcsinh (a double transform is a documented error)"),
           _check("confirmed_linear", data.scale == "linear", data.scale,
                  "the confirmed processing state says log_transformed = no")]
    if p["offset_rule"] == "fixed":
        out.append(_check("offset_value", p["offset"] is not None and float(p["offset"]) > 0, p["offset"],
                          "a fixed offset c > 0 is given"))
    for g in units(data, p.get("assay")):
        X = stack(g)
        cls, _ = scale_evidence(X)
        obs = X[np.isfinite(X)]
        label = f"{g[0].assay_id}"
        out.append(_check(f"scale_class:{label}", cls == "continuous_linear", cls, "A2 scale class continuous_linear"))
        out.append(_check(f"non_negative:{label}", not (obs < 0).any(), int((obs < 0).sum()), "no negative values"))
        if p["offset_rule"] == "none":
            out.append(_check(f"all_positive:{label}", (obs > 0).all(), int((obs <= 0).sum()),
                              "offset rule 'none' needs every observed value > 0"))
    return out


def op_log(data, p, ctx):
    used = {}
    log = {2: np.log2, 10: np.log10, "e": np.log}[p["base"]]
    t = {}
    for g in units(data, p.get("assay")):
        X = stack(g)
        c = _offset(X, p["offset_rule"], p.get("offset"))
        unstack(g, log(X + c))
        used[g[0].assay_id] = c
        t.update(_touch(g))
    return {"offset_used": used, "rule": p["offset_rule"], "_touched": t}


register(OpSpec("log_transform", op_log, ("G5",), {"base": 2, "offset_rule": "half_min", "offset": None}, pre_log,
                scale_after="log2", doc="log_base(x + c); offset rules none, half_min (0.5 x smallest positive "
                                        "observed value), fixed(c), proBatch_compat (c = 1, oracle tests only)."))


# ------------------------------------------------------------------ normalize_median (S1)


def pre_log_scale(data, p, ctx):
    return [_check("log_scale", data.scale in ("log2",), data.scale, "the state is on a log2 scale (after G5 or confirmed)")]


def op_normalize_median(data, p, ctx):
    t = {}
    shifts = {}
    for g in units(data, p.get("assay")):
        X = stack(g)
        with np.errstate(all="ignore"):
            med = np.nanmedian(X, axis=0)
        glob = float(np.nanmedian(X))
        if not np.isfinite(med).all():
            raise ExecutorError("normalize_median: a sample has no observed value.")
        unstack(g, X - med[None, :] + glob)
        shifts[g[0].assay_id] = {"global_median": glob, "sample_shift": dict(zip(g[0].sample_ids, (glob - med).tolist()))}
        t.update(_touch(g))
    return {"shifts": shifts, "_touched": t}


register(OpSpec("normalize_median", op_normalize_median, ("S1",), {}, pre_log_scale,
                doc="x - median(sample) + median(all observed); NaN stays NaN."))


# ------------------------------------------------------------------ arcsinh_transform (G5 alternative)


def pre_arcsinh(data, p, ctx):
    c = p.get("cofactor")
    return [_check("cofactor_given", c is not None and float(c) > 0, c, "a user-supplied cofactor > 0 (no default)"),
            _check("justification", bool(str(p.get("justification") or "").strip()), p.get("justification"),
                   "a justification for the cofactor is recorded"),
            _check("not_already_log", data.scale not in ("log2", "arcsinh"), data.scale,
                   "the state is not already log or arcsinh"),
            _check("confirmed_linear", data.scale == "linear", data.scale,
                   "the confirmed processing state says log_transformed = no")]


def op_arcsinh(data, p, ctx):
    c = float(p["cofactor"])
    t = {}
    for g in units(data, p.get("assay")):
        unstack(g, np.arcsinh(stack(g) / c))
        t.update(_touch(g))
    return {"cofactor": c, "_touched": t}


register(OpSpec("arcsinh_transform", op_arcsinh, ("G5",), {"cofactor": REQUIRED, "justification": REQUIRED},
                pre_arcsinh, scale_after="arcsinh", doc="arcsinh(x / cofactor); the cofactor needs a justification."))


# ------------------------------------------------------------------ helpers for sample-level parameters


def _sample_values(g, mapping):
    """Values of a {sample: value} mapping for a unit's samples ('' when absent)."""
    return [str(mapping.get(s, "")) if mapping.get(s) is not None else "" for s in g[0].sample_ids]


def _orders(g, mapping):
    out = []
    for s in g[0].sample_ids:
        try:
            out.append(float(mapping.get(s)))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.array(out)


# ------------------------------------------------------------------ drift_qc_loess (G6, global)


def pre_drift_qc(data, p, ctx):
    out = [_check("log_scale", data.scale in ("log2", "arcsinh"), data.scale,
                  "the state is on a log (or arcsinh) scale")]
    for g in units(data, p.get("assay")):
        ro = _orders(g, p["run_order"])
        qc = [j for j, s in enumerate(g[0].sample_ids) if s in set(p["qc_samples"])]
        out.append(_check(f"run_order_present:{g[0].assay_id}", np.isfinite(ro).all(), int((~np.isfinite(ro)).sum()),
                          "run order present for every sample"))
        out.append(_check(f"qc_n:{g[0].assay_id}", len(qc) >= p["min_qc"], len(qc),
                          f"at least {p['min_qc']} QC samples (heuristic)"))
        if qc and np.isfinite(ro).any():
            fin = ro[np.isfinite(ro)]
            lo, hi = np.quantile(fin, [0.1, 0.9])
            qo = ro[qc]
            out.append(_check(f"qc_span:{g[0].assay_id}", bool(np.nanmin(qo) <= lo and np.nanmax(qo) >= hi),
                              [float(np.nanmin(qo)), float(np.nanmax(qo))],
                              "QC injections span the run order (first QC at or before the 10th percentile, last at "
                              "or after the 90th)"))
    return out


def op_drift_qc(data, p, ctx):
    from ..batch import loess
    t, info = {}, {"fitted": 0, "left_unchanged": 0, "engines": set()}
    qcs = set(p["qc_samples"])
    for g in units(data, p.get("assay")):
        X = stack(g)
        ro = _orders(g, p["run_order"])
        qc = np.array([s in qcs for s in g[0].sample_ids])
        out = X.copy()
        touched = []
        keys = [k for b in g for k in b.feature_keys]
        for i in range(X.shape[0]):
            ok = qc & np.isfinite(X[i])
            if ok.sum() < p["min_qc_per_feature"]:
                info["left_unchanged"] += 1
                continue
            try:
                fit, eng = loess.fit_predict(ro[ok], X[i, ok], ro, span=p["span"], degree=p["degree"])
            except Exception:
                info["left_unchanged"] += 1
                continue
            info["engines"].add(eng)
            out[i] = X[i] + (fit[ok].mean() - fit)
            info["fitted"] += 1
            touched.append(keys[i])
        unstack(g, out)
        t = _merge_touch(t, _touch(g, set(touched)))
    info["engines"] = sorted(info["engines"])
    info["_touched"] = t
    info["evidence_level"] = "unverified"
    return info


register(OpSpec("drift_qc_loess", op_drift_qc, ("G6",),
                {"qc_samples": REQUIRED, "run_order": REQUIRED, "span": 0.75, "degree": 2, "min_qc": 5,
                 "min_qc_per_feature": 5},
                pre_drift_qc, doc="Per feature, LOESS of QC values against run order fitted on the QC injections only, "
                                  "removed from every sample as x + (mean QC fit - fit). Features with fewer than 5 "
                                  "observed QC values are left unchanged and counted. Evidence: unverified (Dunn et "
                                  "al. 2011 not checked)."))


# ------------------------------------------------------------------ normalize_quantile (S1)


def op_normalize_quantile(data, p, ctx):
    from ..batch import normalize
    t, forms = {}, {}
    for g in units(data, p.get("assay")):
        out, info = normalize.quantile(stack(g))
        unstack(g, out)
        forms[g[0].assay_id] = info["form"]
        t.update(_touch(g))
    return {"form": forms, "_touched": t}


register(OpSpec("normalize_quantile", op_normalize_quantile, ("S1",), {}, pre_log_scale,
                doc="Quantile normalization; NA-aware form with missing values (differs from preprocessCore)."))


# ------------------------------------------------------------------ S2 batch correction through the backend


def _backend_or_refuse():
    from ..batch import backend
    b = backend.get()
    if b.name == "none":
        raise ExecutorError("This op needs the proBatch port, which is disabled (PRISM_DISABLE_PORTS=1).")
    return b


def pre_batch(data, p, ctx):
    out = pre_log_scale(data, p, ctx)
    for g in units(data, p.get("assay")):
        bv = _sample_values(g, p["batch"])
        levels = sorted({v for v in bv if v})
        out.append(_check(f"batch_levels:{g[0].assay_id}", len(levels) >= 2, len(levels), "at least 2 batch levels"))
    return out


def _with_batch(g, p):
    """-> (stacked X, indices of samples with a batch value, their batch labels)."""
    bv = _sample_values(g, p["batch"])
    idx = [j for j, v in enumerate(bv) if v]
    return stack(g), idx, [bv[j] for j in idx]


def op_batch_center(data, p, ctx):
    be = _backend_or_refuse()
    t, info = {}, {"backend": be.name, "source": be.source, "stat": p["stat"]}
    for g in units(data, p.get("assay")):
        X, idx, labels = _with_batch(g, p)
        sub, det = be.center(X[:, idx], labels, p["stat"])
        X = X.copy()
        X[:, idx] = sub
        unstack(g, X)
        info.setdefault("per_assay", {})[g[0].assay_id] = dict(det, samples_without_batch=len(g[0].sample_ids) - len(idx))
        t.update(_touch(g))
    info["_touched"] = t
    return info


register(OpSpec("batch_center", op_batch_center, ("S2",), {"batch": REQUIRED, "stat": "median"}, pre_batch,
                doc="Per feature, per batch: x - stat(batch) + stat(all) (median or mean); NaN tolerant."))


def pre_combat(data, p, ctx):
    out = pre_batch(data, p, ctx)
    for g in units(data, p.get("assay")):
        X, idx, labels = _with_batch(g, p)
        Xs = X[:, idx]
        n_nan = int(np.isnan(Xs).sum())
        complete = float(np.isfinite(Xs).all(1).mean()) if Xs.size else 0.0
        out.append(_check(f"no_missing:{g[0].assay_id}", n_nan == 0, {"nan_cells": n_nan, "share_complete_features":
                                                                      complete},
                          "no missing values in the features ComBat corrects (share of complete features 100%)"))
        counts = {lv: labels.count(lv) for lv in set(labels)}
        out.append(_check(f"two_per_batch:{g[0].assay_id}", bool(counts) and min(counts.values()) >= 2, counts,
                          "at least 2 samples in every batch"))
    return out


def op_batch_combat(data, p, ctx):
    be = _backend_or_refuse()
    t, info = {}, {"backend": be.name, "source": be.source}
    for g in units(data, p.get("assay")):
        X, idx, labels = _with_batch(g, p)
        sub, det = be.combat(X[:, idx], labels)
        X = X.copy()
        X[:, idx] = sub
        unstack(g, X)
        info.setdefault("per_assay", {})[g[0].assay_id] = det
        t.update(_touch(g))
    info["_touched"] = t
    return info


register(OpSpec("batch_combat", op_batch_combat, ("S2",), {"batch": REQUIRED}, pre_combat,
                doc="ComBat, parametric, intercept-only model; refuses any missing value."))


def pre_drift_sample(data, p, ctx):
    out = pre_batch(data, p, ctx)
    for g in units(data, p.get("assay")):
        ro = _orders(g, p["run_order"])
        out.append(_check(f"run_order:{g[0].assay_id}", np.isfinite(ro).sum() >= p["min_measurements"],
                          int(np.isfinite(ro).sum()), f"run order for at least {p['min_measurements']} samples"))
    return out


def op_drift_sample(data, p, ctx):
    be = _backend_or_refuse()
    t, info = {}, {"backend": be.name, "source": be.source}
    for g in units(data, p.get("assay")):
        X = stack(g)
        bv = _sample_values(g, p["batch"])
        ro = _orders(g, p["run_order"])
        idx = [j for j, v in enumerate(bv) if v and np.isfinite(ro[j])]
        sub, det = be.drift_loess(X[:, idx], [bv[j] for j in idx], ro[idx], p["span"], p["degree"], p["min_measurements"])
        touched = det.pop("touched")
        X = X.copy()
        X[:, idx] = sub
        unstack(g, X)
        keys = [k for b in g for k in b.feature_keys]
        t = _merge_touch(t, _touch(g, {keys[i] for i in np.where(touched)[0]}))
        info.setdefault("per_assay", {})[g[0].assay_id] = det
    info["_touched"] = t
    return info


register(OpSpec("drift_sample_loess", op_drift_sample, ("S2",),
                {"batch": REQUIRED, "run_order": REQUIRED, "span": 0.75, "degree": 2, "min_measurements": 8},
                pre_drift_sample, doc="proBatch: per feature, per batch, LOESS of study-sample values against run "
                                      "order; value + (mean fit - fit). Fits that fail leave the feature unchanged."))
