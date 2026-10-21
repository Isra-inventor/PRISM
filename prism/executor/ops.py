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
