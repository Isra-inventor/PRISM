"""The executor (Build Request v4, section 2.2): one function per op, apply(state, params) ->
(new_state, op_record). Phase 1 ops only. **There is no impute op.**

Invariants, checked after every op before anything is written (a violation refuses the op):
- the NaN pattern of the cells that remain is identical before and after, except for the one
  documented exception (`recode_zero_as_missing` turns exact zeros into NaN, nothing else);
- no op turns NaN into a number or a number into NaN (same exception);
- the matrix shape changes only for remove_features / remove_samples, and those change no value;
- values change only inside the features an op says it touched.

Preconditions are checked before running; a failing one refuses the op with the precondition
in the message. Nothing runs without an approval record (user, timestamp, plan entry id).
"""

from __future__ import annotations

import platform

import numpy as np

from ..util import canonical_json, now_iso, sha256_bytes
from . import state as state_mod


class ExecutorError(Exception):
    pass


class OpSpec:
    def __init__(self, name, fn, steps, params, preconditions=None, changes_shape=False, nan_exception=None,
                 scale_after=None, doc=""):
        self.name, self.fn, self.steps, self.params = name, fn, tuple(steps), dict(params)
        self.preconditions = preconditions
        self.changes_shape, self.nan_exception, self.scale_after, self.doc = changes_shape, nan_exception, scale_after, doc


OPS = {}


def register(spec):
    if "imput" in spec.name.lower():
        raise ExecutorError("Phase 1 has no imputation op (v4 section 2.2).")
    OPS[spec.name] = spec
    return spec


def op_names():
    return sorted(OPS)


def package_versions():
    out = {"python": platform.python_version(), "numpy": np.__version__}
    for mod in ("scipy", "skmisc", "statsmodels"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return out


class OpContext:
    """What an op may know besides the data: the state's meta and resolved session facts."""

    def __init__(self, st=None, did=None, meta=None, facts=None):
        self.st, self.did, self.meta, self.facts = st, did, meta or {}, facts or {}


# ------------------------------------------------------------------ groups of blocks (units)


def units(data, assay=None):
    """Blocks grouped like the audit's units: an assay whose blocks have disjoint feature keys is
    one unit (blocks share the samples); blocks that share keys are separate units."""
    out = []
    aids = []
    for b in data.blocks:
        if b.assay_id not in aids:
            aids.append(b.assay_id)
    for aid in aids:
        if assay and aid != assay:
            continue
        bl = [b for b in data.blocks if b.assay_id == aid]
        keys = [set(b.feature_keys) for b in bl]
        disjoint = all(not (keys[i] & keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys)))
        same_samples = all(b.sample_ids == bl[0].sample_ids for b in bl)
        if disjoint and same_samples:
            out.append(bl)
        else:
            out += [[b] for b in bl]
    if assay and not out:
        raise ExecutorError(f"Unknown assay '{assay}'.")
    return out


def stack(blocks):
    return np.vstack([b.X for b in blocks]) if len(blocks) > 1 else blocks[0].X


def unstack(blocks, X):
    i = 0
    for b in blocks:
        n = len(b.feature_keys)
        b.X = X[i:i + n].copy()
        i += n


# ------------------------------------------------------------------ invariants


def _aligned(pb, cb):
    fk = set(cb.feature_keys)
    sk = set(cb.sample_ids)
    ri = [i for i, k in enumerate(pb.feature_keys) if k in fk]
    ci = [j for j, s in enumerate(pb.sample_ids) if s in sk]
    A = pb.X[np.ix_(ri, ci)]
    ix = {k: i for i, k in enumerate(cb.feature_keys)}
    jx = {s: j for j, s in enumerate(cb.sample_ids)}
    B = cb.X[np.ix_([ix[pb.feature_keys[i]] for i in ri], [jx[pb.sample_ids[j]] for j in ci])]
    return A, B, [pb.feature_keys[i] for i in ri]


def check_invariants(spec, parent, child, touched=None):
    """-> counts; raises ExecutorError on a violation."""
    cells_changed = 0
    by = {(b.assay_id, b.block_id): b for b in child.blocks}
    for pb in parent.blocks:
        cb = by.get((pb.assay_id, pb.block_id))
        if cb is None:
            raise ExecutorError(f"{spec.name}: block {pb.assay_id}/{pb.block_id} disappeared.")
        if not spec.changes_shape and (cb.feature_keys != pb.feature_keys or cb.sample_ids != pb.sample_ids):
            raise ExecutorError(f"{spec.name}: the matrix shape or order changed; only remove_* may do that.")
        if set(cb.feature_keys) - set(pb.feature_keys) or set(cb.sample_ids) - set(pb.sample_ids):
            raise ExecutorError(f"{spec.name}: new features or samples appeared.")
        A, B, keys = _aligned(pb, cb)
        na, nb = np.isnan(A), np.isnan(B)
        if spec.nan_exception == "zero_to_nan":
            bad = (na & ~nb) | (nb & ~na & (A != 0))
        else:
            bad = na != nb
        if bad.any():
            raise ExecutorError(f"{spec.name}: the NaN pattern changed in {int(bad.sum())} cell(s) "
                                f"({pb.assay_id}/{pb.block_id}); no op may fill or create missing values.")
        if np.isinf(B).any() and not np.isinf(A).any():
            raise ExecutorError(f"{spec.name}: infinite values were produced.")
        both = ~na & ~nb
        ch = both & (A != B)
        if spec.changes_shape and ch.any():
            raise ExecutorError(f"{spec.name}: a removal op changed values.")
        if touched is not None:
            t = touched.get((pb.assay_id, pb.block_id))
            allowed = np.zeros(len(keys), bool) if t is None else np.array([k in t for k in keys])
            if (ch & ~allowed[:, None]).any():
                raise ExecutorError(f"{spec.name}: values changed outside the features the op touched.")
        cells_changed += int(ch.sum())
    return cells_changed


# ------------------------------------------------------------------ apply


def _validate_approval(approval):
    if not isinstance(approval, dict) or not approval.get("user") or not approval.get("at") \
            or not approval.get("plan_entry"):
        raise ExecutorError("Nothing runs without approval: an approval record needs user, at (timestamp) and "
                            "plan_entry.")


def _params(spec, params):
    params = dict(params or {})
    unknown = sorted(set(params) - set(spec.params) - {"assay"})
    if unknown:
        raise ExecutorError(f"{spec.name}: unknown parameter(s) {', '.join(unknown)}.")
    out = {k: v for k, v in spec.params.items()}
    out.update(params)
    missing = [k for k, v in out.items() if v is REQUIRED]
    if missing:
        raise ExecutorError(f"{spec.name}: missing parameter(s) {', '.join(missing)}.")
    return out


class _Required:
    def __repr__(self):
        return "REQUIRED"


REQUIRED = _Required()


def run_op(parent_data, parent_meta, op, params, approval, step=None, ctx=None):
    """Run one op in memory. -> (child data, op_record, child state id). Writes nothing."""
    spec = OPS.get(op)
    if spec is None:
        raise ExecutorError(f"Unknown op '{op}'. Phase 1 ops: {', '.join(op_names())}.")
    _validate_approval(approval)
    step = step or spec.steps[0]
    if step not in spec.steps:
        raise ExecutorError(f"{op} belongs to step(s) {', '.join(spec.steps)}, not {step}.")
    p = _params(spec, params)
    ctx = ctx or OpContext(meta=parent_meta)
    checks = spec.preconditions(parent_data, p, ctx) if spec.preconditions else []
    failed = [c for c in checks if not c["ok"]]
    if failed:
        raise ExecutorError(f"{op} refused: precondition not met: " +
                            "; ".join(f"{c['text']} (found: {c.get('value')})" for c in failed))
    child = parent_data.copy()
    details = spec.fn(child, p, ctx) or {}
    touched = details.pop("_touched", None)
    if spec.scale_after:
        child.scale = spec.scale_after
    cells = check_invariants(spec, parent_data, child, touched)
    pc, cc = parent_data.counts(), child.counts()
    rec = {"op": op, "step": step, "params": p, "preconditions": checks, "package_versions": package_versions(),
           "seed": p.get("seed", 0), "approval": dict(approval), "cells_changed": cells,
           "nan_before": pc["n_nan"], "nan_after": cc["n_nan"],
           "features_removed": pc["n_features"] - cc["n_features"], "samples_removed": pc["n_samples"] - cc["n_samples"],
           "details": details, "executed_at": now_iso()}
    rec = _jsonable(rec)
    hashes = {b.file: sha256_bytes(state_mod.matrix_bytes(b)) for b in child.blocks}
    sid = state_mod.compute_id(parent_meta["state_id"], rec, hashes)
    return child, rec, sid


def _jsonable(obj):
    from ..audit.finding import clean
    return clean(obj)


def apply(st, did, op, params, approval, parent=None, step=None, facts=None, who="user"):
    """Apply an approved op to a state (default: the dataset's HEAD) and write the child state.
    -> the child's state meta (with op_record)."""
    parent = parent or state_mod.head(st, did)
    if parent == state_mod.root_id(st, did):
        state_mod.ensure_root(st, did)
    data, pm = state_mod.load_data(st, did, parent)
    ctx = OpContext(st, did, pm, facts)
    child, rec, sid = run_op(data, pm, op, params, approval, step, ctx)
    m = state_mod.write_state(st, did, pm, child, rec)
    if state_mod.head(st, did) == parent:
        state_mod.set_head(st, did, m["state_id"])
    from ..ledger import append_jsonl
    append_jsonl(st.dir / "ledger.jsonl", {"tool": f"executor.{op}", "dataset": did, "parent": parent,
                                           "state_id": m["state_id"], "who": who, "at": now_iso(),
                                           "approval": approval, "op_record_sha256":
                                           sha256_bytes(canonical_json(rec).encode("utf-8"))})
    st.log("op_applied", {"dataset": did, "op": op, "parent": parent, "state_id": m["state_id"],
                          "plan_entry": approval.get("plan_entry")})
    return m


def replay(st, did, state_id=None):
    """Re-apply the approved ops from S0 in memory and compare the final state id."""
    target = state_id or state_mod.head(st, did)
    recs = state_mod.chain(st, did, target)
    rid = state_mod.root_id(st, did)
    data = state_mod.StateData.load(st.output_dir(did), state_mod.root_scale(st, did))
    m = {"state_id": rid, "applied_ops": []}
    steps = []
    for r in recs:
        ctx = OpContext(st, did, m)
        data, rec, sid = run_op(data, m, r["op"], r["params"], r["approval"], r.get("step"), ctx)
        m = {"state_id": sid, "applied_ops": m["applied_ops"] + [{"op": r["op"], "step": r.get("step")}]}
        steps.append({"op": r["op"], "state_id": sid})
    return {"target": target, "replayed": m["state_id"], "identical": m["state_id"] == target, "steps": steps}


from . import ops  # noqa: E402,F401  (registers the Phase 1 ops)
