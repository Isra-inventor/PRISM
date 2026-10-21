"""Immutable data states (Build Request v4, section 2.1).

    sessions/<sid>/datasets/<did>/states/<state_id>/
      schema.json, value_matrix_*.csv, feature_metadata.csv, sample_metadata.csv   (an output-folder layout,
                                                                                    so the audit reads a state
                                                                                    exactly like Step 0 output)
      state.json    parent, op_record, counts, scale, applied ops since S0, file hashes
    sessions/<sid>/datasets/<did>/states/HEAD      the current state id

state_id = sha256 of (parent state id, the operation (op, step, params, plan entry), sha256 of every
matrix file). The root state S0 is the Step 0 output folder; its id is computed from the output
folder without writing anything, and `ensure_root` materializes it (a byte copy, CSV) when the
first action is applied. Parent states are never modified: a state folder is written once, to a
temporary name, then renamed; an existing folder with the same id is verified, never rewritten.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
from pathlib import Path

import numpy as np

from ..io.loader import load_output, read_csv
from ..util import canonical_json, now_iso, read_json, sha256_bytes, sha256_file, write_bytes, write_json


class StateError(Exception):
    pass


MATRIX_PREFIX = "value_matrix_"
META_FILES = ("schema.json", "feature_metadata.csv", "sample_metadata.csv")


def states_dir(st, did):
    return st.dataset_dir(did) / "states"


def hashed_op(op_record):
    """The part of an op record that defines the state (who approved and when, package versions
    and counts are recorded but not hashed, so a replay elsewhere reproduces the id)."""
    if op_record is None:
        return {"op": "root"}
    appr = op_record.get("approval") or {}
    return {"op": op_record["op"], "step": op_record.get("step"), "params": op_record.get("params"),
            "seed": op_record.get("seed"), "plan_entry": appr.get("plan_entry")}


def compute_id(parent_id, op_record, matrix_hashes):
    return sha256_bytes(canonical_json({"parent": parent_id, "op": hashed_op(op_record),
                                        "matrices": dict(sorted(matrix_hashes.items()))}).encode("utf-8"))


def _matrix_hashes(folder):
    return {p.name: sha256_file(p) for p in sorted(Path(folder).iterdir())
            if p.is_file() and p.name.startswith(MATRIX_PREFIX)}


def root_id(st, did):
    return compute_id(None, None, _matrix_hashes(st.output_dir(did)))


# ------------------------------------------------------------------ in-memory state data


class Block:
    def __init__(self, assay_id, block_id, file, feature_keys, sample_ids, X):
        self.assay_id, self.block_id, self.file = assay_id, block_id, file
        self.feature_keys, self.sample_ids, self.X = list(feature_keys), list(sample_ids), X


class StateData:
    """What an op sees: the blocks (features x samples), the feature and sample tables, the schema."""

    def __init__(self, schema, blocks, features, samples, scale):
        self.schema, self.blocks, self.scale = schema, blocks, scale
        self.features = features        # {"columns": [...], "rows": [dict]}
        self.samples = samples

    @classmethod
    def load(cls, folder, scale="unknown"):
        out = load_output(folder)
        blocks = []
        for a in out.assays:
            for b in a["blocks"]:
                blocks.append(Block(a["meta"]["assay_id"], b.block_id, b.path.name, b.feature_keys, b.sample_ids,
                                    b.X.copy()))
        return cls(out.schema, blocks, out.feature_table, out.sample_table, scale)

    def copy(self):
        import copy as _c
        return StateData(_c.deepcopy(self.schema),
                         [Block(b.assay_id, b.block_id, b.file, b.feature_keys, b.sample_ids, b.X.copy())
                          for b in self.blocks],
                         _c.deepcopy(self.features), _c.deepcopy(self.samples), self.scale)

    def counts(self):
        keys = {(b.assay_id, k) for b in self.blocks for k in b.feature_keys}
        samples = sorted({s for b in self.blocks for s in b.sample_ids})
        return {"n_features": len(keys), "n_samples": len(samples),
                "n_cells": int(sum(b.X.size for b in self.blocks)),
                "n_nan": int(sum(np.isnan(b.X).sum() for b in self.blocks))}

    def sample_column(self, col):
        """{sample_id: value} of a sample_metadata.csv column (None when absent)."""
        if col not in self.samples["columns"]:
            return None
        return {r["sample_id"]: (r.get(col) or "").strip() for r in self.samples["rows"]}


def _fmt(x):
    return "" if x != x else repr(float(x))


def matrix_bytes(block):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["feature_key"] + block.sample_ids)
    for k, row in zip(block.feature_keys, block.X):
        w.writerow([k] + [_fmt(v) for v in row.tolist()])
    return buf.getvalue().encode("utf-8")


def table_bytes(table):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(table["columns"])
    for r in table["rows"]:
        w.writerow([r.get(c, "") for c in table["columns"]])
    return buf.getvalue().encode("utf-8")


def schema_for(data):
    """schema.json of a child state: counts updated, block files unchanged by name."""
    import copy as _c
    s = _c.deepcopy(data.schema)
    by = {(b.assay_id, b.block_id): b for b in data.blocks}
    for a in s.get("assays", []):
        nf = 0
        for vb in a.get("value_blocks", []):
            b = by.get((a["assay_id"], vb["block_id"]))
            if b is not None:
                vb["n_features"], vb["n_samples"] = len(b.feature_keys), len(b.sample_ids)
                nf += len(b.feature_keys)
        a["n_features"] = nf
        if a.get("value_blocks"):
            first = by.get((a["assay_id"], a["value_blocks"][0]["block_id"]))
            if first is not None:
                a["n_samples"] = len(first.sample_ids)
    return s


# ------------------------------------------------------------------ the store


def head(st, did):
    p = states_dir(st, did) / "HEAD"
    return p.read_text().strip() if p.exists() else root_id(st, did)


def set_head(st, did, state_id):
    write_bytes(states_dir(st, did) / "HEAD", (state_id + "\n").encode())


def folder(st, did, state_id):
    """The folder the audit reads for a state (the output folder for an unmaterialized root)."""
    p = states_dir(st, did) / state_id
    if p.exists():
        return p
    if state_id == root_id(st, did):
        return st.output_dir(did)
    raise StateError(f"Unknown state '{state_id}' for dataset {did}.")


def find(st, state_id):
    """-> (dataset_id, state_id); accepts a unique prefix of at least 6 characters."""
    hits = []
    for d in st.confirmed():
        did = d["dataset_id"]
        ids = {root_id(st, did)} | {p.name for p in states_dir(st, did).glob("*") if p.is_dir()
                                    and not p.name.startswith(".")}
        hits += [(did, i) for i in ids if i == state_id or (len(state_id) >= 6 and i.startswith(state_id))]
    if len(hits) != 1:
        raise StateError(f"Unknown state '{state_id}'." if not hits else f"State prefix '{state_id}' is ambiguous.")
    return hits[0]


SCALE_OPS = {"log_transform": "log2", "arcsinh_transform": "arcsinh"}


def effective_scale(st, did, applied):
    """The scale of a state: set by the last scale-changing op since S0, else the user-confirmed
    processing state of the root (so a later confirmation reaches every state that kept the scale)."""
    for a in reversed(applied or []):
        if a.get("op") in SCALE_OPS:
            return SCALE_OPS[a["op"]]
    return root_scale(st, did)


def meta(st, did, state_id):
    p = states_dir(st, did) / state_id / "state.json"
    if p.exists():
        m = read_json(p)
        m["scale"] = effective_scale(st, did, m.get("applied_ops"))
        return m
    if state_id == root_id(st, did):
        out = st.output_dir(did)
        data = StateData.load(out)
        return {"state_id": state_id, "label": "S0", "dataset": did, "parent": None, "op_record": None,
                "counts": data.counts(), "scale": root_scale(st, did), "applied_ops": [], "checkpoint": "C0",
                "matrices": _matrix_hashes(out), "materialized": False}
    raise StateError(f"Unknown state '{state_id}' for dataset {did}.")


_scale_cache = {}


def root_scale(st, did):
    """linear | log2 | unknown, from the confirmed processing state: the user's answer
    (overrides.json, kind processing_state), else the Step 0 declaration when it matches the
    observed A2 scale class of every assay (A13's `effective` value)."""
    from ..audit import overrides as ov
    from ..audit.a13_processing_state import LOG_LIKE, confirmed, declared
    items = ov.load(st)
    c = confirmed(items, did, "log_transformed")
    if c in ("yes", "no"):
        return {"yes": "log2", "no": "linear"}[c]
    out = st.output_dir(did)
    from ..util import read_json as _rj
    dec = declared(_rj(out / "schema.json").get("processing_history") or {}, "log_transformed")
    if dec not in ("yes", "no"):
        return "unknown"
    key = (str(out), tuple(sorted(_matrix_hashes(out).items())))
    if key not in _scale_cache:
        from ..audit.context import scale_evidence
        data = StateData.load(out)
        classes = set()
        for b in data.blocks:
            classes.add(scale_evidence(b.X)[0])
        _scale_cache[key] = classes
    classes = _scale_cache[key]
    if dec == "yes" and classes <= set(LOG_LIKE):
        return "log2"
    if dec == "no" and classes == {"continuous_linear"}:
        return "linear"
    return "unknown"


def ensure_root(st, did):
    rid = root_id(st, did)
    d = states_dir(st, did) / rid
    if d.exists():
        return rid
    tmp = states_dir(st, did) / f".tmp-{rid[:12]}-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(str(tmp))
    tmp.mkdir(parents=True)
    src = st.output_dir(did)
    for p in sorted(src.iterdir()):
        if p.is_file():
            shutil.copyfile(str(p), str(tmp / p.name))
    m = meta(st, did, rid)
    m.update(materialized=True, created_at=now_iso())
    write_json(tmp / "state.json", m)
    os.replace(str(tmp), str(d))
    if not (states_dir(st, did) / "HEAD").exists():
        set_head(st, did, rid)
    return rid


def load_data(st, did, state_id):
    m = meta(st, did, state_id)
    return StateData.load(folder(st, did, state_id), m["scale"]), m


CHECKPOINT_OF_STEP = {"G3": 0, "G7": 0, "G5": 1, "G6": 1, "S1": 1, "S2": 2, "S3": 2, "S4": 2}


def checkpoint_of(applied):
    """C0 raw (plus approved G3 removals), C1 after G5/G6/S1, C2 after S2/S3."""
    level = max([CHECKPOINT_OF_STEP.get(a.get("step"), 0) for a in applied] + [0])
    return f"C{level}"


def write_state(st, did, parent_meta, data, op_record):
    """Write a child state; -> its meta. An existing state with the same id is verified."""
    blobs = {b.file: matrix_bytes(b) for b in data.blocks}
    hashes = {k: sha256_bytes(v) for k, v in blobs.items()}
    sid = compute_id(parent_meta["state_id"], op_record, hashes)
    d = states_dir(st, did) / sid
    applied = list(parent_meta.get("applied_ops", [])) + [
        {"op": op_record["op"], "step": op_record.get("step"), "state_id": sid,
         "plan_entry": (op_record.get("approval") or {}).get("plan_entry")}]
    m = {"state_id": sid, "label": f"S{len(applied)}", "dataset": did, "parent": parent_meta["state_id"],
         "op_record": op_record, "counts": data.counts(), "scale": data.scale, "applied_ops": applied,
         "checkpoint": checkpoint_of(applied), "matrices": hashes, "materialized": True}
    if d.exists():
        old = read_json(d / "state.json")
        if old["matrices"] != hashes:
            raise StateError(f"State {sid} exists with different matrices.")
        return old
    tmp = states_dir(st, did) / f".tmp-{sid[:12]}-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(str(tmp))
    tmp.mkdir(parents=True)
    for name, b in blobs.items():
        write_bytes(tmp / name, b)
    write_bytes(tmp / "feature_metadata.csv", table_bytes(data.features))
    write_bytes(tmp / "sample_metadata.csv", table_bytes(data.samples))
    write_json(tmp / "schema.json", schema_for(data))
    src = folder(st, did, parent_meta["state_id"])
    if (src / "import_manifest.json").exists():
        shutil.copyfile(str(src / "import_manifest.json"), str(tmp / "import_manifest.json"))
    write_json(tmp / "state.json", dict(m, created_at=now_iso()))
    os.replace(str(tmp), str(d))
    return m


def list_states(st, did=None):
    out = []
    for d in st.confirmed():
        if did and d["dataset_id"] != did:
            continue
        x = d["dataset_id"]
        rid = root_id(st, x)
        h = head(st, x)
        metas = {rid: meta(st, x, rid)}
        for p in sorted(states_dir(st, x).glob("*/state.json")):
            if not p.parent.name.startswith("."):
                metas[p.parent.name] = read_json(p)
        for sid, m in sorted(metas.items(), key=lambda kv: (len(kv[1].get("applied_ops", [])), kv[0])):
            out.append({"dataset": x, "state_id": sid, "label": m.get("label"), "parent": m.get("parent"),
                        "checkpoint": m.get("checkpoint"), "scale": m.get("scale"), "counts": m.get("counts"),
                        "op": (m.get("op_record") or {}).get("op"), "head": sid == h})
    return out


def chain(st, did, state_id):
    """Op records from S0 to the state, in order."""
    recs = []
    m = meta(st, did, state_id)
    while m.get("parent"):
        recs.append(m["op_record"])
        m = meta(st, did, m["parent"])
    if m["state_id"] != root_id(st, did):
        raise StateError("The state chain does not end at this dataset's S0.")
    return list(reversed(recs))


def diff(st, did, a, b):
    ma, mb = meta(st, did, a), meta(st, did, b)
    da, _ = load_data(st, did, a)
    db, _ = load_data(st, did, b)
    blocks = []
    for x in da.blocks:
        y = next((z for z in db.blocks if z.assay_id == x.assay_id and z.block_id == x.block_id), None)
        if y is None:
            continue
        fk = [k for k in x.feature_keys if k in set(y.feature_keys)]
        sk = [s for s in x.sample_ids if s in set(y.sample_ids)]
        ix = {k: i for i, k in enumerate(x.feature_keys)}
        iy = {k: i for i, k in enumerate(y.feature_keys)}
        jx = {s: j for j, s in enumerate(x.sample_ids)}
        jy = {s: j for j, s in enumerate(y.sample_ids)}
        A = x.X[np.ix_([ix[k] for k in fk], [jx[s] for s in sk])]
        B = y.X[np.ix_([iy[k] for k in fk], [jy[s] for s in sk])]
        both = np.isfinite(A) & np.isfinite(B)
        changed = both & (A != B)
        blocks.append({"assay": x.assay_id, "block": x.block_id,
                       "features_removed": len(x.feature_keys) - len(fk), "samples_removed": len(x.sample_ids) - len(sk),
                       "cells_changed": int(changed.sum()),
                       "nan_pattern_identical": bool((np.isnan(A) == np.isnan(B)).all()),
                       "max_abs_change": float(np.abs(A - B)[both].max()) if both.any() else 0.0})
    return {"a": a, "b": b, "dataset": did, "counts": {"a": ma["counts"], "b": mb["counts"]},
            "scale": {"a": ma["scale"], "b": mb["scale"]}, "blocks": blocks}
