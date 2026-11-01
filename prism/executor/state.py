"""Immutable data states (Build Request v4, section 2.1).

A state is the data at one point of the preprocessing: every block as numbers (proteins x samples,
all samples, all proteins at S0), the protein table and the sample table, and the scale (linear,
log2, arcsinh or unknown). The root state S0 is built from the confirmed schema (the step-10 data);
every approved op makes a child state. Parents are never changed.

    state_id = sha256 of (parent state id, the op (op, step, params, seed, plan entry), sha256 of
               every matrix written as CSV)
so the same ops on the same data always give the same id (that is what replay checks).

StateStore keeps states in memory and, when given a folder, writes each one once:
    <folder>/states/<state_id>/  value_matrix_<block>.csv, feature_metadata.csv, sample_metadata.csv,
                                 state.json (parent, op record, counts, scale, ops since S0)
    <folder>/states/HEAD          the current state id
A folder is written under a temporary name and then renamed; an existing state is verified
(same matrix hashes), never rewritten.
"""

from __future__ import annotations

import copy
import csv
import datetime as _dt
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


class StateError(Exception):
    pass


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


class Block:
    def __init__(self, assay_id, block_id, feature_keys, sample_ids, X):
        self.assay_id, self.block_id = assay_id, block_id
        self.feature_keys, self.sample_ids, self.X = list(feature_keys), list(sample_ids), X

    @property
    def file(self):
        return f"value_matrix_{self.block_id}.csv"

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.X, index=self.feature_keys, columns=self.sample_ids)


class StateData:
    """What an op sees: the blocks, the protein table, the sample table, the scale."""

    def __init__(self, blocks, features: pd.DataFrame, samples: pd.DataFrame, scale: str):
        self.blocks, self.features, self.samples, self.scale = blocks, features, samples, scale

    def copy(self):
        return StateData([Block(b.assay_id, b.block_id, b.feature_keys, b.sample_ids, b.X.copy()) for b in self.blocks],
                         self.features.copy(), self.samples.copy(), self.scale)

    def counts(self):
        keys = {(b.assay_id, k) for b in self.blocks for k in b.feature_keys}
        samples = sorted({s for b in self.blocks for s in b.sample_ids})
        return {"n_features": len(keys), "n_samples": len(samples),
                "n_cells": int(sum(b.X.size for b in self.blocks)),
                "n_nan": int(sum(np.isnan(b.X).sum() for b in self.blocks))}

    def block(self, block_id) -> Block:
        for b in self.blocks:
            if b.block_id == block_id:
                return b
        raise StateError(f"No block {block_id} in this state.")


def from_schema(schema, scale: str) -> StateData:
    """S0: every block of the confirmed schema as numbers, all proteins, all samples."""
    blocks = []
    for bid, g in schema.block_map().items():
        m = schema.matrix(bid, numbers=True)
        blocks.append(Block(g.assay_label or "assay", bid, [str(k) for k in m.index], list(m.columns),
                            m.to_numpy(dtype=float)))
    features = schema.feature_metadata()
    features["feature"] = features["feature"].astype(str)
    return StateData(blocks, features, schema.sample_table(), scale)


def _fmt(x):
    return "" if x != x else repr(float(x))


def matrix_bytes(block: Block) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["feature"] + block.sample_ids)
    for k, row in zip(block.feature_keys, block.X):
        w.writerow([k] + [_fmt(v) for v in row.tolist()])
    return buf.getvalue().encode("utf-8")


def matrix_hashes(data: StateData) -> dict:
    return {b.file: sha256_bytes(matrix_bytes(b)) for b in data.blocks}


def hashed_op(op_record):
    """The part of an op record that defines the state (who approved and when, versions and counts
    are recorded but not hashed, so a replay elsewhere reproduces the id)."""
    if op_record is None:
        return {"op": "root"}
    appr = op_record.get("approval") or {}
    return {"op": op_record["op"], "step": op_record.get("step"), "params": op_record.get("params"),
            "seed": op_record.get("seed"), "plan_entry": appr.get("plan_entry")}


def compute_id(parent_id, op_record, hashes) -> str:
    return sha256_bytes(canonical_json({"parent": parent_id, "op": hashed_op(op_record),
                                        "matrices": dict(sorted(hashes.items()))}).encode("utf-8"))


class StateStore:
    """The states of one dataset. folder=None keeps them in memory only."""

    def __init__(self, root: StateData, folder=None):
        self.folder = Path(folder) if folder else None
        self._data, self._meta = {}, {}
        rid = compute_id(None, None, matrix_hashes(root))
        meta = {"state_id": rid, "parent": None, "op_record": None, "counts": root.counts(), "scale": root.scale,
                "applied_ops": [], "matrix_sha256": matrix_hashes(root), "created": now_iso()}
        self.root_id = rid
        self._put(root, meta)
        self.head = rid

    def _put(self, data, meta):
        sid = meta["state_id"]
        if sid in self._meta:
            if self._meta[sid]["matrix_sha256"] != meta["matrix_sha256"]:
                raise StateError(f"State {sid[:12]} exists with different matrices.")
            return
        self._data[sid], self._meta[sid] = data, meta
        if self.folder:
            self._write(data, meta)

    def _write(self, data, meta):
        base = self.folder / "states"
        final = base / meta["state_id"]
        if final.exists():
            have = json.loads((final / "state.json").read_text())
            if have["matrix_sha256"] != meta["matrix_sha256"]:
                raise StateError(f"State folder {final} exists with different matrices.")
            return
        tmp = base / (meta["state_id"] + ".tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        for b in data.blocks:
            (tmp / b.file).write_bytes(matrix_bytes(b))
        data.features.to_csv(tmp / "feature_metadata.csv", index=False)
        data.samples.to_csv(tmp / "sample_metadata.csv")
        (tmp / "state.json").write_text(json.dumps(meta, indent=1, default=str))
        os.replace(tmp, final)

    def add(self, parent_id, child: StateData, op_record, state_id) -> dict:
        pm = self._meta[parent_id]
        meta = {"state_id": state_id, "parent": parent_id, "op_record": op_record, "counts": child.counts(),
                "scale": child.scale, "matrix_sha256": matrix_hashes(child), "created": now_iso(),
                "applied_ops": pm["applied_ops"] + [{"op": op_record["op"], "step": op_record.get("step"),
                                                     "state_id": state_id}]}
        self._put(child, meta)
        if self.head == parent_id:
            self.head = state_id
            if self.folder:
                (self.folder / "states" / "HEAD").write_text(state_id)
        return meta

    def data(self, sid=None) -> StateData:
        return self._data[sid or self.head]

    def meta(self, sid=None) -> dict:
        return self._meta[sid or self.head]

    def chain(self, sid=None) -> list[dict]:
        """The op records from S0 to this state, in order."""
        out, cur = [], sid or self.head
        while self._meta[cur]["parent"] is not None:
            out.append(self._meta[cur]["op_record"])
            cur = self._meta[cur]["parent"]
        return out[::-1]

    def list(self) -> pd.DataFrame:
        return pd.DataFrame([{"state_id": m["state_id"][:12], "parent": (m["parent"] or "")[:12],
                              "op": (m["op_record"] or {}).get("op", "S0 (root)"), "scale": m["scale"],
                              **m["counts"]} for m in self._meta.values()])

    def file_hashes(self, sid) -> dict:
        """sha256 of every file of a written state (to prove parents are never modified)."""
        d = self.folder / "states" / sid
        return {p.name: sha256_bytes(p.read_bytes()) for p in sorted(d.iterdir())}
