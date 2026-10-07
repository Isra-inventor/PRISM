"""Read one Step 0 output folder (the audit's only input contract, v3 §1.2).

    schema.json               the confirmed schema
    value_matrix_A{n}_B{m}.csv  features x samples, 'feature_key' then one column per sample ID
    feature_metadata.csv      feature_key, assay_id, (source_block), kept annotations, derived parts
    sample_metadata.csv       sample_id, sample_label, is_study_sample, kept sample columns
    import_manifest.json      how the folder was made (wizard or import)

Values are parsed as numbers; an empty cell is missing (NaN). A cell that is not a number is
also NaN and counted (A1 reports it). Nothing is ever written here.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from ..util import read_json, sha256_file


class LoadError(Exception):
    pass


def _num(v):
    v = v.strip()
    if v == "":
        return math.nan, False
    try:
        return float(v), False
    except ValueError:
        pass
    if v.count(",") == 1 and "." not in v:   # a decimal comma copied as-is by Step 0
        try:
            return float(v.replace(",", ".")), False
        except ValueError:
            pass
    return math.nan, True


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise LoadError(f"{Path(path).name} is empty.")
    return rows[0], rows[1:]


class Block:
    """One value matrix: X (features x samples), its feature keys and sample IDs."""

    def __init__(self, assay_id, block_id, meta, path):
        self.assay_id, self.block_id, self.meta, self.path = assay_id, block_id, meta, Path(path)
        header, rows = read_csv(path)
        if not header or header[0] != "feature_key":
            raise LoadError(f"{self.path.name}: the first column must be 'feature_key'.")
        self.sample_ids = header[1:]
        self.feature_keys = [r[0] if r else "" for r in rows]
        n, m = len(rows), len(self.sample_ids)
        X = np.full((n, m), np.nan)
        bad = 0
        for i, r in enumerate(rows):
            if len(r) - 1 != m:
                raise LoadError(f"{self.path.name}: row {i + 2} has {len(r) - 1} values for {m} samples.")
            for j, v in enumerate(r[1:]):
                x, b = _num(v)
                X[i, j] = x
                bad += b
        self.X = X
        self.n_unparsable = bad

    @property
    def label(self):
        return self.meta.get("label") or self.block_id


class Output:
    def __init__(self, path):
        self.path = Path(path)
        if not (self.path / "schema.json").exists():
            raise LoadError(f"{self.path} has no schema.json: not a Step 0 output folder.")
        self.schema = read_json(self.path / "schema.json")
        self.files = {p.name: sha256_file(p) for p in sorted(self.path.iterdir()) if p.is_file()}
        self.manifest = read_json(self.path / "import_manifest.json") if (self.path / "import_manifest.json").exists() else None
        self.assays = []
        for a in self.schema.get("assays", []):
            blocks = []
            for b in a.get("value_blocks", []):
                f = self.path / b["file"]
                if not f.exists():
                    raise LoadError(f"{b['file']} is listed in schema.json but missing from the folder.")
                blocks.append(Block(a["assay_id"], b["block_id"], b, f))
            self.assays.append({"meta": a, "blocks": blocks})
        self.feature_table = self._table("feature_metadata.csv")
        self.sample_table = self._table("sample_metadata.csv")

    def _table(self, name):
        p = self.path / name
        if not p.exists():
            return {"columns": [], "rows": []}
        header, rows = read_csv(p)
        return {"columns": header, "rows": [dict(zip(header, r)) for r in rows]}

    def assay(self, aid):
        for a in self.assays:
            if a["meta"]["assay_id"] == aid:
                return a
        raise LoadError(f"Unknown assay '{aid}'.")

    def features_of(self, aid):
        """{feature_key: row} of feature_metadata.csv for one assay (first row wins on duplicates)."""
        out = {}
        for r in self.feature_table["rows"]:
            if r.get("assay_id") == aid:
                out.setdefault(r.get("feature_key"), r)
        return out


def load_output(path):
    return Output(path)
