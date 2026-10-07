"""The saved-schemas library (v3 §4.8): every confirmed schema, keyed by the sha256 of the data
file it describes. Uploading a file already seen suggests its saved schema ("use it?"). Never
applied automatically: using one goes through the normal import and its review.

    <sessions root>/_library/<file_sha256>/<schema_sha256>.json   the schema
    <sessions root>/_library/index.json                           [{schema_id, file_sha256, ...}]
"""

from __future__ import annotations

import re

from .. import store
from ..util import now_iso, read_json, schema_sha256, write_json

_SHA = re.compile(r"^[0-9a-f]{64}$")


def root():
    return store.root() / "_library"


def _index():
    p = root() / "index.json"
    return read_json(p) if p.exists() else []


def save(doc, origin):
    """Add a confirmed schema (idempotent per schema fingerprint). -> the index entry."""
    fsha = doc.get("file_sha256") or ""
    if not _SHA.match(fsha):
        return None
    ssha = schema_sha256(doc)
    idx = _index()
    hit = next((e for e in idx if e["file_sha256"] == fsha and e["schema_sha256"] == ssha), None)
    if hit:
        return hit
    write_json(root() / fsha / f"{ssha}.json", doc)
    meta = next((f for f in doc.get("files", []) if f.get("file_role") == "metadata"), None)
    entry = {"schema_id": ssha[:16], "schema_sha256": ssha, "file_sha256": fsha, "source_file": doc.get("source_file"),
             "schema_version": doc.get("schema_version"),
             "assays": [a.get("assay_label") for a in doc.get("assays", [])],
             "metadata_file": {"name": meta.get("name"), "sha256": meta.get("sha256")} if meta else None,
             "saved_at": now_iso(), **origin}
    idx.append(entry)
    write_json(root() / "index.json", idx)
    return entry


def matches(file_sha):
    return [e for e in _index() if e["file_sha256"] == file_sha]


def get(schema_id):
    """-> (entry, schema bytes)."""
    e = next((x for x in _index() if x["schema_id"] == schema_id), None)
    if e is None:
        raise store.SessionError(f"Unknown saved schema '{schema_id}'.")
    return e, (root() / e["file_sha256"] / f"{e['schema_sha256']}.json").read_bytes()


def use(st, did, schema_id):
    """Import the wizard dataset `did` (its upload copy) with a saved schema. The import report
    replaces the in-progress wizard dataset only when the import succeeds."""
    from . import importer
    d = st.dataset(did)
    if d["status"] != "wizard_in_progress":
        raise store.SessionError("A saved schema can only replace a dataset still in the wizard.")
    entry, raw_schema = get(schema_id)
    name = (d.get("files") or {}).get("data", {}).get("name")
    up = st.dataset_dir(did) / "upload" / (name or "")
    if not name or not up.exists():
        raise store.SessionError("The uploaded file of this dataset is not available.")
    data = up.read_bytes()
    from ..util import sha256_bytes
    if sha256_bytes(data) != entry["file_sha256"]:
        raise store.SessionError("This saved schema describes a different file.")
    rep = importer.import_bytes(st, name, data, raw_schema)
    importer.reject(st, did)          # the wizard attempt is superseded; nothing confirmed is touched
    st.log("saved_schema_used", {"schema_id": schema_id, "replaced": did, "dataset_id": rep["dataset_id"]})
    return rep
