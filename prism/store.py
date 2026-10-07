"""Sessions of one or more datasets (v3 §2).

    sessions/<session_id>/
      session.json              datasets, status, schema fingerprints, merge state
      session_schema.json       merged view (merge.py)
      session_sample_table.csv  unified sample table (merge.py)
      overrides.json            audit overrides
      datasets/<dataset_id>/
        upload/                 original files
        schema.json             the confirmed schema
        import_report.json      imports only
        output/                 the Step 0 output folder: the audit's only input
      audit/<run_id>/           manifest.json, findings/, ledger.jsonl, report.html
      logs/

A dataset is made by the Step 0 wizard (a backend session linked to it) or by importing a
schema. Session ids start with 's' so they never collide with Step 0 session ids (12 hex).
"""

from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path

from . import __version__
from .util import now_iso, read_json, schema_sha256, sha256_file, write_json

ROOT = None   # tests and the CLI may set this; default: the Step 0 sessions folder
STATUSES = ("wizard_in_progress", "imported_awaiting_confirm", "confirmed")


class SessionError(Exception):
    pass


def root():
    if ROOT is not None:
        return Path(ROOT)
    if os.environ.get("PRISM_SESSIONS_DIR"):
        return Path(os.environ["PRISM_SESSIONS_DIR"])
    try:
        from backend import workflow
        return Path(workflow.SESSIONS_DIR)
    except Exception:
        return Path(__file__).resolve().parent.parent / "backend" / "sessions"


_ID = re.compile(r"^s[0-9a-f]{11}$")


class StudySession:
    def __init__(self, sid, data):
        self.sid, self.data = sid, data

    @property
    def dir(self):
        return root() / self.sid

    @property
    def datasets(self):
        return self.data["datasets"]

    def dataset(self, did):
        for d in self.datasets:
            if d["dataset_id"] == did:
                return d
        raise SessionError(f"Unknown dataset '{did}'.")

    def dataset_dir(self, did):
        return self.dir / "datasets" / did

    def output_dir(self, did):
        return self.dataset_dir(did) / "output"

    def save(self):
        self.data["updated_at"] = now_iso()
        write_json(self.dir / "session.json", self.data)

    def log(self, event, payload):
        from .ledger import append_jsonl
        append_jsonl(self.dir / "logs" / "session.jsonl", dict({"event": event, "at": now_iso()}, **payload))

    def add_dataset(self, name, origin, status):
        if status not in STATUSES:
            raise SessionError(f"Unknown status '{status}'.")
        n = max([int(d["dataset_id"][1:]) for d in self.datasets] + [0]) + 1
        did = f"D{n}"
        entry = {"dataset_id": did, "name": name, "origin": origin, "status": status, "created_at": now_iso(),
                 "confirmed_at": None, "step0_session_id": None, "schema_sha256": None, "import": None, "files": {}}
        self.datasets.append(entry)
        self.dataset_dir(did).mkdir(parents=True, exist_ok=True)
        self.save()
        self.log("dataset_added", {"dataset_id": did, "name": name, "origin": origin, "status": status})
        return entry

    def remove_dataset(self, did):
        """Only for an import the user rejected: nothing of it is left behind."""
        d = self.dataset(did)
        if d["status"] == "confirmed":
            raise SessionError("A confirmed dataset is never deleted.")
        self.data["datasets"] = [x for x in self.datasets if x["dataset_id"] != did]
        shutil.rmtree(str(self.dataset_dir(did)), ignore_errors=True)
        self.data.get("schema_fingerprints", {}).pop(did, None)
        self.save()
        self.log("dataset_removed", {"dataset_id": did})

    def set_status(self, did, status):
        if status not in STATUSES:
            raise SessionError(f"Unknown status '{status}'.")
        d = self.dataset(did)
        d["status"] = status
        if status == "confirmed":
            d["confirmed_at"] = now_iso()
        self.save()

    def confirmed(self):
        return [d for d in self.datasets if d["status"] == "confirmed"]


def create(name=""):
    sid = "s" + uuid.uuid4().hex[:11]
    s = StudySession(sid, {"session_id": sid, "name": name or sid, "created_at": now_iso(), "prism_version": __version__,
                           "datasets": [], "schema_fingerprints": {}, "merge": {}, "status": "open"})
    s.dir.mkdir(parents=True, exist_ok=True)
    s.save()
    s.log("session_created", {"name": s.data["name"]})
    return s


def load(sid):
    if not _ID.match(sid or ""):
        raise SessionError(f"'{sid}' is not a session id.")
    p = root() / sid / "session.json"
    if not p.exists():
        raise SessionError(f"Unknown session '{sid}'.")
    return StudySession(sid, read_json(p))


def list_sessions():
    out = []
    if not root().exists():
        return out
    for p in sorted(root().glob("s*/session.json")):
        if _ID.match(p.parent.name):
            out.append(read_json(p))
    return out


def publish_output(study, did, src_output, origin_manifest):
    """Copy a finalized Step 0 output folder into the dataset (the one output contract, v3 §4.6):
    output/ gets the files, schema.json, and import_manifest.json; the dataset is confirmed."""
    out = study.output_dir(did)
    if out.exists():
        shutil.rmtree(str(out))
    out.mkdir(parents=True)
    for p in sorted(Path(src_output).iterdir()):
        if p.is_file() and p.name != "import_manifest.json":
            shutil.copyfile(str(p), str(out / p.name))
    schema = read_json(out / "schema.json")
    shutil.copyfile(str(out / "schema.json"), str(study.dataset_dir(did) / "schema.json"))
    manifest = dict(origin_manifest, schema_sha256=schema_sha256(schema), prism_version=__version__,
                    files={p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()})
    write_json(out / "import_manifest.json", manifest)
    d = study.dataset(did)
    d["schema_sha256"] = manifest["schema_sha256"]
    study.data.setdefault("schema_fingerprints", {})[did] = manifest["schema_sha256"]
    study.set_status(did, "confirmed")
    study.log("dataset_confirmed", {"dataset_id": did, "schema_sha256": manifest["schema_sha256"],
                                    "mode": origin_manifest.get("mode")})
    try:   # the saved-schemas library (v3 §4.8): suggested on a later upload of the same file, never applied
        from .io import library
        library.save(schema, {"session_id": study.sid, "dataset_id": did, "mode": origin_manifest.get("mode")})
    except Exception as e:
        study.log("library_not_saved", {"error": str(e)})
    try:   # keep session_schema.json / session_sample_table.csv current; the merge report rewrites them anyway
        from .session import merge
        merge.write(study)
    except Exception as e:
        study.log("merge_outputs_not_written", {"error": str(e)})
    return manifest
