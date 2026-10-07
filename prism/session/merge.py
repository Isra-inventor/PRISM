"""Multi-dataset sessions: compare sample IDs, unify them, merge the sample tables (v3 §5, M1 to M7).

Deterministic. Reads only the confirmed datasets' output folders and the user's merge decisions
(session.json "merge"). Sample IDs are never rewritten: a unified ID is a separate column, and a
mapping is applied only when the user confirms a suggestion or uploads a mapping CSV.

    session_schema.json       datasets[] and cross_dataset {sample_overlap, id_mapping, conflicts,
                              design_agreement, ...}
    session_sample_table.csv  unified_id, datasets, sample_id@D.., presence, merged columns

Items the user decides on carry stable ids derived from their content:
    ms...  an ID suggestion (near miss)          confirm | dismiss | reopen
    mc...  a value conflict in a shared column   take:D1 | keep_both | drop | reopen
    mq...  a question                            an option_id | dismiss | reopen
"""

from __future__ import annotations

import csv
import io
import re
from itertools import combinations

from ..io.loader import LoadError, load_output, read_csv
from ..util import canonical_json, now_iso, sha256_bytes, write_bytes, write_json

PRISM_COLUMNS = ("sample_label", "is_study_sample")


class MergeError(Exception):
    pass


def _id(prefix, *parts):
    return prefix + sha256_bytes(canonical_json(list(parts)).encode("utf-8"))[:10]


# ------------------------------------------------------------------ M1: comparing IDs

_SEP = re.compile(r"[-_ ]+")
_ZEROS = re.compile(r"(?<![0-9])0+(?=[0-9])")

RULES = (
    ("whitespace", lambda s: s.strip()),
    ("case", lambda s: s.casefold()),
    ("separator", lambda s: _SEP.sub("_", s)),
    ("leading_zeros", lambda s: _ZEROS.sub("", s)),
)


def normalize(s, skip=None):
    for name, f in RULES:
        if name != skip:
            s = f(s)
    return s


def rules_needed(a, b):
    """The near-miss rules without which the two IDs stay different, or None if they never match."""
    if normalize(a) != normalize(b):
        return None
    return [name for name, _ in RULES if normalize(a, name) != normalize(b, name)]


def near_misses(ids_a, ids_b):
    """Pairs (a, b) not equal but equal after the rules; ambiguous keys are reported apart."""
    only_a = sorted(set(ids_a) - set(ids_b))
    only_b = sorted(set(ids_b) - set(ids_a))
    ka, kb = {}, {}
    for x in only_a:
        ka.setdefault(normalize(x), []).append(x)
    for x in only_b:
        kb.setdefault(normalize(x), []).append(x)
    pairs, ambiguous = [], []
    for k in sorted(set(ka) & set(kb)):
        if len(ka[k]) == 1 and len(kb[k]) == 1:
            pairs.append((ka[k][0], kb[k][0], rules_needed(ka[k][0], kb[k][0])))
        else:
            ambiguous.append({"key": k, "a": ka[k], "b": kb[k]})
    return pairs, ambiguous


# ------------------------------------------------------------------ dataset views


class DatasetView:
    """What the merge needs from one confirmed dataset's output folder."""

    def __init__(self, study, d):
        self.did, self.name = d["dataset_id"], d["name"]
        self.entry = d
        out = load_output(study.output_dir(self.did))
        self.output = out
        self.schema = out.schema
        tbl = out.sample_table
        self.columns = [c for c in tbl["columns"] if c != "sample_id"]
        self.rows = {r["sample_id"]: r for r in tbl["rows"]}
        self.sample_ids = [r["sample_id"] for r in tbl["rows"]]
        if not self.sample_ids:          # no sample table: the value matrices' headers
            seen = []
            for a in out.assays:
                for b in a["blocks"]:
                    seen += [s for s in b.sample_ids if s not in seen]
            self.sample_ids = seen
            self.rows = {s: {"sample_id": s} for s in seen}
        self.kinds = {c["column"]: c.get("audit_kind") or "other" for c in self.schema.get("sample_metadata", [])}
        for c in PRISM_COLUMNS:
            self.kinds[c] = "prism"
        design = self.schema.get("design") or {}
        self.subject_col = self._design_col(design.get("subject") or {}, "derived_subject")
        self.time_col = self._design_col(design.get("time") or {}, "derived_time")
        unit = ((design.get("time") or {}).get("unit") or {})
        self.time_unit = unit.get("value") if isinstance(unit, dict) else unit

    def _design_col(self, side, derived):
        src = side.get("source")
        if src == "derived_from_sample_names":
            return derived if derived in self.columns else None
        if src == "metadata_column" and side.get("column") in self.columns:
            return side["column"]
        return None

    def value(self, sid, col):
        return (self.rows.get(sid) or {}).get(col, "")

    def summary(self):
        s = self.schema
        return {"dataset_id": self.did, "name": self.name, "schema_sha256": self.entry.get("schema_sha256"),
                "origin": self.entry.get("origin"), "source_file": s.get("source_file"),
                "file_sha256": s.get("file_sha256"),
                "omics_family": (s.get("omics_family") or {}).get("value"),
                "assays": [{"assay_id": a["assay_id"], "assay_label": a.get("assay_label"),
                            "omics_type": a.get("omics_type"),
                            "omics_family": (a.get("omics_family") or {}).get("value"),
                            "n_features": a.get("n_features"), "n_samples": a.get("n_samples")}
                           for a in s.get("assays", [])],
                "n_samples": len(self.sample_ids),
                "n_study_samples": sum(1 for x in self.sample_ids if self.value(x, "is_study_sample") != "false"),
                "subject_column": self.subject_col, "time_column": self.time_col, "time_unit": self.time_unit}


# ------------------------------------------------------------------ M2: unified IDs


def parse_mapping_csv(raw, views):
    """dataset_id, sample_id, unified_id. Every row must name a known dataset and sample, and the
    mapping must be injective within each dataset (two samples of one dataset never share a
    unified ID). -> list of entries, or MergeError with every problem."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise MergeError("The mapping file is not UTF-8 text.")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise MergeError("The mapping file is empty.")
    head = [h.strip() for h in rows[0]]
    need = ("dataset_id", "sample_id", "unified_id")
    if any(n not in head for n in need):
        raise MergeError("The mapping file needs the columns dataset_id, sample_id, unified_id.")
    ix = [head.index(n) for n in need]
    by = {v.did: v for v in views}
    errors, entries, seen = [], [], {}
    for n, r in enumerate(rows[1:], start=2):
        if not any(x.strip() for x in r):
            continue
        try:
            did, sid, uid = (r[i] for i in ix)
        except IndexError:
            errors.append(f"row {n}: missing values")
            continue
        did, uid = did.strip(), uid.strip()
        if did not in by:
            errors.append(f"row {n}: unknown dataset '{did}'")
            continue
        if sid not in by[did].rows:
            errors.append(f"row {n}: '{sid}' is not a sample of {did}")
            continue
        if not uid:
            errors.append(f"row {n}: empty unified_id")
            continue
        if (did, sid) in seen and seen[(did, sid)] != uid:
            errors.append(f"row {n}: {did} '{sid}' is mapped twice ('{seen[(did, sid)]}' and '{uid}')")
            continue
        seen[(did, sid)] = uid
        entries.append({"dataset_id": did, "sample_id": sid, "unified_id": uid})
    if not errors:
        errors = _injectivity(views, {(e["dataset_id"], e["sample_id"]): e["unified_id"] for e in entries})
    if errors:
        raise MergeError("The mapping is not valid: " + "; ".join(errors[:20]) +
                         (f" (and {len(errors) - 20} more)" if len(errors) > 20 else ""))
    uniq = {}
    for e in entries:
        uniq[(e["dataset_id"], e["sample_id"])] = e
    return [uniq[k] for k in sorted(uniq)]


def _injectivity(views, mapping):
    errors = []
    for v in views:
        back = {}
        for sid in v.sample_ids:
            uid = mapping.get((v.did, sid), sid)
            back.setdefault(uid, []).append(sid)
        for uid, sids in sorted(back.items()):
            if len(sids) > 1:
                errors.append(f"in {v.did}, {', '.join(repr(s) for s in sids)} would all become '{uid}'")
    return errors


# ------------------------------------------------------------------ helpers


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def same_value(a, b):
    a, b = (a or "").strip(), (b or "").strip()
    if a == b:
        return True
    x, y = _num(a), _num(b)
    if x is not None and y is not None:
        return x == y or abs(x - y) <= 1e-9 * max(abs(x), abs(y))
    return a.casefold() in ("true", "false") and a.casefold() == b.casefold()


def _state(study):
    m = study.data.setdefault("merge", {})
    m.setdefault("decisions", {})
    m.setdefault("mapping_csv", None)
    return m


# ------------------------------------------------------------------ the report


def views_of(study):
    out = []
    for d in study.confirmed():
        try:
            out.append(DatasetView(study, d))
        except LoadError as e:
            raise MergeError(f"{d['dataset_id']}: {e}")
    return out


def compute(study):
    """Everything the merge knows, from the output folders and the decisions; no writes."""
    st = _state(study)
    dec = st["decisions"]
    views = views_of(study)
    by = {v.did: v for v in views}

    # M1: near-miss suggestions per pair of datasets (exact matches need nothing)
    csv_map = {(e["dataset_id"], e["sample_id"]): e["unified_id"] for e in (st["mapping_csv"] or {}).get("entries", [])}
    suggestions, ambiguous = [], []
    for a, b in combinations(views, 2):
        pairs, amb = near_misses(a.sample_ids, b.sample_ids)
        for ida, idb, rules in pairs:
            sid = _id("ms", a.did, ida, b.did, idb)
            d = dec.get(sid) or {}
            status = d.get("status", "open")
            if status == "open" and ((b.did, idb) in csv_map or (a.did, ida) in csv_map):
                status = "superseded_by_mapping"
            suggestions.append({"suggestion_id": sid, "a": {"dataset_id": a.did, "sample_id": ida},
                                "b": {"dataset_id": b.did, "sample_id": idb}, "rules": rules, "status": status,
                                "unified_id": d.get("unified_id") or ida})
        for x in amb:
            ambiguous.append(dict(x, a_dataset=a.did, b_dataset=b.did))

    # M2: unified IDs (identity unless the user said otherwise)
    mapping = {}
    entries = []
    for s in suggestions:
        if s["status"] == "confirmed":
            for side in ("a", "b"):
                k = (s[side]["dataset_id"], s[side]["sample_id"])
                if k[1] != s["unified_id"]:
                    mapping[k] = s["unified_id"]
                    entries.append({"dataset_id": k[0], "sample_id": k[1], "unified_id": s["unified_id"],
                                    "source": "confirmed_suggestion", "suggestion_id": s["suggestion_id"]})
    for k, uid in sorted(csv_map.items()):
        mapping[k] = uid
        entries = [e for e in entries if (e["dataset_id"], e["sample_id"]) != k]
        if uid != k[1]:
            entries.append({"dataset_id": k[0], "sample_id": k[1], "unified_id": uid, "source": "mapping_csv"})
    mapping_errors = _injectivity(views, mapping)

    def uid_of(did, sid):
        return mapping.get((did, sid), sid)

    members = {}          # unified id -> {did: original sample id}
    for v in views:
        for sid in v.sample_ids:
            members.setdefault(uid_of(v.did, sid), {})[v.did] = sid
    order = []
    for v in views:
        for sid in v.sample_ids:
            u = uid_of(v.did, sid)
            if u not in order:
                order.append(u)
    order_ix = {u: i for i, u in enumerate(order)}

    # M3: overlap
    sets = {v.did: {u for u, m in members.items() if v.did in m} for v in views}
    pairs = []
    for a, b in combinations(views, 2):
        sa, sb = sets[a.did], sets[b.did]
        union = sa | sb
        pairs.append({"a": a.did, "b": b.did, "n_shared": len(sa & sb), "n_only_a": len(sa - sb),
                      "n_only_b": len(sb - sa), "jaccard": round(len(sa & sb) / len(union), 6) if union else None,
                      "shared": sorted(sa & sb, key=order_ix.get), "only_a": sorted(sa - sb, key=order_ix.get),
                      "only_b": sorted(sb - sa, key=order_ix.get)})
    all_sets = list(sets.values())
    in_all = set.intersection(*all_sets) if all_sets else set()
    overlap = {"n_datasets": len(views), "n_unified": len(order), "n_in_all": len(in_all),
               "per_dataset": {v.did: len(sets[v.did]) for v in views}, "pairs": pairs,
               "presence": {u: [v.did for v in views if v.did in members[u]] for u in order}}

    # M4: unified sample table
    questions, conflicts, columns = [], [], []
    names = []
    for v in views:
        names += [c for c in v.columns if c not in names]
    for col in names:
        holders = [v for v in views if col in v.columns]
        kinds = {v.did: v.kinds.get(col, "other") for v in holders}
        if len(holders) == 1:
            columns.append({"column": col, "datasets": [holders[0].did], "audit_kind": kinds[holders[0].did],
                            "status": "single"})
            continue
        groups = [holders]
        if len(set(kinds.values())) > 1:
            qid = _id("mq", "audit_kind_mismatch", col, sorted(kinds.items()))
            d = dec.get(qid) or {}
            opts = [{"option_id": f"same:{k}", "label": f"One variable: treat it as '{k}' in every dataset"}
                    for k in sorted(set(kinds.values()))]
            opts.append({"option_id": "different", "label": "Different variables: keep one column per dataset"})
            q = {"question_id": qid, "kind": "audit_kind_mismatch", "column": col, "kinds": kinds,
                 "text": f"'{col}' is described differently: " +
                         ", ".join(f"{did} '{k}'" for did, k in sorted(kinds.items())) + ". Is it one variable?",
                 "options": opts, "status": d.get("status", "open"), "answer": d.get("answer")}
            questions.append(q)
            ans = d.get("answer") if d.get("status") == "answered" else None
            if ans and ans.startswith("same:"):
                kinds = {k: ans[5:] for k in kinds}
            else:
                columns.append({"column": col, "datasets": [v.did for v in holders], "audit_kind": kinds,
                                "status": "kept_per_dataset", "question_id": qid})
                continue
        kind = next(iter(kinds.values()))
        disagree = []
        shared = [u for u in order if sum(1 for v in holders if v.did in members[u]) > 1]
        for u in shared:
            vals = {v.did: v.value(members[u][v.did], col) for v in holders if v.did in members[u]}
            filled = {k: x for k, x in vals.items() if (x or "").strip() != ""}
            if len(filled) > 1:
                ref = next(iter(filled.values()))
                if not all(same_value(ref, x) for x in filled.values()):
                    disagree.append({"unified_id": u, "values": vals})
        entry = {"column": col, "datasets": [v.did for v in holders], "audit_kind": kind, "n_shared": len(shared)}
        if not disagree:
            entry["status"] = "merged"
            columns.append(entry)
            continue
        cid = _id("mc", col, [v.did for v in holders], canonical_json(disagree))
        d = dec.get(cid) or {}
        res = d.get("resolution") if d.get("status") == "resolved" else None
        conflicts.append({"conflict_id": cid, "column": col, "audit_kind": kind,
                          "datasets": [v.did for v in holders], "n_disagree": len(disagree),
                          "disagreements": disagree[:200], "status": d.get("status", "open"), "resolution": res,
                          "options": [f"take:{v.did}" for v in holders] + ["keep_both", "drop"]})
        entry["status"] = {None: "conflict_open", "keep_both": "kept_per_dataset", "drop": "dropped"}.get(
            res, "taken")
        entry["conflict_id"] = cid
        if res and res.startswith("take:"):
            entry["take"] = res[5:]
        columns.append(entry)

    # column suggestions: different names, identical values on shared samples (never applied)
    col_suggestions = []
    for a, b in combinations(views, 2):
        shared = [u for u in order if a.did in members[u] and b.did in members[u]]
        if len(shared) < 3:
            continue
        ca = [c for c in a.columns if c not in PRISM_COLUMNS]
        cb = [c for c in b.columns if c not in PRISM_COLUMNS]
        sig_b = {}
        for c in cb:
            vals = tuple(b.value(members[u][b.did], c).strip() for u in shared)
            if all(vals) and len(set(vals)) > 1:
                sig_b.setdefault(vals, []).append(c)
        for c in ca:
            vals = tuple(a.value(members[u][a.did], c).strip() for u in shared)
            for c2 in [x for x in sig_b.get(vals, []) if x != c]:
                col_suggestions.append({"a": {"dataset_id": a.did, "column": c}, "b": {"dataset_id": b.did, "column": c2},
                                        "n_shared": len(shared), "note": "identical values on every shared sample; "
                                        "a suggestion only: nothing is merged"})

    # M5/M6: design agreement and possible ID reuse
    design = []
    reuse = []
    for a, b in combinations(views, 2):
        shared = [u for u in order if a.did in members[u] and b.did in members[u]]
        item = {"a": a.did, "b": b.did, "n_shared": len(shared)}
        if a.subject_col and b.subject_col and shared:
            pairs_s = [(u, a.value(members[u][a.did], a.subject_col), b.value(members[u][b.did], b.subject_col))
                       for u in shared]
            bad = _partition_disagreements(pairs_s)
            item["subject"] = {"columns": {a.did: a.subject_col, b.did: b.subject_col}, "n_compared": len(pairs_s),
                               "n_disagree": len(bad), "same_labels": all(x == y for _, x, y in pairs_s),
                               "agree": not bad}
            for u, x, y in bad:
                reuse.append({"unified_id": u, "field": "subject", "a": a.did, "b": b.did, "values": {a.did: x, b.did: y}})
        else:
            item["subject"] = None
        ua, ub = a.time_unit, b.time_unit
        if a.time_col and b.time_col:
            item["time_unit"] = {a.did: ua, b.did: ub, "match": bool(ua and ub and ua == ub)}
            if not item["time_unit"]["match"]:
                qid = _id("mq", "time_unit_mismatch", a.did, b.did, ua, ub)
                d = dec.get(qid) or {}
                questions.append({
                    "question_id": qid, "kind": "time_unit_mismatch", "datasets": [a.did, b.did],
                    "text": f"Time is in '{ua or 'unknown'}' in {a.did} and '{ub or 'unknown'}' in {b.did}. "
                            "Time values are only compared when the units match.",
                    "options": [{"option_id": "acknowledged", "label": "Noted: compare time per dataset only"}],
                    "status": d.get("status", "open"), "answer": d.get("answer")})
            if shared and item["time_unit"]["match"]:
                bad = []
                for u in shared:
                    x, y = a.value(members[u][a.did], a.time_col), b.value(members[u][b.did], b.time_col)
                    if x.strip() and y.strip() and not same_value(x, y):
                        bad.append((u, x, y))
                item["time"] = {"columns": {a.did: a.time_col, b.did: b.time_col}, "n_compared": len(shared),
                                "n_disagree": len(bad), "agree": not bad}
                for u, x, y in bad:
                    reuse.append({"unified_id": u, "field": "time", "a": a.did, "b": b.did,
                                  "values": {a.did: x, b.did: y}})
            else:
                item["time"] = None
        else:
            item["time_unit"] = None
            item["time"] = None
        design.append(item)
    for (a, b), group in _group_by(reuse, lambda r: (r["a"], r["b"])):
        qid = _id("mq", "possible_id_reuse", a, b, [(r["unified_id"], r["field"]) for r in group])
        d = dec.get(qid) or {}
        fields = sorted({r["field"] for r in group})
        questions.append({
            "question_id": qid, "kind": "possible_id_reuse", "datasets": [a, b],
            "unified_ids": [r["unified_id"] for r in group],
            "text": f"{len({r['unified_id'] for r in group})} shared sample ID(s) have a different "
                    f"{' and '.join(fields)} in {a} and {b}: possibly the same ID used for different samples.",
            "options": [{"option_id": "same_samples", "label": "Same samples: only the labels differ"},
                        {"option_id": "flag", "label": "Possibly different samples: flag them in the audit"}],
            "status": d.get("status", "open"), "answer": d.get("answer")})

    # M7: ready for audit
    blocking = []
    for d in study.datasets:
        if d["status"] != "confirmed":
            blocking.append(f"{d['dataset_id']} ({d['name']}) is not confirmed ({d['status']}).")
    if not views:
        blocking.append("No confirmed dataset yet.")
    n_open_s = sum(1 for s in suggestions if s["status"] == "open")
    if n_open_s:
        blocking.append(f"{n_open_s} ID suggestion(s) to confirm or dismiss.")
    if mapping_errors:
        blocking.append("The ID mapping is not injective: " + "; ".join(mapping_errors))
    n_open_c = sum(1 for c in conflicts if c["status"] == "open")
    if n_open_c:
        blocking.append(f"{n_open_c} value conflict(s) to resolve.")
    n_open_q = sum(1 for q in questions if q["status"] == "open")
    if n_open_q:
        blocking.append(f"{n_open_q} merge question(s) to answer or dismiss.")

    return {"views": views, "members": members, "order": order, "suggestions": suggestions, "ambiguous": ambiguous,
            "id_mapping": {"default": "exact sample ID", "entries": entries,
                           "mapping_csv": {k: v for k, v in (st["mapping_csv"] or {}).items() if k != "entries"}
                           if st["mapping_csv"] else None, "errors": mapping_errors},
            "overlap": overlap, "columns": columns, "conflicts": conflicts, "column_suggestions": col_suggestions,
            "design_agreement": design, "possible_id_reuse": reuse, "questions": questions,
            "ready_for_audit": not blocking, "blocking": blocking}


def _group_by(items, key):
    out = {}
    for x in items:
        out.setdefault(key(x), []).append(x)
    return sorted(out.items())


def _partition_disagreements(pairs):
    """Subjects agree when the two labelings split the shared samples the same way (labels may
    differ, e.g. 'DEAB' and 'NHP-DEAB'). Returns the samples that break the correspondence."""
    fwd, back = {}, {}
    for _, x, y in pairs:
        if x.strip() and y.strip():
            fwd.setdefault(x, {}).setdefault(y, 0)
            fwd[x][y] += 1
            back.setdefault(y, {}).setdefault(x, 0)
            back[y][x] += 1

    def major(d):
        return sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    bad = []
    for u, x, y in pairs:
        if not (x.strip() and y.strip()):
            continue
        if major(fwd[x]) != y or major(back[y]) != x:
            bad.append((u, x, y))
    return bad


# ------------------------------------------------------------------ outputs


def unified_table(r, only=None):
    """(header, rows) of the unified sample table; only=did keeps that dataset's samples (the M of
    a one-dataset audit)."""
    views = r["views"]
    header = ["unified_id", "datasets"] + [f"sample_id@{v.did}" for v in views] + [f"in_{v.did}" for v in views]
    plan = []
    for c in r["columns"]:
        if c["status"] in ("single", "merged"):
            plan.append((c["column"], c["column"], c["datasets"]))
        elif c["status"] == "taken":
            plan.append((c["column"], c["column"], [c["take"]]))
        elif c["status"] in ("kept_per_dataset", "conflict_open"):
            for did in c["datasets"]:
                plan.append((f"{c['column']}@{did}", c["column"], [did]))
    header += [p[0] for p in plan]
    by = {v.did: v for v in views}
    rows = []
    for u in r["order"]:
        m = r["members"][u]
        if only and only not in m:
            continue
        row = [u, ";".join(v.did for v in views if v.did in m)]
        row += [m.get(v.did, "") for v in views]
        row += ["1" if v.did in m else "0" for v in views]
        for _, col, dids in plan:
            val = ""
            for did in dids:
                if did in m:
                    x = by[did].value(m[did], col)
                    if (x or "").strip():
                        val = x
                        break
            row.append(val)
        rows.append(row)
    return header, rows


def session_schema(study, r):
    return {"session_id": study.sid, "name": study.data.get("name"),
            "datasets": [v.summary() for v in r["views"]],
            "unconfirmed_datasets": [{"dataset_id": d["dataset_id"], "name": d["name"], "status": d["status"]}
                                     for d in study.datasets if d["status"] != "confirmed"],
            "cross_dataset": {
                "sample_overlap": r["overlap"],
                "id_mapping": dict(r["id_mapping"], suggestions=r["suggestions"], ambiguous=r["ambiguous"]),
                "conflicts": r["conflicts"],
                "columns": r["columns"],
                "column_suggestions": r["column_suggestions"],
                "design_agreement": r["design_agreement"],
                "possible_id_reuse": r["possible_id_reuse"],
                "questions": r["questions"]},
            "ready_for_audit": r["ready_for_audit"], "blocking": r["blocking"]}


def _csv_bytes(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def write(study, r=None):
    r = r or compute(study)
    doc = session_schema(study, r)
    write_json(study.dir / "session_schema.json", doc)
    write_bytes(study.dir / "session_sample_table.csv", _csv_bytes(*unified_table(r)))
    return doc


def report(study):
    """The merge report (and the two session outputs, rewritten from the current state)."""
    return write(study)


def sample_table(study, only=None):
    """[{column: value}] of the unified table, for the audit (only=did: one dataset's samples)."""
    header, rows = unified_table(compute(study), only)
    return header, [dict(zip(header, x)) for x in rows]


# ------------------------------------------------------------------ decisions


def decide(study, item_id, decision, who="user", unified_id=None):
    """Record the user's decision on a suggestion (ms), conflict (mc) or question (mq)."""
    r = compute(study)
    st = _state(study)
    if item_id.startswith("ms"):
        s = next((x for x in r["suggestions"] if x["suggestion_id"] == item_id), None)
        if s is None:
            raise MergeError(f"Unknown ID suggestion '{item_id}'.")
        if decision not in ("confirm", "dismiss", "reopen"):
            raise MergeError("An ID suggestion is confirmed, dismissed or reopened.")
        rec = {"status": {"confirm": "confirmed", "dismiss": "dismissed", "reopen": "open"}[decision]}
        if decision == "confirm":
            rec["unified_id"] = unified_id or s["a"]["sample_id"]
            trial = dict(st["decisions"], **{item_id: rec})
            saved = st["decisions"]
            st["decisions"] = trial
            errs = compute(study)["id_mapping"]["errors"]
            st["decisions"] = saved
            if errs:
                raise MergeError("Confirming this would map two samples of one dataset to one ID: " + "; ".join(errs))
    elif item_id.startswith("mc"):
        c = next((x for x in r["conflicts"] if x["conflict_id"] == item_id), None)
        if c is None:
            raise MergeError(f"Unknown conflict '{item_id}'.")
        if decision == "reopen":
            rec = {"status": "open"}
        elif decision in c["options"]:
            rec = {"status": "resolved", "resolution": decision}
        else:
            raise MergeError(f"Choose one of {', '.join(c['options'])}.")
    elif item_id.startswith("mq"):
        q = next((x for x in r["questions"] if x["question_id"] == item_id), None)
        if q is None:
            raise MergeError(f"Unknown merge question '{item_id}'.")
        if decision == "dismiss":
            rec = {"status": "dismissed", "answer": None}
        elif decision == "reopen":
            rec = {"status": "open", "answer": None}
        elif decision in [o["option_id"] for o in q["options"]]:
            rec = {"status": "answered", "answer": decision}
        else:
            raise MergeError(f"Choose one of {', '.join(o['option_id'] for o in q['options'])}, or dismiss.")
    else:
        raise MergeError(f"'{item_id}' is not a merge item.")
    rec.update(by=who, at=now_iso())
    st["decisions"][item_id] = rec
    study.save()
    study.log("merge_decision", {"item_id": item_id, "decision": decision, "by": who,
                                 "unified_id": rec.get("unified_id")})
    return write(study)


def set_mapping_csv(study, raw, name="mapping.csv", who="user"):
    """Replace the uploaded mapping (validated first: nothing is applied when it is invalid)."""
    entries = parse_mapping_csv(raw, views_of(study))
    st = _state(study)
    st["mapping_csv"] = {"name": name, "sha256": sha256_bytes(raw), "n_rows": len(entries), "by": who,
                         "at": now_iso(), "entries": entries}
    write_bytes(study.dir / "merge" / "mapping.csv", raw)
    study.save()
    study.log("merge_mapping_uploaded", {"name": name, "sha256": st["mapping_csv"]["sha256"], "n_rows": len(entries),
                                         "by": who})
    return write(study)


def clear_mapping_csv(study, who="user"):
    st = _state(study)
    st["mapping_csv"] = None
    study.save()
    study.log("merge_mapping_cleared", {"by": who})
    return write(study)


def read_sample_table(study):
    """The written session_sample_table.csv as [{column: value}] (the audit reads this file)."""
    p = study.dir / "session_sample_table.csv"
    if not p.exists():
        write(study)
    header, rows = read_csv(p)
    return header, [dict(zip(header, x)) for x in rows]
