"""MockLLM: canned, rule-based proposals from the digest (LLM_PROVIDER=mock).

Used by the tests (which also inject deliberately wrong answers) and as an
offline demo. It reads only the digest, exactly like a real model would.
"""

from __future__ import annotations

import fnmatch
import json
import math
import re

# text columns: regex on the name -> (role, label, audit_kind, marks_rows_as_suspect)
_TEXT_RULES = [
    (r"reverse|decoy", ("feature_annotation", "decoy hit flag", None, True)),
    (r"contaminant", ("feature_annotation", "contaminant flag", None, True)),
    (r"only identified by site", ("feature_annotation", "identified only by site flag", None, True)),
    (r"^(sample.?type|type)$", ("sample_metadata", "sample type", "sample_type", False)),
    (r"gene", ("feature_annotation", "gene symbol", None, False)),
    (r"majority protein|protein.?ids?$|protein.?group|accession|^protein$", ("feature_annotation", "protein accession", None, False)),
    (r"protein.?name|description|entry.?name", ("feature_annotation", "protein name", None, False)),
    (r"sequence|peptide", ("feature_annotation", "peptide sequence", None, False)),
    (r"formula", ("feature_annotation", "molecular formula", None, False)),
    (r"adduct", ("feature_annotation", "ion adduct", None, False)),
    (r"hmdb|kegg|chebi|pubchem|inchi", ("feature_annotation", "metabolite database id", None, False)),
    (r"compound|metabolite|^name$|taxonom|genus|species", ("feature_annotation", "feature name", None, False)),
    (r"subject|patient|donor|participant", ("sample_metadata", "subject identifier", "subject_id", False)),
    (r"visit|time.?point|^day|^week|^time$", ("sample_metadata", "time point", "timepoint", False)),
    (r"batch", ("sample_metadata", "batch", "batch", False)),
    (r"plate|slide", ("sample_metadata", "plate / slide", "batch", False)),
    (r"group|severity|arm|treatment|condition|diagnosis|status", ("sample_metadata", "group-like variable", "group", False)),
    (r"sex|gender", ("sample_metadata", "sex", "covariate", False)),
    (r"rowcheck|check|flag", ("sample_metadata", "quality flag", "other", False)),
    (r"fasta|header", ("feature_annotation", "FASTA header", None, False)),
]
_NUM_RULES = [
    (r"m/z|^mz$|mass", ("feature_annotation", "m/z", None, False)),
    (r"retention|^rt$", ("feature_annotation", "retention time", None, False)),
    (r"score|q.?value|pep$|probability", ("feature_annotation", "identification score", None, False)),
    (r"coverage", ("feature_annotation", "sequence coverage", None, False)),
    (r"peptides", ("feature_annotation", "peptide count", None, False)),
    (r"^age$|bmi|cd4|iron|weight|height|crp", ("sample_metadata", "numeric clinical covariate", "covariate", False)),
    (r"scale|norm", ("sample_metadata", "technical scale factor", "other", False)),
    (r"order|injection", ("sample_metadata", "run order", "run_order", False)),
    (r"batch", ("sample_metadata", "batch", "batch", False)),
    (r"slide|plate", ("sample_metadata", "plate / slide", "batch", False)),
]
_COVARIATE_NAMES = re.compile(r"(?i)^(age|bmi|cd4.*|iron|weight|height|crp|sex)$")


def _first(name, rules):
    n = name.lower()
    for pat, res in rules:
        if re.search(pat, n):
            return res
    return None


def _names(g):
    return g["columns"]


def _omics(names):
    joined = " ".join(names).lower()
    if re.search(r"seq\.|lfq|ibaq|intensity|protein|pg\.", joined) or all(
            re.match(r"^[opq]\d[a-z0-9]{3}\d", n.lower()) or re.match(r"^p\d{5}", n.lower()) for n in names):
        return "proteomics"
    return "metabolomics"


def _family_key(c):
    """The mock's own reading of the shared-name facts: a prefix cut at its first
    digit, or a suffix cut after its last digit (e.g. 'LFQ intensity S', ' Peak area')."""
    keys = []
    p = (c.get("shared_prefix") or [{}])[0].get("text") or ""
    p = re.split(r"\d", p, maxsplit=1)[0]
    s = (c.get("shared_suffix") or [{}])[0].get("text") or ""
    s = re.split(r"\d", s)[-1] if s else ""
    for side, k in (("prefix", p), ("suffix", s)):
        if len(k.strip()) >= 3 and k != c["column"]:
            rest = c["column"][len(k):] if side == "prefix" else c["column"][:len(c["column"]) - len(k)]
            if re.search(r"\d", rest):  # the varying part looks like a sample name
                keys.append((len(k), side, k))
    return max(keys)[1:] if keys else None


def _clusters(cols):
    """Name-less numeric columns: split at gaps of more than a decade in typical value."""
    sub = sorted(((math.log10(c["median"]) if (c.get("median") or 0) > 0 else -9), k) for k, c in enumerate(cols))
    out, cur = [], []
    for lm, k in sub:
        if cur and lm - cur[-1][0] > 1.0:
            out.append([j for _, j in cur])
            cur = []
        cur.append((lm, k))
    if cur:
        out.append([j for _, j in cur])
    return out


def _group_columns(columns, layout):
    """-> [(key, [column digests])]: the mock's grouping proposal."""
    fam, rest = {}, []
    for c in columns:
        key = _family_key(c) if c["type"] == "numeric" else None
        if key:
            fam.setdefault(key, []).append(c)
        else:
            rest.append(c)
    groups = []
    for key, cs in fam.items():
        if len(cs) >= 2:
            groups.append((key[1], cs))
        else:
            rest.extend(cs)
    singles = []
    if layout == "samples_in_rows":
        free = [c for c in rest if c["type"] == "numeric" and not _first(c["column"], _NUM_RULES)
                and not _COVARIATE_NAMES.match(c["column"]) and not re.match(r"(?i)^(row.?id|id)$", c["column"])]
        for cl in _clusters(free):
            if len(cl) >= 2:
                groups.append(("", [free[k] for k in cl]))
            else:
                singles.append(free[cl[0]])
        rest = [c for c in rest if c not in free] + singles
    groups += [(None, [c]) for c in rest]
    return sorted(groups, key=lambda g: min(c["position"] for c in g[1]))


def _pseudo_group(gid, key, cs):
    names = [c["column"] for c in cs]
    if len(cs) == 1:
        return {"group_id": gid, "kind": "single_column", "type": cs[0]["type"], "columns": names,
                "pattern": None, "profile": cs[0], "n_columns": 1}
    meds = sorted(c.get("median") or 0 for c in cs)
    prof = {"median": meds[len(meds) // 2], "p99": max(c.get("p99") or 0 for c in cs),
            "integer_valued": all(c.get("integer_valued") for c in cs)}
    samples = [n[len(key):] if n.startswith(key or "\0") else (n[:len(n) - len(key)] if key and n.endswith(key) else n)
               for n in names]
    return {"group_id": gid, "kind": "numeric_block", "type": "numeric", "columns": names, "pattern": key,
            "profile": prof, "n_columns": len(cs), "sample_names": [x.strip(" _.-") for x in samples]}


class MockLLM:
    @staticmethod
    def respond(system, prompt):
        if prompt.startswith("Literature"):
            return MockLLM.literature(json.loads(prompt.split("\n", 1)[1]))
        if prompt.startswith("Consolidation"):
            return MockLLM.consolidate(json.loads(prompt.split("\n", 1)[1]))
        if prompt.startswith("Chat"):
            return MockLLM.chat(json.loads(prompt.split("\n", 1)[1]))
        digest = json.loads(prompt.split("\n", 1)[1])
        hints = digest["file"]["layout_hints"]
        fixed = digest.get("already_confirmed") or {}
        layout = fixed.get("layout")
        if not layout:
            ratio = hints.get("rows_to_numeric_columns_ratio")
            if hints.get("long_format_pattern"):
                layout = "long"
            elif ratio is not None and ratio < 1:
                layout = "samples_in_rows"
            else:
                layout = "samples_in_columns"
        proposed = [_pseudo_group(f"m{k}", key, cs)
                    for k, (key, cs) in enumerate(_group_columns(digest["columns"], layout), 1)]
        groups, samples, fid, assays = [], [], [], {}
        file_omics = _omics([c["column"] for c in digest["columns"]])
        for g in proposed:
            names = _names(g)
            name = names[0] if names else ""
            prof = g.get("profile") or {}
            e = {"group_id": g["group_id"], "columns": names, "role": "unresolved", "assay_label": None, "label": "",
                 "audit_kind": None, "marks_rows_as_suspect": False,
                 "confidence": 0.6, "evidence": "mock rule", "suggest_split": None}
            if g["kind"] == "numeric_block" and not re.search(r"scale|norm", (g.get("pattern") or "").lower()):
                om = _omics(names) if layout == "samples_in_rows" else file_omics
                label = om
                assays.setdefault(label, om)
                logscale = (not prof.get("integer_valued") and (prof.get("p99") or 0) < 40 and (prof.get("median") or 0) > 5)
                e.update(role="value", assay_label=label,
                         label=f"{(g.get('pattern') or '').strip() or om} values, apparently "
                               f"{'log' if logscale else 'linear'} scale",
                         confidence=0.85, evidence=f"{g['n_columns']} numeric columns; median {prof.get('median')}")
                for sn in g.get("sample_names") or []:
                    for pat, lab in (("qc", "QC injection"), ("blank", "blank"), ("pool", "pooled sample"),
                                     ("calib", "calibrator")):
                        if sn.lower().startswith(pat):
                            samples.append({"pattern_or_sample": sn, "label": lab, "is_study_sample": False,
                                            "confidence": 0.8, "evidence": f"sample name starts with '{pat}'"})
            elif g["kind"] == "numeric_block":
                if layout == "samples_in_rows":
                    e.update(role="sample_metadata", audit_kind="other", label="technical scale factors",
                             confidence=0.7, evidence="scale-factor-like names")
                else:
                    e.update(role="feature_annotation", label="numeric feature annotations", confidence=0.5)
            elif layout == "long" and g["type"] == "numeric":
                e.update(role="value", assay_label="assay", label=f"{name} (single value column)",
                         confidence=0.7, evidence="the single numeric column of a long table")
                assays.setdefault("assay", "unknown")
            elif layout == "long" and re.search(r"(?i)sample|run|file", name):
                e.update(role="sample_id", label="sample name", confidence=0.7, evidence=f"column name '{name}'")
            else:
                rules = _NUM_RULES if g["type"] == "numeric" else _TEXT_RULES
                rule = _first(name, rules)
                uniq = prof.get("unique_ratio") == 1
                if g["type"] == "numeric" and re.match(r"(?i)^(row.?id|id)$", name):
                    rule = ("feature_id", "row id", None, False) if layout != "samples_in_rows" else ("ignore", "row index", None, False)
                if g["type"] != "numeric" and layout == "samples_in_rows" and uniq and re.search(r"(?i)sample|^id$", name):
                    rule = ("sample_id", "sample identifier", None, False)
                if g["type"] != "numeric" and layout == "samples_in_columns" and uniq and re.search(r"(?i)otu|^#|^id_ref$|feature", name):
                    rule = ("feature_id", "feature identifier", None, False)
                if rule:
                    role, label, kind, suspect = rule
                    if layout == "samples_in_rows" and role == "feature_annotation":
                        role, kind = "sample_metadata", "other"
                    if layout == "samples_in_columns" and role == "sample_metadata":
                        role, kind = "feature_annotation", None
                    e.update(role=role, label=label, audit_kind=kind, marks_rows_as_suspect=suspect,
                             confidence=0.8, evidence=f"column name '{name}'")
                    if label == "protein accession" and not fid and (uniq or layout == "long") and layout != "samples_in_rows":
                        fid = [g["group_id"]]
                        e["role"] = "feature_id"
                    if kind == "sample_type":
                        for v in prof.get("values") or []:
                            study = v["value"].lower() in ("sample", "study")
                            samples.append({"pattern_or_sample": v["value"], "label": v["value"], "is_study_sample": study,
                                            "confidence": 0.7, "evidence": "sample type column value"})
                elif g["type"] == "numeric" and layout == "samples_in_columns":
                    e.update(role="feature_annotation", label="numeric feature annotation", confidence=0.5)
            if e["role"] == "feature_id" and not fid:
                fid = [g["group_id"]]
            elif e["role"] == "feature_id" and g["group_id"] not in fid:
                e["role"] = "feature_annotation"
            groups.append(e)
        if not assays:
            assays["assay"] = "unknown"
        return json.dumps({
            "layout": {"value": layout, "confidence": 0.8, "evidence": "mock: from layout hints"},
            "assays": [{"assay_label": lab, "omics_type": om, "source_software": "unknown",
                        "in_supported_scope": "yes" if om in ("proteomics", "metabolomics") else "unsure",
                        "scope_reason": "mock", "feature_identity": {"group_ids": fid, "composite": len(fid) > 1},
                        "confidence": 0.7, "evidence": "mock: from column names"} for lab, om in assays.items()],
            "groups": groups,
            "samples": samples,
            "clarifying_questions": [],
            "propose_merge": [],
        })

    @staticmethod
    def chat(payload):
        """Very small message reader (the real AI does this properly). Reads only the payload."""
        text = payload["message"].lower()
        cols = [c["column"] for c in payload.get("columns", [])] + \
            [c for g in payload["groups"] for c in g["columns"] if not c.startswith("...")]
        stop = {"all", "the", "columns", "column", "from", "output", "outputs", "in", "of", "and", "any", "these",
                "those", "except", "but", "keep"}
        m = re.search(r"(?:exclude|remove|drop|delete|leave out|don'?t include|do not include)\s+(.*)", text)
        if m:
            rest = m.group(1)
            keep_m = re.search(r"(?:except|but|apart from)\s+(?:the\s+)?(.*)", rest)
            head = rest[:keep_m.start()] if keep_m else rest
            except_cols = []
            if keep_m:
                words = keep_m.group(1)
                except_cols = [c for c in dict.fromkeys(cols) if c.lower() in words or (
                    "id" in words.split() and any(g["role"] == "feature_id" and c in g["columns"] for g in payload["groups"]))]
            if "annotation" in head:
                target = {"selector": {"role": "feature_annotation"}, "except_columns": except_cols}
            else:
                terms = [w.rstrip("s") for w in re.findall(r"[a-z0-9/]+", head) if w not in stop and len(w) > 2]
                hit = [c for c in dict.fromkeys(cols) if any(w in c.lower() for w in terms)]
                hit += [c for g in payload["groups"] if any(w in (g.get("label") or "").lower() for w in terms)
                        for c in g["columns"] if not c.startswith("...") and c not in hit]
                if not hit:
                    return json.dumps({"reply": "I found no column matching that.", "patches": [], "questions": []})
                target = {"selector": {"columns": hit}, "except_columns": except_cols}
            return json.dumps({"reply": "I've prepared a change: leave these columns out of the outputs.",
                               "patches": [{"patch_id": "p1", "op": "set_keep", "target": target, "args": {"keep": False},
                                            "reason": "You asked to leave them out.",
                                            "consequences": ["A later QC audit could use some of these columns."]}],
                               "questions": []})
        m = re.search(r"mark\s+(\S+)\s+(?:samples?\s+)?as\s+(non-study|not study|qc|blank)", text)
        if m:
            pat = payload["message"].split()[1]
            return json.dumps({"reply": f"I've prepared a change for {pat}.", "questions": [],
                               "patches": [{"patch_id": "p1", "op": "set_sample_label",
                                            "target": {"selector": {"samples": [pat]}},
                                            "args": {"is_study_sample": False, "label": m.group(2)}, "reason": "You asked."}]})
        return json.dumps({"reply": "mock: I did not understand that.", "patches": [], "questions": []})

    @staticmethod
    def consolidate(payload):
        groups = payload["groups"]
        if payload.get("user_hint"):
            same = len({g["role"] for g in groups}) == 1
            return json.dumps({"cross_chunk_merges": [{"group_ids": [g["group_id"] for g in groups],
                                                       "reason": "mock: same role"}] if same else [],
                               "comment": "mock: roles agree" if same else "mock: different roles"})
        by = {}
        for g in groups:
            if g["role"] == "value":
                by.setdefault((g["role"], g["label"]), []).append(g["group_id"])
        return json.dumps({"cross_chunk_merges": [{"group_ids": ids, "reason": "mock: same label across chunks"}
                                                  for ids in by.values() if len(ids) > 1]})


def _literature(payload):
    """Cite the first passage mentioning the block's name, quoting its first words."""
    passages = payload["passages"]
    blocks = []
    for b in payload["file"]["value_blocks"]:
        name = (b.get("name") or "").lower()
        hit = next((p for p in passages if name and name in p["text"].lower()), None)
        cites = [{"passage_id": hit["id"], "quote": " ".join(hit["text"].split()[:8])}] if hit else []
        yes = hit is not None and re.search(r"quantif|analy[sz]ed|used for", hit["text"], re.I)
        blocks.append({"group_id": b["group_id"], "description": f"{b.get('name') or 'values'} (mock)",
                       "typical_use": "mock: from the first matching passage" if hit else "not covered",
                       "suggested_for_analysis": "yes" if yes else "unsure",
                       "reason": "mock rule", "citations": cites})
    first = passages[0] if passages else None
    return json.dumps({
        "summary": "mock summary",
        "summary_citations": [{"passage_id": first["id"], "quote": " ".join(first["text"].split()[:8])}] if first else [],
        "blocks": blocks,
        "for_later_steps": [{"topic": "normalization", "passage_ids": [p["id"] for p in passages
                                                                       if "normaliz" in p["text"].lower()][:2]}],
    })


MockLLM.literature = staticmethod(_literature)


def expand_sample_rules(rules, sample_ids):
    """Glob patterns / exact names -> {sample_id: rule}."""
    out = {}
    for r in rules:
        pat = r["pattern_or_sample"]
        for s in sample_ids:
            if s == pat or fnmatch.fnmatchcase(s, pat) or fnmatch.fnmatchcase(s.lower(), pat.lower()):
                out.setdefault(s, r)
    return out
