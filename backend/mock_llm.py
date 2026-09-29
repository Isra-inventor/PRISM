"""MockLLM: canned, rule-based proposals from the digest (LLM_PROVIDER=mock).

Used by the tests (which also inject deliberately wrong answers) and as an
offline demo. It reads only the digest, exactly like a real model would.
"""

from __future__ import annotations

import fnmatch
import json
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
    return g.get("columns") or (g.get("first_columns", []) + g.get("last_columns", []))


def _omics(names):
    joined = " ".join(names).lower()
    if re.search(r"seq\.|lfq|ibaq|intensity|protein|pg\.", joined) or all(
            re.match(r"^[opq]\d[a-z0-9]{3}\d", n.lower()) or re.match(r"^p\d{5}", n.lower()) for n in names):
        return "proteomics"
    return "metabolomics"


class MockLLM:
    @staticmethod
    def respond(system, prompt):
        digest = json.loads(prompt.split("\n", 1)[1])
        hints = digest["file"]["layout_hints"]
        fixed = digest.get("already_confirmed") or {}
        layout = fixed.get("layout")
        if not layout:
            ratio = hints.get("rows_to_block_columns_ratio")
            if hints.get("long_format_pattern"):
                layout = "long"
            elif ratio is not None and ratio < 1:
                layout = "samples_in_rows"
            else:
                layout = "samples_in_columns"
        groups, samples, fid, assays = [], [], [], {}
        file_omics = _omics([n for g in digest["groups"] for n in _names(g)])
        has_lfq = any("lfq" in (g.get("pattern") or "").lower() for g in digest["groups"])
        for g in digest["groups"]:
            names = _names(g)
            name = names[0] if names else ""
            prof = g.get("profile") or {}
            e = {"group_id": g["group_id"], "role": "unresolved", "assay_label": None, "label": "",
                 "block_role": None, "audit_kind": None, "marks_rows_as_suspect": False,
                 "confidence": 0.6, "evidence": "mock rule", "suggest_split": None}
            if g["kind"] == "numeric_block" and not re.search(r"scale|norm", (g.get("pattern") or "").lower()):
                om = _omics(names) if layout == "samples_in_rows" else file_omics
                label = om
                assays.setdefault(label, om)
                pat = (g.get("pattern") or "").lower()
                block_role = "primary" if "lfq" in pat else ("auxiliary" if has_lfq else None)
                logscale = (not prof.get("integer_valued") and (prof.get("p99") or 0) < 40 and (prof.get("median") or 0) > 5)
                e.update(role="value", assay_label=label, block_role=block_role,
                         label=f"{(g.get('pattern') or 'values').strip()}, apparently {'log' if logscale else 'linear'} "
                               f"scale (median {prof.get('median')})",
                         confidence=0.85, evidence=f"{g['n_columns']} numeric columns; median {prof.get('median')}")
                if layout == "samples_in_rows":
                    split = [n for n in (g.get("columns") or []) if _COVARIATE_NAMES.match(n)]
                    if split:
                        e.update(suggest_split=split, suggest_split_role="sample_metadata",
                                 suggest_split_audit_kind="covariate", suggest_split_label="numeric clinical covariate")
                for sn in g.get("sample_names_after_stripping_pattern") or []:
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
        })


def expand_sample_rules(rules, sample_ids):
    """Glob patterns / exact names -> {sample_id: rule}."""
    out = {}
    for r in rules:
        pat = r["pattern_or_sample"]
        for s in sample_ids:
            if s == pat or fnmatch.fnmatchcase(s, pat) or fnmatch.fnmatchcase(s.lower(), pat.lower()):
                out.setdefault(s, r)
    return out
