"""MockLLM: canned, rule-based proposals from the digest (LLM_PROVIDER=mock).

Used by the tests (which also inject deliberately wrong answers) and as an
offline demo. It reads only the digest, exactly like a real model would.
"""

from __future__ import annotations

import fnmatch
import json
import re

_TEXT_RULES = [
    (r"reverse|decoy", ("feature_annotation", "flag_decoy")),
    (r"contaminant", ("feature_annotation", "flag_contaminant")),
    (r"only identified by site", ("feature_annotation", "flag_other")),
    (r"^(sample.?type|type)$", ("sample_metadata", "sample_type")),
    (r"gene", ("feature_annotation", "gene_symbol")),
    (r"majority protein|protein.?ids?$|protein.?group|accession|^protein$", ("feature_annotation", "protein_accession")),
    (r"protein.?name|description|entry.?name", ("feature_annotation", "protein_name")),
    (r"sequence|peptide", ("feature_annotation", "peptide_sequence")),
    (r"formula", ("feature_annotation", "molecular_formula")),
    (r"adduct", ("feature_annotation", "adduct")),
    (r"hmdb|kegg|chebi|pubchem|inchi", ("feature_annotation", "metabolite_db_id")),
    (r"compound|metabolite|^name$", ("feature_annotation", "metabolite_name")),
    (r"subject|patient|donor|participant", ("sample_metadata", "subject_id")),
    (r"visit|time.?point|^day|^week|^time$", ("sample_metadata", "timepoint")),
    (r"batch", ("sample_metadata", "batch")),
    (r"plate|slide", ("sample_metadata", "plate_or_slide")),
    (r"group|severity|arm|treatment|condition|diagnosis|status", ("sample_metadata", "group")),
    (r"sex|gender", ("sample_metadata", "covariate_categorical")),
    (r"rowcheck|check|flag", ("sample_metadata", "other_sample_metadata")),
    (r"fasta|header", ("feature_annotation", "other_annotation")),
]
_NUM_RULES = [
    (r"m/z|^mz$|mass", ("feature_annotation", "mz")),
    (r"retention|^rt$", ("feature_annotation", "retention_time")),
    (r"score|q.?value|pep$|probability", ("feature_annotation", "identification_score")),
    (r"coverage", ("feature_annotation", "sequence_coverage")),
    (r"peptides", ("feature_annotation", "peptide_count")),
    (r"^age$|bmi|cd4|iron|weight|height|crp", ("sample_metadata", "covariate_numeric")),
    (r"scale|norm", ("sample_metadata", "technical_numeric")),
    (r"order|injection", ("sample_metadata", "run_order")),
    (r"batch", ("sample_metadata", "batch")),
    (r"slide|plate", ("sample_metadata", "plate_or_slide")),
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
    if re.search(r"seq\.|lfq|ibaq|intensity|^p\d|protein|pg\.", joined) or all(
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
        groups, sample_types, fid = [], [], []
        omics_seen = []
        for g in digest["groups"]:
            names = _names(g)
            name = names[0] if names else ""
            prof = g.get("profile") or {}
            entry = {"group_id": g["group_id"], "role": "unresolved", "kind": None, "detail": None,
                     "measurement_type": None, "scale": None, "omics_type": None, "label": None,
                     "confidence": 0.6, "evidence": "mock rule", "suggest_split": None}
            if g["kind"] == "numeric_block" and not _first(g.get("pattern") or "", [(r"scale|norm", 1)]):
                om = _omics(names)
                omics_seen.append(om)
                logscale = (not prof.get("integer_valued") and (prof.get("p99") or 0) < 40
                            and (prof.get("median") or 0) > 5)
                entry.update(role="value", omics_type=om, label=(g.get("pattern") or "values").strip(),
                             measurement_type="intensity", scale="log2" if logscale else "linear",
                             confidence=0.85, evidence=f"{g['n_columns']} numeric columns sharing a pattern")
                if layout == "samples_in_rows":
                    split = [n for n in (g.get("columns") or []) if _COVARIATE_NAMES.match(n)]
                    if split:
                        entry.update(suggest_split=split, suggest_split_role="sample_metadata",
                                     suggest_split_kind="covariate_numeric")
                for sn in g.get("sample_names_after_stripping_pattern") or []:
                    for pat, t in (("qc", "qc"), ("blank", "blank"), ("pool", "pool"), ("calib", "calibrator")):
                        if sn.lower().startswith(pat):
                            sample_types.append({"pattern_or_sample": sn, "type": t, "confidence": 0.8,
                                                 "evidence": f"sample name starts with '{pat}'"})
            elif g["kind"] == "numeric_block":
                entry.update(role="sample_metadata" if layout == "samples_in_rows" else "feature_annotation",
                             kind="technical_numeric" if layout == "samples_in_rows" else "other_annotation",
                             confidence=0.7, evidence="scale-factor-like columns")
            elif layout == "long" and g["type"] == "numeric":
                entry.update(role="value", measurement_type="intensity", scale="linear", label=name,
                             omics_type="unknown", confidence=0.7, evidence="the single numeric column of a long table")
            elif layout == "long" and re.search(r"(?i)sample|run|file", name):
                entry.update(role="sample_id", confidence=0.7, evidence=f"column name '{name}'")
            elif g["type"] == "numeric":
                rule = _first(name, _NUM_RULES)
                if re.match(r"(?i)^(row.?id|id)$", name):
                    rule = ("feature_id", None) if layout != "samples_in_rows" else ("ignore", None)
                if rule:
                    role, kind = rule
                    if layout == "samples_in_rows" and role == "feature_annotation":
                        role, kind = "sample_metadata", "other_sample_metadata"
                    if layout != "samples_in_rows" and role == "sample_metadata" and kind != "covariate_numeric":
                        role, kind = "feature_annotation", "other_annotation"
                    entry.update(role=role, kind=kind, confidence=0.8, evidence=f"column name '{name}'")
                elif layout == "samples_in_columns":
                    entry.update(role="feature_annotation", kind="other_annotation", confidence=0.5,
                                 evidence="single numeric column in a feature table")
            else:
                rule = _first(name, _TEXT_RULES)
                uniq = prof.get("unique_ratio") == 1
                if layout == "samples_in_rows" and uniq and re.search(r"(?i)sample|^id$", name):
                    rule = ("sample_id", None)
                if rule:
                    role, kind = rule
                    if role == "feature_annotation" and layout == "samples_in_rows":
                        role, kind = "sample_metadata", "other_sample_metadata"
                    if kind == "timepoint":
                        entry["detail"] = "ordinal_label"
                    entry.update(role=role, kind=kind, confidence=0.8, evidence=f"column name '{name}'")
                    if kind == "protein_accession" and not fid and (uniq or layout == "long") \
                            and layout != "samples_in_rows":
                        fid = [g["group_id"]]
                        entry.update(role="feature_id", kind=None)
                    if kind == "sample_type":
                        for v in prof.get("values") or []:
                            t = {"qc": "qc", "calibrator": "calibrator", "buffer": "blank", "blank": "blank",
                                 "pool": "pool", "sample": "study"}.get(v["value"].lower())
                            if t:
                                sample_types.append({"pattern_or_sample": v["value"], "type": t,
                                                     "confidence": 0.7, "evidence": "sample type column value"})
            if entry["role"] == "feature_id" and not fid:
                fid = [g["group_id"]]
            groups.append(entry)
        omics = max(set(omics_seen), key=omics_seen.count) if omics_seen else "unknown"
        return json.dumps({
            "layout": {"value": layout, "confidence": 0.8, "evidence": "mock: from layout hints"},
            "omics_type": {"value": omics, "confidence": 0.7, "evidence": "mock: from column names"},
            "source_software": {"value": "unknown", "confidence": 0.0, "evidence": "mock"},
            "feature_identity": {"group_ids": fid, "composite": len(fid) > 1, "confidence": 0.7,
                                 "evidence": "mock: identifier-like column"},
            "groups": groups,
            "sample_types": sample_types,
            "clarifying_questions": [],
        })


def expand_sample_types(rules, sample_ids):
    """Glob patterns / exact names -> {sample_id: rule}."""
    out = {}
    for r in rules:
        pat = r["pattern_or_sample"]
        for s in sample_ids:
            if s == pat or fnmatch.fnmatchcase(s, pat) or fnmatch.fnmatchcase(s.lower(), pat.lower()):
                out.setdefault(s, r)
    return out
