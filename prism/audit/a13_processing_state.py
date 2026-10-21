"""A13. Processing state (Build Request v4, section 4.2): the user's declared processing_history
(Step 0) against observed evidence, giving the confirmed processing state every precondition uses.

Per item: declared (yes / no / unsure / absent), observed_evidence (templated sentences with numbers
from A2, A3, A4), match (matches | does_not_match | not_assessed), confirmed (null until the user
answers; overrides.json kind processing_state). batch_corrected and removed_before_upload stay
not_assessed. A `confirm_processing_state` question is created for every unconfirmed
does_not_match, and for log_transformed / normalized when they were not declared (G5 and S1 need
them). Platform notes are shown next to a mismatch, never applied.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .a02_scale import median_scaling
from .a04_missingness import floor_ties
from .finding import Finding

AUDIT_ID, CODE = "A13", "processing_state"
ITEMS = ("log_transformed", "normalized", "imputed", "batch_corrected", "removed_before_upload")
STEP0_KEY = {"removed_before_upload": "features_or_samples_removed_before_upload"}
NOT_ASSESSED = ("batch_corrected", "removed_before_upload")
ASK_WHEN_UNDECLARED = ("log_transformed", "normalized")
LOG_LIKE = ("continuous_symmetric_or_log_like", "continuous_signed")


def declared(hist, item):
    v = (hist or {}).get(item, (hist or {}).get(STEP0_KEY.get(item, ""), None))
    if isinstance(v, dict):
        v = v.get("answer")
    if v in ("yes", "no"):
        return v
    if v in ("not_sure", "unsure"):
        return "unsure"
    return "absent"


def confirmed(overrides, dataset_id, item):
    for o in reversed(overrides or []):
        if o.get("kind") == "processing_state" and o.get("item") == item and o.get("dataset") in (None, "", dataset_id):
            return o.get("value")
    return None


def platform_notes(schema, feature_columns):
    import yaml
    doc = yaml.safe_load((Path(__file__).resolve().parent / "platform_notes.yaml").read_text(encoding="utf-8"))
    text = " ".join(str(a.get(k, "")) for a in schema.get("assays", []) for k in ("assay_label", "omics_type",
                                                                                   "source_software")).lower()
    text += " " + str(schema.get("source_file", "")).lower()
    cols = {str(c) for c in feature_columns}
    out = []
    for n in doc.get("notes", []):
        d = n.get("detect", {})
        hit = bool(cols & set(d.get("feature_columns_any", []))) or any(t in text for t in d.get("text_any", []))
        if hit:
            out.append({k: n[k] for k in ("id", "platform", "items", "text", "verify_with_vendor_documentation",
                                          "evidence", "evidence_level")})
    return out


def evidence(unit, params, results=None):
    """The observed facts A13 compares with the declaration (numbers from A2, A3, A4)."""
    X = unit.X
    ms = median_scaling(X, params["median_scaling_mad"])
    tied, _, _, _ = floor_ties(X, params["floor_min_observed"])
    floor_share = float(tied.mean()) if tied.size else 0.0
    Y = unit.Y
    with np.errstate(invalid="ignore"):
        resid = Y - np.nanmedian(Y, axis=1, keepdims=True)
        smed = np.nanmedian(resid, axis=0)
    smed = smed[np.isfinite(smed)]
    spread = float(np.median(np.abs(smed - np.median(smed)))) if smed.size else None
    a3 = ((results or {}).get(unit.dataset_id, {}).get(unit.unit_id, {}) or {}).get("a3_slope")
    return {"scale_class": unit.scale, "negative_values": bool(unit.scale_evidence.get("negative_present")),
            "median_skew_x": unit.scale_evidence.get("median_skew_x"), "median_skew_y": unit.scale_evidence.get("median_skew_y"),
            "median_scaling_signature": bool(ms and ms["signature"]),
            "mad_log2_feature_medians": ms["mad_log2_feature_medians"] if ms else None,
            "sample_median_spread": spread,
            "narrow_sample_median_spread": spread is not None and spread < params["narrow_median_spread"],
            "floor_tied_share": floor_share, "n_floor_tied": int(tied.sum()), "of": int(tied.size),
            "has_nan": bool(np.isnan(X).any()), "a3_slope": a3}


def _f(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}g}" if isinstance(x, float) else str(x))


def assess(item, dec, ev, params):
    """-> (observed sentences, match)."""
    if item in NOT_ASSESSED:
        return ["This item cannot be assessed from the values."], "not_assessed"
    if item == "log_transformed":
        log_like = ev["scale_class"] in LOG_LIKE
        s = [f"A2 scale class: {ev['scale_class']} (median feature skewness {_f(ev['median_skew_x'])} on the values, "
             f"{_f(ev['median_skew_y'])} after a diagnostic log2)."]
        if ev["negative_values"]:
            s.append("Negative values are present.")
        if ev.get("a3_slope") is not None:
            s.append(f"A3 log SD on log mean slope b = {_f(ev['a3_slope'])}.")
        if ev["scale_class"] not in LOG_LIKE + ("continuous_linear",):
            return s, "not_assessed"
        if dec == "yes":
            return s, "matches" if log_like else "does_not_match"
        if dec == "no":
            return s, "does_not_match" if log_like else "matches"
        return s, "not_assessed"
    if item == "normalized":
        sig = ev["median_scaling_signature"] or ev["narrow_sample_median_spread"]
        s = [f"MAD of log2 per-feature medians {_f(ev['mad_log2_feature_medians'])} (median-scaling signature below "
             f"{params['median_scaling_mad']}: {'present' if ev['median_scaling_signature'] else 'absent'}).",
             f"MAD of sample medians (log2) {_f(ev['sample_median_spread'])} (reads as normalized below "
             f"{params['narrow_median_spread']}: {'yes' if ev['narrow_sample_median_spread'] else 'no'})."]
        if dec == "no":
            return s, "does_not_match" if sig else "matches"
        if dec == "yes":
            return s, "matches" if sig else "not_assessed"
        return s, "not_assessed"
    if item == "imputed":
        floor = ev["floor_tied_share"] >= params["floor_tie_share_indicator"]
        s = [f"{ev['n_floor_tied']} of {ev['of']} features ({100 * ev['floor_tied_share']:.1f}%) have two or more samples "
             f"at their minimum (floor ties; indicator from {100 * params['floor_tie_share_indicator']:.0f}%).",
             "The matrix has missing values." if ev["has_nan"] else "The matrix has no missing values."]
        if dec == "no":
            return s, "does_not_match" if floor else "matches"
        if dec == "yes":
            return s, "matches" if (floor or not ev["has_nan"]) else "not_assessed"
        return s, "not_assessed"
    return [], "not_assessed"


def run(unit, ctx, params, results):
    f = Finding(AUDIT_ID, CODE, unit).feeds("transformation", "normalization", "missingness_handling")
    if (getattr(ctx, "state", None) or {}).get("parent"):
        return f.status("not_applicable", "the processing state describes the Step 0 output (S0); this state has PRISM "
                                          "ops, so A13 is read from S0")
    hist = ctx.schema.get("processing_history") or {}
    ev = evidence(unit, params, results)
    notes = platform_notes(ctx.schema, ctx.output.feature_table.get("columns", []))
    rows, questions = [], []
    for item in ITEMS:
        dec = declared(hist, item)
        sentences, match = assess(item, dec, ev, params)
        conf = confirmed(ctx.overrides, ctx.dataset_id, item)
        pn = [n for n in notes if item in n["items"]]
        row = {"item": item, "declared": dec, "observed_evidence": sentences, "match": match, "confirmed": conf,
               "platform_notes": pn}
        row["effective"] = conf if conf in ("yes", "no") else (dec if match == "matches" else None)
        rows.append(row)
        ask = conf is None and (match == "does_not_match" or (item in ASK_WHEN_UNDECLARED and dec in ("absent", "unsure")))
        if ask:
            questions.append({"kind": "confirm_processing_state", "question_id": f"cps:{ctx.dataset_id}:{unit.unit_id}:{item}",
                              "dataset": ctx.dataset_id, "assay": unit.unit_id, "item": item, "declared": dec,
                              "match": match, "evidence": sentences, "platform_notes": pn,
                              "reason": "the declaration does not match the evidence" if match == "does_not_match"
                              else "not declared; G5 and S1 need it",
                              "text": f"Was this data {item.replace('_', ' ')} before upload?",
                              "options": ["yes", "no", "unsure"]})
        if match == "does_not_match":
            f.indicate("processing_state_mismatch", {"item": item, "declared": dec, "evidence": sentences},
                       item=item.replace("_", " "), declared=dec, confirmed=conf or "not yet")
    M = f.measures
    M["items"] = rows
    M["observed"] = ev
    M["questions"] = questions
    M["unconfirmed_mismatches"] = [r["item"] for r in rows if r["match"] == "does_not_match" and r["confirmed"] is None]
    M["gate"] = {"blocks": ["G5", "S1"] if M["unconfirmed_mismatches"] else [],
                 "text": "Insufficient metadata: confirm processing state" if M["unconfirmed_mismatches"] else None}
    M["platform_notes"] = notes
    f.method("declared processing history (Step 0) against A2 scale class, the median-scaling signature, sample "
             "median spread and floor ties; confirmations from overrides.json", n_used=unit.n)
    return f
