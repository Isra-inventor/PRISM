"""G2 / A13: what was done to the values before upload? Your answers (step 9) against the evidence.

For each of the five step-9 questions, per block:
    declared   your answer: yes | no | unsure
    observed   sentences with the numbers the evidence rests on
    match      matches | does_not_match | not_assessed
    confirmed  empty until you answer again after seeing a mismatch (a.confirm_processing_state)

What the evidence is:
    log_transformed   the A2 scale class (negative values or "log-like" -> log; "linear" -> not log)
    normalized        the median-scaling signature (A2), and how close the sample medians are to each
                      other (MAD of log2 sample medians below 0.05 reads as normalized; heuristic)
    imputed           floor ties: proteins where two or more samples sit exactly at the protein's
                      minimum value (a sign of values set to a floor, e.g. a detection limit or an
                      imputed minimum); and whether any value is missing
    batch_corrected, features_or_samples_removed_before_upload: cannot be seen in the values
                      (not_assessed)

Absence of a signature is not proof: "declared yes, no signature" is not_assessed, not a mismatch.
Platform notes (platform_notes.yaml) are shown next to a mismatch; they never decide.

The gate: while a mismatch is unconfirmed, G5 (log transform) and S1 (normalization) are
"Insufficient metadata: confirm processing state". The audits still run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .finding import Finding
from .scale import classify, diagnostic_log, median_scaling

ITEMS = ("log_transformed", "normalized", "imputed", "batch_corrected", "features_or_samples_removed_before_upload")
NOT_ASSESSED = ("batch_corrected", "features_or_samples_removed_before_upload")
ASK_WHEN_UNDECLARED = ("log_transformed", "normalized")      # G5 and S1 need these
LOG_LIKE = ("continuous_symmetric_or_log_like", "continuous_signed")


def floor_ties(X: np.ndarray, min_obs: int) -> tuple[np.ndarray, np.ndarray]:
    """Per protein with >= min_obs observed values: True when >= 2 samples sit exactly at its minimum.
    Exact equality is meaningful because values are copied unchanged from the file."""
    ok = np.isfinite(X)
    n_obs = ok.sum(1)
    with np.errstate(invalid="ignore"):
        m = np.where(n_obs > 0, np.nanmin(np.where(ok, X, np.inf), axis=1), np.nan)
    t = (ok & (X == m[:, None])).sum(1)
    return (n_obs >= min_obs) & (t >= 2), n_obs >= min_obs


def platform_notes(schema) -> list[dict]:
    doc = yaml.safe_load((Path(__file__).parent / "platform_notes.yaml").read_text(encoding="utf-8"))
    cols = {c for g in schema.groups_with("feature_id") + schema.groups_with("feature_annotation") for c in g.columns}
    text = " ".join(str(a.get(k, "")) for a in schema.assays for k in ("assay_label", "omics_type", "source_software"))
    text = (text + " " + " ".join(g.label for g in schema.groups_with("value"))).lower()
    out = []
    for n in doc["notes"]:
        d = n.get("detect", {})
        if cols & set(d.get("feature_columns_any", [])) or any(t in text for t in d.get("text_any", [])):
            out.append({k: n[k] for k in ("id", "platform", "items", "text", "verify_with_vendor_documentation",
                                          "evidence_level")})
    return out


def evidence(X: np.ndarray, cls: str, params: dict) -> dict:
    ms = median_scaling(X, params["median_scaling_mad"])
    tied, eligible = floor_ties(X, params["floor_min_observed"])
    Y = diagnostic_log(X)[0]
    with np.errstate(invalid="ignore"):
        smed = np.nanmedian(Y - np.nanmedian(Y, axis=1, keepdims=True), axis=0)
    smed = smed[np.isfinite(smed)]
    spread = float(np.median(np.abs(smed - np.median(smed)))) if smed.size else None
    return {"scale_class": cls, "median_scaling_signature": bool(ms and ms["signature"]),
            "mad_log2_protein_medians": ms["mad_log2_protein_medians"] if ms else None,
            "sample_median_spread_log2": round(spread, 4) if spread is not None else None,
            "narrow_sample_median_spread": spread is not None and spread < params["narrow_median_spread"],
            "n_floor_tied": int(tied.sum()), "n_eligible": int(eligible.sum()),
            "floor_tied_share": float(tied.sum() / eligible.sum()) if eligible.sum() else 0.0,
            "has_missing": bool(np.isnan(X).any())}


def _declared(v) -> str:
    return {"yes": "yes", "no": "no", "not_sure": "unsure"}.get(v, "absent")


def assess(item: str, dec: str, ev: dict, p: dict) -> tuple[list[str], str]:
    if item in NOT_ASSESSED:
        return ["This cannot be seen in the values."], "not_assessed"
    if item == "log_transformed":
        log_like = ev["scale_class"] in LOG_LIKE
        s = [f"A2 scale class: {ev['scale_class']}."]
        if ev["scale_class"] not in LOG_LIKE + ("continuous_linear",) or dec not in ("yes", "no"):
            return s, "not_assessed"
        return s, "matches" if (dec == "yes") == log_like else "does_not_match"
    if item == "normalized":
        sig = ev["median_scaling_signature"] or ev["narrow_sample_median_spread"]
        s = [f"MAD of log2 per-protein medians {ev['mad_log2_protein_medians']} (shared-median signature below "
             f"{p['median_scaling_mad']}: {'present' if ev['median_scaling_signature'] else 'absent'}).",
             f"MAD of log2 sample medians {ev['sample_median_spread_log2']} (reads as normalized below "
             f"{p['narrow_median_spread']}: {'yes' if ev['narrow_sample_median_spread'] else 'no'})."]
        if dec == "no":
            return s, "does_not_match" if sig else "matches"
        if dec == "yes":
            return s, "matches" if sig else "not_assessed"
        return s, "not_assessed"
    if item == "imputed":
        floor = ev["floor_tied_share"] >= p["floor_tie_share_indicator"]
        s = [f"{ev['n_floor_tied']} of {ev['n_eligible']} proteins ({100 * ev['floor_tied_share']:.1f}%) have two or "
             f"more samples exactly at their minimum (floor ties; counted as present from "
             f"{100 * p['floor_tie_share_indicator']:.0f}%).",
             "Some values are missing." if ev["has_missing"] else "No value is missing."]
        if dec == "no":
            return s, "does_not_match" if floor else "matches"
        if dec == "yes":
            return s, "matches" if (floor or not ev["has_missing"]) else "not_assessed"
        return s, "not_assessed"
    return [], "not_assessed"


def run(audit) -> Finding:
    s = audit.schema
    f = Finding("A13", "Processing state", step="G2", feeds=["G5 log transform", "S1 normalization",
                                                             "G7 missingness"],
                method="your step-9 answers against the A2 scale class, the shared-median signature, the spread "
                       "of sample medians and floor ties (audit set); confirmations from a.confirm_processing_state")
    notes = platform_notes(s)
    rows, questions = [], []
    for bid, g in s.block_map().items():
        assay = g.assay_label or "assay"
        X = audit.audit_matrix(bid).to_numpy(dtype=float)
        cls = audit.results.get("scale", {}).get(bid) or classify(X, audit.params)[0]
        ev = evidence(X, cls, audit.params)
        hist = s.history_answers.get(assay, {})
        for item in ITEMS:
            dec = _declared(hist.get(item))
            sentences, match = assess(item, dec, ev, audit.params)
            conf = audit.confirmed(assay, item)
            pn = [n for n in notes if item in n["items"]]
            rows.append({"block": bid, "assay": assay, "item": item, "declared": dec, "observed": sentences,
                         "match": match, "confirmed": conf, "platform_notes": [n["id"] for n in pn],
                         "effective": conf if conf in ("yes", "no") else (dec if match == "matches" else None)})
            if conf is None and (match == "does_not_match" or (item in ASK_WHEN_UNDECLARED and dec in ("absent", "unsure"))):
                questions.append({"assay": assay, "block": bid, "item": item, "declared": dec, "match": match,
                                  "evidence": sentences, "platform_notes": [n["text"] for n in pn],
                                  "text": f"Was {assay} {item.replace('_', ' ')} before upload?",
                                  "answer_with": f"a.confirm_processing_state('{item}', 'yes' | 'no' | 'unsure', "
                                                 f"assay='{assay}')"})
            if match == "does_not_match":
                note = f" Platform note: {pn[0]['text']}" if pn else ""
                status = "ok" if conf else "note"
                f.say(status, f"{bid} {item.replace('_', ' ')}: you answered '{dec}', but {' '.join(sentences)} "
                              f"Confirmed: {conf or 'not yet'}.{note}")
        f.measures.setdefault("observed", {})[bid] = ev
    f.measures.update({"items": rows, "questions": questions, "platform_notes": notes})
    open_mismatch = sorted({r["item"] for r in rows if r["match"] == "does_not_match" and r["confirmed"] is None})
    f.measures["gate"] = {"blocks": ["G5", "S1"] if open_mismatch else [],
                          "text": "Insufficient metadata: confirm processing state" if open_mismatch else None,
                          "unconfirmed_mismatches": open_mismatch}
    if open_mismatch:
        f.say("note", f"Until you confirm {', '.join(open_mismatch)}, the plan will mark G5 (log transform) and S1 "
                      "(normalization) as 'Insufficient metadata: confirm processing state'. See a.questions().")
    if not f.sentences:
        f.say("ok", "Your answers do not contradict the evidence for any assessable item.")
    return f
