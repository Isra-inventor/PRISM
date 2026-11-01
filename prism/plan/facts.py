"""Facts for the rule base (Build Request v4, section 8): one flat dictionary per block, built from
the findings of the audit at the current state (with A1, A2, A13 ... taken from the nearest ancestor
state that has them, A13 always from S0), the intake and the confirmed schema.

The keys are the ones rules_v0.yaml uses (ported from the v4 build). A fact that could not be
computed is None ("indeterminate"); a rule condition on an indeterminate fact never holds, so an
indeterminate input can never yield Recommended.
"""

from __future__ import annotations


def _q_ok(r, alpha=0.05):
    return r.get("q") is not None and r["q"] <= alpha


def _m(findings, aid, block):
    f = findings.get(aid)
    return (f.measures.get(block) or {}) if f is not None else {}


def block_facts(study, block: str, findings: dict, state_meta: dict, checkpoint: str, intake: dict,
                params: dict) -> dict:
    s = study.schema
    g = s.block_map()[block]
    F = {"dataset": s.table.path.name, "assay": block, "assay_label": g.assay_label, "block": block,
         "state_id": state_meta["state_id"], "checkpoint": checkpoint, "state_scale": state_meta.get("scale")}

    # G2 / A13 (always the S0 finding) and A2
    a13 = findings.get("A13")
    rows = [r for r in (a13.measures.get("items", []) if a13 else []) if r["block"] == block]
    items13 = {r["item"]: r for r in rows}
    F["processing_state.log_transformed"] = (items13.get("log_transformed") or {}).get("effective")
    F["processing_state.normalized"] = (items13.get("normalized") or {}).get("effective")
    unconf = sorted({r["item"] for r in rows if r["match"] == "does_not_match" and r["confirmed"] is None})
    F["processing_state.unconfirmed"] = unconf
    F["processing_state.any_unconfirmed"] = bool(unconf)
    F["processing_state.questions"] = [q for q in (a13.measures.get("questions", []) if a13 else [])
                                       if q["block"] == block]
    F["scale.class"] = _m(findings, "A2", block).get("class")
    F["scale.log_like"] = (state_meta.get("scale") in ("log2", "arcsinh")) or F["processing_state.log_transformed"] == "yes"
    F["scale.out_of_scope"] = F["scale.class"] in ("count_like", "proportion_like", "compositional_like")

    # A3
    a3 = _m(findings, "A3", block)
    msx = a3.get("mean_sd") or {}
    ci = msx.get("ci95")
    F["a3.slope"], F["a3.slope_ci"] = msx.get("slope"), ci
    F["a3.slope_ci_excludes_zero"] = (ci[0] > 0 or ci[1] < 0) if ci else None
    F["a3.rho_y"] = (a3.get("mean_sd_diagnostic") or {}).get("spearman_rho")

    # G3
    g3 = (study.g3.get(state_meta["state_id"]) or {}).get(block) or {}
    F["g3.classes"] = g3.get("classes", [])
    F["g3.n_candidates"] = len(F["g3.classes"])

    # zeros, floors, missingness (A4, A12)
    a12, a4 = _m(findings, "A12", block), _m(findings, "A4", block)
    enc = a12.get("cell_encoding") or {}
    F["zeros.present"] = bool(enc.get("zero")) if enc else None
    zm = g.zero_means
    F["zeros.answer"] = {"not_detected": "yes", "zero": "no"}.get(zm)      # step 7: do zeros mean missing?
    F["zeros.question"] = bool(F["zeros.present"]) and zm is None
    if state_meta.get("parent") and any(o["op"] == "recode_zero_as_missing" for o in state_meta.get("applied_ops", [])):
        F["zeros.answer"] = "done"
    ft = a4.get("floor_ties") or {}
    F["floor.present"] = (ft.get("share", 0) >= params["floor_tie_share_indicator"]) if ft else None
    F["floor.answer"] = study.answer("floor_censored")
    pat = a12.get("pattern") or {}
    ind = pat.get("indicators") or {}
    F["missing.any"] = (enc.get("missing", 0) > 0) if enc else None
    F["missing.pattern_class"] = pat.get("pattern_class")
    F["missing.batch_association"] = ind.get("batch_association")
    F["missing.intensity"] = ind.get("intensity_dependence")
    F["missing.share_complete_features"] = (a12.get("protein_missing_shares") or {}).get("no_missing")
    pc = (a12.get("protein_classes") or {}).get("counts")
    F["missing.batch_wide_absent_any"] = (pc.get("batch_wide_absent", 0) > 0) if pc else None
    F["missing.imputation_candidates"] = a12.get("imputation_candidates")

    # QC and run order (A6)
    a6 = _m(findings, "A6", block)
    types = a6.get("sample_types") or {}
    F["qc.n"] = types.get("qc", 0) + types.get("pool", 0)
    ro = a6.get("run_order")
    F["run_order.present"] = bool(ro) and ro.get("n_used", 0) > 0
    F["run_order.drift"] = (abs(ro.get("spearman_median") or 0) >= params["run_order_rho_indicator"]) \
        if ro and "spearman_median" in ro else None
    qtr = (ro or {}).get("qc_protein_trend")
    F["qc.drift"] = (qtr["share_abs_rho_gt_0_5"] >= params["qc_drift_share"]) \
        if qtr and qtr.get("share_abs_rho_gt_0_5") is not None else None

    # batch structure (A7)
    a7 = _m(findings, "A7", block)
    bvars = (a7.get("variables") or {}).get("B", [])
    F["batch.variables"] = [v["name"] for v in bvars if v["kind"] == "categorical"]
    F["batch.run_order_variables"] = [v["name"] for v in bvars if v["kind"] == "numeric"]
    F["batch.present"] = bool(F["batch.variables"])
    F["batch.primary"] = F["batch.variables"][0] if F["batch.variables"] else None
    spb = a7.get("samples_per_batch") or []
    F["batch.small"] = any(x.get("below_suggestion") for x in spb) if spb else None
    F["batch.min_size"] = min([x["min"] for x in spb if x.get("min") is not None], default=None)
    assoc = (a7.get("associations") or {}).get("rows") or []
    F["batch.association"] = any(_q_ok(r) for r in assoc if r.get("role") == "batch") if assoc else None
    rc = a7.get("relationship_correlation") or {}
    prim = next((b for b in rc.get("by_batch", []) if b.get("variable") == F["batch.primary"]), None)
    eff = (prim or {}).get("batch_effect_on_correlation")
    F["batch.corr_effect"] = eff
    F["batch.same_r_above_diff"] = (eff > 0) if eff is not None else None
    contrast = intake.get("contrast_variable")
    struct = a7.get("structure") or []
    if contrast and contrast != "none":
        F["confound.contrast"] = any(x.get("code") == "confounded" and x.get("design") == contrast
                                     for x in struct) if struct else None
    else:
        F["confound.contrast"] = False if contrast == "none" else None
    nested = [x for x in struct if x.get("code") == "nested" and x.get("design") == "subject"
              and "'subject' is constant within" in (x.get("nesting") or "")]
    F["nested.subject_in_batch"] = bool(nested)
    F["nested.text"] = [x["nesting"] for x in nested]
    pairs = a7.get("structure_batch_pairs") or []
    F["batch.pairs_confounded"] = [[x["batch"], x["design"]] for x in pairs if x.get("code") == "confounded"]
    F["batch.two_confounded"] = bool(F["batch.pairs_confounded"])

    # sample medians (A3, the S1 decision node median_shift_technical)
    ms = a3.get("median_shift") or {}
    F["median.differ"] = ms.get("medians_differ")
    rows_ms = ms.get("associations") or []
    F["median.technical"] = any(_q_ok(r) for r in rows_ms if r.get("technical")) if rows_ms else None
    if contrast and contrast != "none":
        cr = [r for r in rows_ms if r.get("variable") == contrast]
        F["median.contrast"] = any(_q_ok(r) for r in cr) if cr else None
    else:
        F["median.contrast"] = False if contrast == "none" else None

    # outliers (A11), design (A5, A8)
    F["outliers.flagged"] = (_m(findings, "A11", block).get("samples") or {}).get("flagged", [])
    a5, a8 = _m(findings, "A5", block), _m(findings, "A8", block)
    de = a5.get("design_effect") or {}
    F["a5.n"], F["a5.n_eff"] = a5.get("n"), de.get("n_eff")
    F["a5.smallest_cell"] = a5.get("smallest_cell")
    F["a8.subject_column"] = "subject" if a8.get("design") else None
    cws = a8.get("constant_within_subject") or {}
    F["a8.constant_within_subject"] = sorted(k for k, v in cws.items() if v)
    F["a8.varies_within_subject"] = sorted(k for k, v in cws.items() if v is False)
    F["a8.nested"] = a8.get("nested") or []

    # questions you can answer
    qs = [dict(q, kind="confirm_processing_state",
               answer_with=q.get("answer_with")) for q in F["processing_state.questions"]]
    if F["floor.present"] and F["floor.answer"] is None:
        qs.append({"kind": "floor_censored", "block": block,
                   "text": "Should floor-tied values be treated as censored (below a detection limit) in diagnostics?",
                   "reason": f"{ft.get('n_proteins')} of {ft.get('n_eligible')} proteins have floor ties",
                   "consequence": "changes the diagnostic policy only; the data are never changed",
                   "answer_with": "study.answer_question('floor_censored', 'yes' | 'no')"})
    F["questions"] = qs
    for k in ("contrast_variable", "protected_variables", "nuisance_variables", "question_type", "downstream_family"):
        F[f"intake.{k}"] = intake.get(k)
    return F
