"""Session facts for the rule base (Build Request v4, section 8): the findings of the latest audit
run at each dataset's current state, the intake and the session. A fact that could not be
computed is None ("indeterminate"); a rule condition on an indeterminate fact never holds, so an
indeterminate input can never yield Recommended."""

from __future__ import annotations

from ..audit import engine, overrides as ov
from ..executor import state as state_mod
from ..util import read_json
from . import intake as intake_mod


def _q_ok(r, alpha=0.05):
    return r.get("q") is not None and r["q"] <= alpha


def latest_run(st, did, state_id):
    rid = engine.latest_run_for_state(st, state_id)
    if rid is None:
        return None
    return rid


def ensure_run(st, did, state_id, who="plan"):
    rid = latest_run(st, did, state_id)
    if rid is None:
        man = engine.run(st, datasets=[did], state=state_id, who=who) if state_id != state_mod.root_id(st, did) \
            else engine.run(st, datasets=[did], who=who)
        rid = man["run_id"]
    return rid


def findings_of(st, rid, did):
    man = read_json(engine.runs_dir(st) / rid / "manifest.json")
    out = {}
    for i in man["findings"]:
        if i["dataset"] != did:
            continue
        out.setdefault(i["assay"], {})[i["audit_id"]] = engine.finding(st, rid, i["file"])
    return man, out


def unit_facts(st, did, assay, F, P, intake, state_meta, items):
    """Facts for one dataset unit. F: {audit_id: finding}."""
    def M(aid):
        f = F.get(aid)
        return (f or {}).get("measures") or {}

    f = {"dataset": did, "assay": assay, "state_id": state_meta["state_id"], "checkpoint": state_meta.get("checkpoint"),
         "state_scale": state_meta.get("scale")}
    # G2 / A13
    a13 = M("A13")
    items13 = {r["item"]: r for r in a13.get("items", [])}
    f["processing_state.log_transformed"] = (items13.get("log_transformed") or {}).get("effective")
    f["processing_state.normalized"] = (items13.get("normalized") or {}).get("effective")
    f["processing_state.unconfirmed"] = a13.get("unconfirmed_mismatches", [])
    f["processing_state.any_unconfirmed"] = bool(a13.get("unconfirmed_mismatches"))
    f["processing_state.questions"] = a13.get("questions", [])
    f["scale.class"] = M("A2").get("classification")
    f["scale.log_like"] = (state_meta.get("scale") in ("log2", "arcsinh")) or f["processing_state.log_transformed"] == "yes"
    f["scale.out_of_scope"] = f["scale.class"] in ("count_like", "proportion_like", "compositional_like")
    msx = M("A3").get("mean_sd_x") or {}
    ci = msx.get("ci95")
    f["a3.slope"] = msx.get("slope")
    f["a3.slope_ci"] = ci
    f["a3.slope_ci_excludes_zero"] = (ci[0] > 0 or ci[1] < 0) if ci else None
    f["a3.rho_y"] = (M("A3").get("mean_sd_y") or {}).get("spearman_rho")
    # G3
    g3 = M("A1").get("g3_candidates") or {}
    f["g3.classes"] = g3.get("classes", [])
    f["g3.n_candidates"] = len(f["g3.classes"])
    # zeros / floors / missingness (A4, A12)
    a12 = M("A12")
    enc = a12.get("cell_encoding") or {}
    f["zeros.present"] = bool(enc.get("zero")) if enc else None
    f["zeros.answer"] = a12.get("zeros_answer")
    f["zeros.question"] = bool(a12.get("questions"))
    a4 = M("A4")
    ft = a4.get("floor_ties") or {}
    f["floor.present"] = (ft.get("share", 0) >= P["floor_tie_share_indicator"]) if ft else None
    f["floor.answer"] = next((o.get("value") for o in reversed(items) if o.get("kind") == "floor_censored"
                              and o.get("dataset") in (None, "", did)), None)
    pat = a12.get("pattern") or {}
    f["missing.any"] = (enc.get("nan", 0) > 0) if enc else None
    f["missing.pattern_class"] = pat.get("pattern_class")
    ind = pat.get("indicators") or {}
    f["missing.batch_association"] = ind.get("batch_association")
    f["missing.intensity"] = ind.get("intensity_dependence")
    f["missing.share_complete_features"] = ((a12.get("rates") or {}).get("pattern") or {}).get("share_no_nan")
    bwa = a12.get("batch_wide_absence") or {}
    f["missing.batch_wide_absent_any"] = (bwa.get("n_features_any", 0) > 0) if "n_features_any" in bwa else None
    f["missing.imputation_candidates"] = a12.get("imputation_candidates")
    # QC and run order (A6)
    a6 = M("A6")
    roles = (a6.get("roles") or {}).get("counts") or {}
    f["qc.n"] = roles.get("qc", 0) + roles.get("pool", 0)
    ro = a6.get("run_order")
    f["run_order.present"] = ro is not None
    f["run_order.drift"] = (abs(ro.get("spearman_median_y", 0) or 0) >= 0.3) if ro else None
    qtr = (ro or {}).get("qc_feature_trend")
    f["qc.drift"] = (qtr["share_abs_rho_gt_0_5"] >= P["qc_drift_share"]) if qtr else None
    # batch structure (A7)
    a7 = M("A7")
    variables = (a7.get("variables") or {})
    bvars = [v for v in variables.get("B", [])]
    f["batch.variables"] = [v["name"] for v in bvars if v["kind"] == "categorical"]
    f["batch.run_order_variables"] = [v["name"] for v in bvars if v["kind"] == "numeric"]
    f["batch.present"] = bool(f["batch.variables"])
    f["batch.primary"] = f["batch.variables"][0] if f["batch.variables"] else None
    spb = a7.get("samples_per_batch") or []
    f["batch.small"] = any(x.get("below_suggestion") for x in spb) if spb else None
    f["batch.min_size"] = min([x["min"] for x in spb if x.get("min") is not None], default=None)
    assoc = (a7.get("associations") or {}).get("rows") or []
    if assoc:
        f["batch.association"] = any(_q_ok(r) for r in assoc if r.get("role") == "batch")
    else:
        f["batch.association"] = None
    rc = a7.get("relationship_correlation") or {}
    prim = next((b for b in rc.get("by_batch", []) if b.get("variable") == f["batch.primary"]), None)
    eff = (prim or {}).get("batch_effect_on_correlation")
    f["batch.corr_effect"] = eff
    f["batch.same_r_above_diff"] = (eff > 0) if eff is not None else None
    contrast = intake.get("contrast_variable")
    struct = a7.get("structure") or []
    f["confound.contrast"] = (any(s.get("code") == "confounded" and s.get("design") == contrast for s in struct)
                              if contrast and contrast != "none" and struct else (False if contrast == "none" else None))
    nested = [s for s in struct if s.get("code") == "nested" and s.get("design") == "subject"
              and "'subject' is constant within" in (s.get("nesting") or "")]
    f["nested.subject_in_batch"] = bool(nested)
    f["nested.text"] = [s["nesting"] for s in nested]
    pairs = a7.get("structure_batch_pairs") or []
    f["batch.pairs_confounded"] = [[s["batch"], s["design"]] for s in pairs if s.get("code") == "confounded"]
    f["batch.two_confounded"] = bool(f["batch.pairs_confounded"])
    # medians (A3, S1 decision node)
    ms = M("A3").get("median_shift") or {}
    f["median.differ"] = ms.get("medians_differ")
    rows = (ms.get("associations") or {}).get("rows") or []
    f["median.technical"] = any(_q_ok(r) for r in rows if r.get("technical")) if rows else None
    if contrast and contrast != "none":
        cr = [r for r in rows if r.get("variable") == contrast]
        f["median.contrast"] = any(_q_ok(r) for r in cr) if cr else None
    else:
        f["median.contrast"] = False if contrast == "none" else None
    # outliers (A11), design (A5, A8)
    f["outliers.flagged"] = (M("A11").get("samples") or {}).get("flagged", [])
    a5 = M("A5")
    de = a5.get("design_effect") or {}
    f["a5.n"], f["a5.n_eff"] = a5.get("n"), de.get("n_eff")
    f["a8.subject_column"] = (M("A8").get("design") or {}).get("subject_column")
    f["a8.constant_within_subject"] = sorted(k for k, v in (M("A8").get("constant_within_subject") or {}).items() if v)
    f["a8.varies_within_subject"] = sorted(k for k, v in (M("A8").get("constant_within_subject") or {}).items()
                                           if v is False)
    f["a8.nested"] = M("A8").get("nested") or []
    f["a5.smallest_cell"] = a5.get("smallest_cell")
    # intake
    for k in ("contrast_variable", "protected_variables", "nuisance_variables", "question_type", "downstream_family"):
        f[f"intake.{k}"] = intake.get(k)
    return f


def collect(st, run_audit=True, who="plan"):
    """-> {"units": [facts...], "runs": {did: run_id}, "intake": {...}}."""
    items = ov.load(st)
    from ..audit import params as params_mod
    P, _ = params_mod.resolve({}, items)
    out = {"units": [], "runs": {}, "states": {}}
    for d in st.confirmed():
        did = d["dataset_id"]
        head = state_mod.head(st, did)
        meta = state_mod.meta(st, did, head)
        rid = ensure_run(st, did, head, who) if run_audit else latest_run(st, did, head)
        out["runs"][did], out["states"][did] = rid, head
        if rid is None:
            continue
        man, by_unit = findings_of(st, rid, did)
        # audits outside this checkpoint's battery (A1, A2, A13, ...) come from the nearest ancestor
        # state that has them (the root is audited with the full C0 battery when needed)
        chain = [head]
        m_ = meta
        while m_.get("parent"):
            chain.append(m_["parent"])
            m_ = state_mod.meta(st, did, m_["parent"])
        carried = {}
        for sid in chain[1:]:
            r_ = latest_run(st, did, sid)
            if r_ is None and sid == chain[-1] and run_audit:
                r_ = ensure_run(st, did, sid, who)
            if r_ is None:
                continue
            _, older = findings_of(st, r_, did)
            for assay, F in older.items():
                for aid, fnd in F.items():
                    if aid not in by_unit.get(assay, {}):
                        by_unit.setdefault(assay, {})[aid] = fnd
                        carried.setdefault(f"{assay}:{aid}", sid)
        out.setdefault("carried_from_ancestor", {})[did] = carried
        bnames = []
        for F in by_unit.values():
            bnames += [v["name"] for v in ((F.get("A7") or {}).get("measures", {}).get("variables") or {}).get("B", [])]
        intake = intake_mod.resolved(st, sorted(set(bnames)))
        out["intake"] = intake
        for assay, F in sorted(by_unit.items()):
            if assay == "session":
                continue
            out["units"].append(unit_facts(st, did, assay, F, P, intake, meta, items))
    out.setdefault("intake", intake_mod.resolved(st))
    out["params"] = P
    return out
