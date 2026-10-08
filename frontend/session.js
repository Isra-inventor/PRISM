// PRISM v3: sessions of one or more datasets, schema import and its review (v3 §3, §4.5).
// The wizard itself lives in app.js; this file adds the session bar, the start paths and the
// "Imported schema" review. Nothing is imported or merged without an explicit click.

(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const KEY = "prism.session";
  const STATUS = { wizard_in_progress: "wizard in progress", imported_awaiting_confirm: "imported, awaiting confirm", confirmed: "confirmed" };
  const MODE = {
    exact: ["Exact", "The data file is the one this schema was made from (same sha256). The schema is accepted as stored; every value-dependent number was recomputed and compared."],
    template: ["Template", "Same column names, different values. Roles, blocks, annotations, exclusions and rules are reused; everything computed from values is recomputed. Processing history, the time unit and answered value-dependent questions are NOT imported: you answer them again."],
    seeded_wizard: ["Seeded wizard", "The columns differ. What matches carries over; the rest is unresolved and the wizard opens with the schema as its starting proposal."],
  };
  let ST = null;

  function el(tag, attrs = {}, ...children) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else if (k === "disabled") n.disabled = !!v;
      else n.setAttribute(k, v);
    }
    for (const c of children.flat(Infinity)) if (c != null && c !== false) n.append(c.nodeType ? c : document.createTextNode(String(c)));
    return n;
  }
  const show = (n, on = true) => n.classList.toggle("hidden", !on);
  async function api(path, body, method = "POST") {
    const opts = { method, headers: {} };
    if (body instanceof FormData) opts.body = body;
    else if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
    const r = await fetch(path, opts);
    const j = await r.json().catch(() => ({}));
    if (!r.ok) {
      const d = j.detail;
      const e = new Error(typeof d === "string" ? d : d?.message || JSON.stringify(d || j));
      e.errors = d?.errors || [];
      throw e;
    }
    return j;
  }
  const store = { get: () => { try { return localStorage.getItem(KEY); } catch (_) { return null; } },
                  set: (v) => { try { v ? localStorage.setItem(KEY, v) : localStorage.removeItem(KEY); } catch (_) {} } };

  let J = { merge: null, runs: [] };        // journey state: merge readiness, audit runs
  async function load(sid) {
    try { ST = await api(`/api/sessions/${sid}`, undefined, "GET"); store.set(sid); }
    catch (_) { ST = null; store.set(null); }
    J = { merge: null, runs: [] };
    if (ST && ST.datasets.some((d) => d.status === "confirmed")) {
      [J.merge, J.runs] = await Promise.all([
        api(`/api/sessions/${sid}/merge-report`, undefined, "GET").catch(() => null),
        api(`/api/sessions/${sid}/audit`, undefined, "GET").catch(() => [])]);
    }
    renderBar();
  }
  async function ensure() {
    if (ST) return ST.session_id;
    ST = await api("/api/sessions", { name: "" });
    store.set(ST.session_id);
    renderBar();
    return ST.session_id;
  }
  async function refresh() { if (ST) await load(ST.session_id); }

  function ready() {
    return ST && ST.datasets.length && ST.datasets.every((d) => d.status === "confirmed");
  }
  const ICON = {
    data: '<path d="M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3Zm0 0v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    merge: '<path d="M6 4v5a5 5 0 0 0 5 5h7M6 20v-5M15 11l3 3-3 3"/>',
    audit: '<path d="M4 19V9M10 19V5M16 19v-7M22 19H2"/>',
    report: '<path d="M7 3h7l5 5v13H7zM14 3v5h5M10 13h6M10 17h6"/>',
  };
  const icon = (k) => { const s2 = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    s2.setAttribute("viewBox", "0 0 24 24"); s2.setAttribute("aria-hidden", "true"); s2.innerHTML = ICON[k]; return s2; };

  function step(n, key, title, state, detail, onclick, extra, id) {
    return el("li", { class: `j-step j-${state}` },
      el("button", { type: "button", class: "j-main", id, disabled: state === "locked", onclick,
        "aria-label": `Step ${n}: ${title} — ${detail}` },
        el("span", { class: "j-icon" }, icon(key), el("i", { text: state === "done" ? "✓" : String(n) })),
        el("span", { class: "j-text" }, el("b", { text: title }), el("span", { text: detail }))),
      extra || null);
  }
  function renderBar() {
    const bar = $("session-bar");
    if (!ST) {
      bar.replaceChildren(el("ol", { class: "journey" },
        step(1, "data", "Data", "current", "Add your first dataset", showStart),
        step(2, "merge", "Merge", "locked", "Line up datasets"),
        step(3, "audit", "Audit", "locked", "Check the data"),
        step(4, "report", "Report", "locked", "Export the summary")));
      return;
    }
    const ds = ST.datasets, conf = ds.filter((d) => d.status === "confirmed");
    const pending = ds.length - conf.length;
    const m = J.merge, runs = J.runs || [];
    const mergeOpen = m ? (m.blocking || []).length : 0;
    const dataState = !ds.length ? "current" : pending ? "current" : "done";
    const mergeState = conf.length < 2 ? (conf.length && !pending ? "skip" : "locked") : mergeOpen ? "current" : "done";
    const auditOk = ready() && (!m || m.ready_for_audit);
    const auditState = !auditOk ? "locked" : runs.length ? "done" : "current";
    const reportState = runs.length ? "current" : "locked";
    const chips = el("div", { class: "j-chips" }, ds.map((d) => el("button", {
        class: `ds-chip st-${d.status}`, type: "button", title: `${d.origin} · ${STATUS[d.status] || d.status}`,
        onclick: () => openDataset(d) }, el("b", { text: d.dataset_id }), ` ${d.name}`)),
      el("button", { class: "ds-chip add", type: "button", text: "+ Add", onclick: showStart }));
    bar.replaceChildren(
      el("div", { class: "sb-head" }, el("span", { class: "item-sub" }, "Session ", el("code", { text: ST.session_id })),
        el("span", { class: "spacer" }),
        el("button", { class: "linkbtn", type: "button", text: "New session", onclick: async () => { ST = null; J = { merge: null, runs: [] }; store.set(null); await ensure(); showStart(); } })),
      el("ol", { class: "journey" },
        step(1, "data", "Data", dataState, !ds.length ? "Add your first dataset" : `${conf.length} confirmed${pending ? ` · ${pending} in progress` : ""}`, showStart, chips),
        step(2, "merge", "Merge", mergeState, conf.length < 2 ? (mergeState === "skip" ? "One dataset: nothing to merge" : "Needs two datasets") :
          mergeOpen ? `${mergeOpen} item(s) to decide` : `${get(m, "cross_dataset.sample_overlap.n_in_all")} shared samples`, () => showMerge(), null, "open-merge"),
        step(3, "audit", "Audit", auditState, !ready() ? "Confirm every dataset first" : !auditOk ? "Finish the merge first" :
          runs.length ? `${runs.length} run(s) · open the latest` : "Run the 11 checks", () => window.PRISM_AUDIT && window.PRISM_AUDIT.open(ST), null, "run-audit"),
        step(4, "report", "Report", reportState, runs.length ? "Preview and export" : "Run the audit first",
          () => window.PRISM_FINAL && window.PRISM_FINAL.open(ST), null, "open-report")));
  }
  const get = (o, path) => path.split(".").reduce((x, k) => (x == null ? x : x[k]), o);

  async function openDataset(d) {
    if (d.status === "imported_awaiting_confirm") return showReview(d.dataset_id);
    if (d.status === "wizard_in_progress" && d.step0_session_id) {
      show($("panel-import"), false);
      try { await window.PRISM_APP.loadStep0(d.step0_session_id); } catch (e) { alert(e.message); }
      return;
    }
    if (d.status === "confirmed" && d.step0_session_id) {
      show($("panel-import"), false);
      try { await window.PRISM_APP.loadStep0(d.step0_session_id); } catch (e) { alert(e.message); }
    }
  }

  function showStart() {
    show($("panel-import"), false);
    window.PRISM_APP?.reset();
    show($("panel-upload"));
  }

  // ---------------------------------------------------------------- start paths
  document.querySelectorAll("#start-tabs .opt-card").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#start-tabs .opt-card").forEach((x) => x.classList.toggle("sel", x === b));
    show($("import-form"), b.dataset.path === "import");
    show($("wizard-form"), b.dataset.path === "wizard");
  }));
  $("imp-go").addEventListener("click", async () => {
    const data = $("imp-data").files[0], schema = $("imp-schema").files[0], meta = $("imp-meta").files[0];
    show($("imp-error"), false);
    if (!data || !schema) { $("imp-error").textContent = "Choose the data file and its schema.json."; show($("imp-error")); return; }
    const sid = await ensure();
    const fd = new FormData();
    fd.append("data", data); fd.append("schema", schema);
    if (meta) fd.append("metadata", meta);
    $("imp-go").disabled = true;
    try {
      const rep = await api(`/api/sessions/${sid}/datasets`, fd);
      await load(sid);
      if (rep.mode === "seeded_wizard") { show($("panel-upload"), false); await window.PRISM_APP.loadStep0(rep.step0_session_id); }
      else await showReview(rep.dataset_id);
    } catch (e) {
      $("imp-error").replaceChildren(el("div", { text: e.message }),
        e.errors?.length ? el("ul", {}, e.errors.slice(0, 20).map((x) => el("li", {}, el("code", { text: x.path }), " ", x.message))) : null);
      show($("imp-error"));
    } finally { $("imp-go").disabled = false; }
  });

  // ---------------------------------------------------------------- import review
  const fmt = (v) => typeof v === "number" ? String(+v.toPrecision(6)) : typeof v === "string" ? v : JSON.stringify(v);
  async function showReview(did) {
    const box = $("panel-import");
    let rep;
    try { rep = await api(`/api/sessions/${ST.session_id}/datasets/${did}/import-report`, undefined, "GET"); }
    catch (e) { alert(e.message); return; }
    show($("panel-upload"), false); show($("workspace"), false); show(box);
    const [mt, md] = MODE[rep.mode] || [rep.mode, ""];
    const openQ = (rep.questions || []).filter((q) => q.status === "open");
    const act = async (path, label) => {
      try { const r = await api(`/api/sessions/${ST.session_id}/datasets/${did}/${path}`, {}); ST = r.session; renderBar(); return r; }
      catch (e) { alert(e.message); return null; }
    };
    const s = rep.summary || {};
    box.replaceChildren(
      el("div", { class: "panel-head" }, el("h2", { text: `Imported schema · ${rep.dataset?.name || did}` }),
        el("span", { class: `pill mode-${rep.mode}`, text: `Mode: ${mt}` })),
      el("div", { class: "panel-body" },
        el("p", { class: "q", text: md }),
        rep.edited_or_corrupt ? el("div", { class: "alert alert-error", text: `The data file is identical, but ${rep.n_differences} number(s) stored in the schema differ from what PRISM recomputes: the schema file was edited or is corrupt. Check the list below before accepting.` }) : null,
        el("div", { class: "section-label", text: "Checks (re-run from the data)" }),
        el("ul", { class: "np" }, (rep.checks || []).map((c) => el("li", { class: c.ok ? "" : "warn-text" }, c.ok ? "✓ " : "✗ ", c.check, c.detail ? ` — ${c.detail}` : ""))),
        el("div", { class: "section-label", text: `Recomputed vs stored (${rep.n_differences || 0} difference${rep.n_differences === 1 ? "" : "s"})` }),
        rep.differences?.length ? el("table", { class: "ledger" }, el("thead", {}, el("tr", {}, ["path", "stored", "recomputed"].map((h) => el("th", { text: h })))),
          el("tbody", {}, rep.differences.slice(0, 200).map((x) => el("tr", {}, el("td", {}, el("code", { text: x.path })), el("td", { text: fmt(x.stored) }), el("td", { text: fmt(x.recomputed) })))))
          : el("p", { class: "item-sub", text: "None: everything computed from the values matches the schema." }),
        (rep.differences?.length || rep.binding?.main?.missing_in_file?.length || rep.binding?.main?.extra_in_file?.length) ?
          el("div", { id: "explain-box" }, el("button", { class: "linkbtn", type: "button", text: "Explain differences", onclick: async () => {
            try {
              const r = await api(`/api/sessions/${ST.session_id}/datasets/${did}/explain-differences`, undefined, "GET");
              $("explain-box").replaceChildren(el("div", { class: "section-label", text: "What the differences mean" }),
                el("ul", { class: "np" }, r.explanations.map((x) => el("li", { text: x.text }))),
                el("p", { class: "item-sub", text: "Written by fixed rules from the differences above (no AI in import)." }));
            } catch (e) { alert(e.message); } } })) : null,
        (rep.warnings || []).length ? el("ul", { class: "warns" }, rep.warnings.map((w) => el("li", { text: w }))) : null,
        el("div", { class: "section-label", text: "Summary" }),
        el("ul", { class: "review-list" },
          el("li", {}, el("span", { text: "Layout" }), el("span", { text: (s.layout || "").replace(/_/g, " ") })),
          ...(s.assays || []).map((a) => el("li", {}, el("span", { text: `Assay ${a.assay_id}` }), el("span", { text: `${a.assay_label} · ${a.n_blocks} block(s), ${a.n_value_columns} value columns` }))),
          el("li", {}, el("span", { text: "Annotation columns" }), el("span", { text: `${s.annotations_kept} kept, ${s.annotations_excluded} excluded` })),
          el("li", {}, el("span", { text: "Excluded columns" }), el("span", { text: String(s.excluded_columns ?? 0) })),
          el("li", {}, el("span", { text: "Design" }), el("span", { text: s.design || "—" })),
          el("li", {}, el("span", { text: "Samples" }), el("span", { text: String(s.n_samples ?? "—") })),
          el("li", {}, el("span", { text: "Processing history" }), el("span", { text: Object.entries(s.processing_history || {}).map(([k, v]) => `${k.replace(/_/g, " ")}: ${v ?? "unanswered"}`).join(" · ") }))),
        openQ.length ? el("div", {}, el("div", { class: "section-label", text: `${openQ.length} open question(s): answer or dismiss before accepting` }),
          openQ.map((q) => el("div", { class: "question" }, el("b", { text: q.text }), el("div", { class: "question-opts" },
            (q.options || []).map((o) => el("button", { class: "btn btn-sm", type: "button", text: o.label, onclick: async () => {
              await api("/api/question/answer", { session_id: rep.step0_session_id, question_id: q.question_id, option_ids: [o.option_id] }).catch((e) => alert(e.message));
              showReview(did); } })),
            el("button", { class: "linkbtn", type: "button", text: "Dismiss", onclick: async () => {
              await api("/api/question/dismiss", { session_id: rep.step0_session_id, question_id: q.question_id }).catch((e) => alert(e.message));
              showReview(did); } }))))) : null,
        el("div", { class: "guide-actions" },
          el("button", { class: "btn btn-light btn-sm", type: "button", text: "Accept", disabled: rep.dataset?.status !== "imported_awaiting_confirm",
            title: "A logged human confirmation: the output folder is regenerated through the wizard's own finalize",
            onclick: async () => { if (await act("accept-import")) { show(box, false); showStart(); } } }),
          el("button", { class: "btn btn-sm", type: "button", text: "Open in the wizard", onclick: async () => {
            const r = await act("open-wizard"); if (r) { show(box, false); await window.PRISM_APP.loadStep0(r.result); } } }),
          el("button", { class: "btn btn-sm", type: "button", text: "Reject", onclick: async () => {
            if (!confirm("Reject this import? Nothing of it is kept.")) return;
            if (await act("reject")) { show(box, false); showStart(); } } }))));
  }


  // ---------------------------------------------------------------- merge (v3 §5)
  const MCOLS = { single: "one dataset", merged: "merged", taken: "taken from one dataset", kept_per_dataset: "kept per dataset",
                  dropped: "dropped", conflict_open: "conflict: both kept until resolved" };
  async function showMerge() {
    const box = $("panel-import");
    const base = `/api/sessions/${ST.session_id}`;
    let doc;
    try { doc = await api(`${base}/merge-report`, undefined, "GET"); } catch (e) { alert(e.message); return; }
    show($("panel-upload"), false); show($("workspace"), false); show(box);
    const x = doc.cross_dataset;
    const decide = async (path, body) => { try { await api(`${base}/merge/${path}`, body); } catch (e) { alert(e.message); } await refresh(); showMerge(); };
    const csvIn = el("input", { type: "file", accept: ".csv,text/csv" });
    const sug = x.id_mapping.suggestions || [];
    box.replaceChildren(
      el("div", { class: "panel-head" }, el("h2", { text: "Merge the datasets" }),
        el("span", { class: `pill ${doc.ready_for_audit ? "mode-exact" : "mode-template"}`, text: doc.ready_for_audit ? "Ready for audit" : "Not ready yet" })),
      el("div", { class: "panel-body" },
        el("p", { class: "q", text: "Sample IDs are compared exactly; near misses are only suggested. Nothing is mapped or merged until you decide, and the IDs in the datasets are never rewritten." }),
        doc.blocking.length ? el("ul", { class: "warns" }, doc.blocking.map((b) => el("li", { text: b }))) : null,
        el("div", { class: "section-label", text: "Sample overlap" }),
        el("table", { class: "ledger" }, el("thead", {}, el("tr", {}, ["pair", "shared", "only in first", "only in second", "Jaccard"].map((h) => el("th", { text: h })))),
          el("tbody", {}, x.sample_overlap.pairs.map((p) => el("tr", {}, el("td", { text: `${p.a} · ${p.b}` }), el("td", { text: p.n_shared }),
            el("td", { text: p.n_only_a, title: p.only_a.join(", ") }), el("td", { text: p.n_only_b, title: p.only_b.join(", ") }), el("td", { text: p.jaccard ?? "—" }))))),
        el("p", { class: "item-sub", text: `${x.sample_overlap.n_unified} unified sample(s), ${x.sample_overlap.n_in_all} in every dataset.` }),
        el("div", { class: "section-label", text: `ID suggestions (${sug.length})` }),
        sug.length ? sug.map((g) => el("div", { class: `question${g.status === "open" ? "" : " done"}` },
          el("b", { text: `${g.a.dataset_id} '${g.a.sample_id}'  ≈  ${g.b.dataset_id} '${g.b.sample_id}'` }), ` (differ by: ${g.rules.join(", ")}) · ${g.status.replace(/_/g, " ")}`,
          el("div", { class: "question-opts" },
            g.status === "open" ? [el("button", { class: "btn btn-sm", type: "button", text: `Same sample (unified ID '${g.a.sample_id}')`, onclick: () => decide("confirm-mapping", { item_id: g.suggestion_id, decision: "confirm" }) }),
              el("button", { class: "linkbtn", type: "button", text: "Dismiss", onclick: () => decide("confirm-mapping", { item_id: g.suggestion_id, decision: "dismiss" }) })]
              : el("button", { class: "linkbtn", type: "button", text: "Reopen", onclick: () => decide("confirm-mapping", { item_id: g.suggestion_id, decision: "reopen" }) }))))
          : el("p", { class: "item-sub", text: "None." }),
        el("div", { class: "section-label", text: "Mapping file (dataset_id, sample_id, unified_id)" }),
        el("div", { class: "question-opts" }, csvIn,
          el("button", { class: "btn btn-sm", type: "button", text: "Upload mapping", onclick: async () => {
            if (!csvIn.files[0]) return; const fd = new FormData(); fd.append("file", csvIn.files[0]); await decide("mapping-csv", fd); } }),
          x.id_mapping.mapping_csv ? el("button", { class: "linkbtn", type: "button", text: `Remove ${x.id_mapping.mapping_csv.name}`, onclick: async () => {
            try { await api(`${base}/merge/mapping-csv`, undefined, "DELETE"); } catch (e) { alert(e.message); } showMerge(); } }) : null),
        x.id_mapping.entries.length ? el("p", { class: "item-sub", text: `${x.id_mapping.entries.length} sample(s) mapped: ` + x.id_mapping.entries.slice(0, 10).map((e) => `${e.dataset_id} ${e.sample_id} → ${e.unified_id}`).join("; ") }) : null,
        el("div", { class: "section-label", text: `Value conflicts (${x.conflicts.length})` }),
        x.conflicts.length ? x.conflicts.map((c) => el("div", { class: `question${c.status === "open" ? "" : " done"}` },
          el("b", { text: `'${c.column}' (${c.audit_kind}): ${c.n_disagree} shared sample(s) disagree` }), c.resolution ? ` · ${c.resolution}` : "",
          el("div", { class: "item-sub", text: c.disagreements.slice(0, 5).map((d) => `${d.unified_id}: ` + Object.entries(d.values).map(([k, v]) => `${k}=${v}`).join(" / ")).join("  ·  ") }),
          el("div", { class: "question-opts" }, c.options.map((o) => el("button", { class: `btn btn-sm${c.resolution === o ? " sel" : ""}`, type: "button", text: o.replace("take:", "take ").replace("_", " "),
            onclick: () => decide("resolve-conflict", { item_id: c.conflict_id, decision: o }) })))))
          : el("p", { class: "item-sub", text: "None." }),
        el("div", { class: "section-label", text: `Questions (${x.questions.length})` }),
        x.questions.length ? x.questions.map((q) => el("div", { class: `question${q.status === "open" ? "" : " done"}` },
          el("b", { text: q.text }), q.status !== "open" ? ` · ${q.answer || q.status}` : "",
          el("div", { class: "question-opts" }, q.options.map((o) => el("button", { class: "btn btn-sm", type: "button", text: o.label,
            onclick: () => decide("resolve-conflict", { item_id: q.question_id, decision: o.option_id }) })),
            el("button", { class: "linkbtn", type: "button", text: q.status === "open" ? "Dismiss" : "Reopen",
              onclick: () => decide("resolve-conflict", { item_id: q.question_id, decision: q.status === "open" ? "dismiss" : "reopen" }) }))))
          : el("p", { class: "item-sub", text: "None." }),
        el("div", { class: "section-label", text: "Design agreement" }),
        el("ul", { class: "np" }, x.design_agreement.map((d) => el("li", {}, `${d.a} · ${d.b}: subject `,
          d.subject ? (d.subject.agree ? "agrees" : `${d.subject.n_disagree} disagree`) : "not compared",
          ", time ", d.time ? (d.time.agree ? "agrees" : `${d.time.n_disagree} disagree`) : "not compared",
          d.time_unit ? ` (units ${d.time_unit[d.a] || "?"} / ${d.time_unit[d.b] || "?"})` : ""))),
        x.column_suggestions.length ? el("div", {}, el("div", { class: "section-label", text: "Columns that look the same (suggestion only, not merged)" }),
          el("ul", { class: "np" }, x.column_suggestions.map((c) => el("li", { text: `${c.a.dataset_id} '${c.a.column}' = ${c.b.dataset_id} '${c.b.column}' on ${c.n_shared} shared samples` })))) : null,
        el("div", { class: "section-label", text: "Unified sample table" }),
        el("ul", { class: "np" }, x.columns.map((c) => el("li", { text: `${c.column}: ${c.take ? `taken from ${c.take}` : MCOLS[c.status] || c.status} (${[].concat(c.datasets).join(", ")})` }))),
        el("div", { class: "guide-actions" },
          el("button", { class: "btn btn-light btn-sm", type: "button", text: "Close", onclick: () => { show(box, false); showStart(); } }))));
  }

  window.PRISM_SESSION = { ensure, refresh, showStart, showMerge, showReview, current: () => ST };
  const saved = store.get();
  if (saved) load(saved); else renderBar();
})();
