// PRISM v3: the Tier 1 audit panel (v3 §7). Runs the audit through the API (the same engine as
// `python -m prism audit run`), renders a run with audit_report.js, edits parameters and
// overrides for the next run, and compares a run with the previous one. No AI here.

(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const R = () => window.PRISM_REPORT;
  let SID = null, CUR = null, DEFAULTS = null, CTRL = null;
  const CHAT = [];        // this page's conversation with the operator (not stored)

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
  async function api(path, body, method) {
    const opts = { method: method || (body === undefined ? "GET" : "POST"), headers: {} };
    if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
    let r;
    try { r = await fetch(path, opts); } catch (e) { throw new Error(`Could not reach the server (${e.message}). Is it still running?`); }
    const text = await r.text();
    let j = null;
    try { j = text ? JSON.parse(text) : {}; } catch (_) { j = null; }
    if (!r.ok) {
      if (j && typeof j.detail === "string") throw new Error(j.detail);
      if (j && j.detail) throw new Error(JSON.stringify(j.detail));
      throw new Error(`The server answered HTTP ${r.status}${text ? `: ${text.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim().slice(0, 300)}` : ""}` +
        (r.status === 502 || r.status === 504 ? " (a proxy timeout: the job may still be running)" : ""));
    }
    if (j === null) throw new Error(`The server answered with something that is not JSON: ${text.slice(0, 200)}`);
    return j;
  }
  // a long job: start it, then poll; onProgress(text, seconds) shows what is running
  async function job(path, body, onProgress) {
    let v = await api(path, body);
    while (v.status === "running") {
      if (onProgress) onProgress(v.progress, v.elapsed_s);
      await new Promise((res) => setTimeout(res, 1000));
      v = await api(`/api/jobs/${v.job_id}`);
    }
    if (v.status === "error") throw new Error(v.error || "The job failed.");
    return v.result;
  }
  const box = () => $("panel-audit");
  function hideOthers() {
    for (const id of ["panel-upload", "panel-import", "workspace"]) { const n = $(id); if (n) n.classList.add("hidden"); }
    box().classList.remove("hidden");
  }

  async function open(st) {
    SID = st.session_id;
    hideOthers();
    if (!DEFAULTS) DEFAULTS = await api("/api/audit/params").catch(() => ({ defaults: {}, heuristic: [] }));
    const runs = await api(`/api/sessions/${SID}/audit`).catch(() => []);
    if (runs.length) await show(runs[runs.length - 1].run_id);
    else frame(el("p", { class: "q", text: "No audit run yet. The audit reads only the confirmed output folders and overrides.json; it changes nothing." }));
  }

  function toolbar(b) {
    const runs = b ? b.runs : [];
    return el("div", { class: "audit-bar" },
      el("button", { class: "btn btn-light btn-sm", type: "button", id: "audit-run", text: "Run audit", onclick: () => run({}) }),
      runs.length ? el("label", { class: "item-sub" }, "Run ", el("select", { id: "audit-runs", onchange: (e) => show(e.target.value) },
        runs.slice().reverse().map((r) => el("option", { value: r, selected: b && r === b.manifest.run_id ? "selected" : null, text: r })))) : null,
      el("button", { class: "linkbtn", type: "button", text: "Parameters", onclick: () => toggle("audit-params") }),
      el("button", { class: "linkbtn", type: "button", text: "Overrides", onclick: () => toggle("audit-overrides") }),
      el("button", { class: "linkbtn", type: "button", text: "Intake", onclick: () => { toggle("audit-intake"); loadIntake(); } }),
      el("button", { class: "btn btn-light btn-sm", type: "button", text: "Plan", onclick: () => { toggle("audit-plan"); loadPlan(); } }),
      b ? el("a", { class: "linkbtn", href: `/api/sessions/${SID}/audit/${b.manifest.run_id}/report.html`, download: `prism_audit_${b.manifest.run_id}.html`, text: "Export this audit" }) : null,
      b ? el("button", { class: "btn btn-sm", type: "button", text: "Final report →", onclick: () => window.PRISM_FINAL && window.PRISM_FINAL.open({ session_id: SID }) }) : null,
      el("span", { class: "spacer" }),
      el("button", { class: "linkbtn", type: "button", text: "Close", onclick: close }));
  }
  function toggle(id) { const n = $(id); if (n) n.classList.toggle("hidden"); }
  function close() { box().classList.add("hidden"); window.PRISM_SESSION && window.PRISM_SESSION.showStart(); }

  function frame(content, b, diff) {
    box().replaceChildren(
      el("div", { class: "panel-head" }, el("h2", { text: "Tier 1 audit" }), el("span", { class: "meta", text: "deterministic · no AI" })),
      el("div", { class: "panel-body" }, toolbar(b), el("div", { id: "audit-error", class: "alert alert-error hidden" }),
        paramsDrawer(b), overridesDrawer(), intakeDrawer(), planDrawer(), chatBox(), diff || null, content));
    loadOverrides();
    renderChat();
  }
  function error(msg) { const n = $("audit-error"); if (n) { n.textContent = msg; n.classList.remove("hidden"); } }

  async function show(runId, diff) {
    let b;
    try { b = await api(`/api/sessions/${SID}/audit/${runId}`); } catch (e) { error(e.message); return; }
    CUR = b;
    const root = el("div", { id: "audit-report" });
    frame(root, b, diff);
    CTRL = R().render(root, b, { live: true, actions: el("span") });
    return CTRL;
  }

  // ---------------------------------------------------------------- the AI operator (v3 §8)
  function chatBox() {
    const input = el("textarea", { class: "input", rows: "2", id: "audit-chat-input",
      placeholder: "Ask the AI: explain A4, run A7 with permutations = 199, colour the PCA by plate, mark S12 as qc…" });
    const send = async () => {
      const message = input.value.trim();
      if (!message) return;
      input.value = "";
      CHAT.push({ who: "you", text: message });
      renderChat();
      try {
        const r = await api(`/api/sessions/${SID}/audit/chat`, { message, run_id: CUR ? CUR.manifest.run_id : null });
        CHAT.push({ who: "ai", r });
        await applyTools(r);
      } catch (e) { CHAT.push({ who: "ai", r: { reply: null, error: e.message, tool_results: [] } }); }
      renderChat();
    };
    input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
    return el("details", { class: "audit-chat", open: CHAT.length ? "" : null },
      el("summary", {}, el("b", { text: "Ask the AI about these results" }), el("span", { class: "item-sub", text: " · optional" })),
      el("div", { id: "audit-chat-log" }),
      el("div", { class: "audit-chat-row" }, input, el("button", { class: "btn btn-sm", type: "button", id: "audit-chat-send", text: "Ask", onclick: send })),
      el("p", { class: "item-sub", text: "The AI sees the schemas and these findings only, never your data rows. Its tools run the same deterministic audit; every call is in the ledger. Parameters change only when your message names them." }));
  }
  async function applyTools(r) {
    for (const t of r.tool_results || []) {
      if (!t.ok) continue;
      if (t.tool === "run_audit" || t.tool === "rerun") {
        const prev = CUR ? CUR.manifest.run_id : null;
        let diff = null;
        if (prev) { const cmp = await api(`/api/sessions/${SID}/audit/${t.result.run_id}/compare/${prev}`).catch(() => null); if (cmp) diff = R().diffPanel(cmp); }
        await show(t.result.run_id, diff);
      }
    }
    for (const t of r.tool_results || []) {
      if (!t.ok || !CTRL) continue;
      if (t.tool === "show_finding") CTRL.focus(t.result.finding);
      if (t.tool === "plot") CTRL.setView(t.result.view);
    }
  }
  function renderChat() {
    const log = $("audit-chat-log");
    if (!log) return;
    log.replaceChildren(...CHAT.map((m) => m.who === "you" ? el("div", { class: "audit-msg you", text: m.text }) : aiMsg(m.r)));
    log.scrollTop = log.scrollHeight;
  }
  function aiMsg(r) {
    const tools = (r.tool_results || []).map((t) => {
      if (!t.ok) return el("li", { class: "warn-text", text: `${t.tool}: not done — ${t.error}` });
      const x = t.result;
      if (t.tool === "set_override") return el("li", {}, `Proposed override: ${x.proposal.kind} · ${x.proposal.sample || x.proposal.column || x.proposal.key}${x.proposal.role ? ` → ${x.proposal.role}` : ""}${x.proposal.value != null ? ` = ${JSON.stringify(x.proposal.value)}` : ""} `,
        el("button", { class: "btn btn-sm", type: "button", text: "Apply", onclick: async (e) => {
          try { await api(`/api/sessions/${SID}/overrides`, x.proposal); e.target.replaceWith(el("span", { class: "item-sub", text: "applied: used by the next run" })); loadOverrides(); }
          catch (err) { error(err.message); } } }));
      if (t.tool === "run_audit" || t.tool === "rerun") return el("li", { text: `${t.tool}: run ${x.run_id}` });
      if (t.tool === "show_finding") return el("li", { text: `Showing ${x.finding}` });
      if (t.tool === "plot") return el("li", { text: `PCA coloured by ${x.view.color_by || "—"}, PC${x.view.pcs[0]} vs PC${x.view.pcs[1]}` });
      if (t.tool === "request_addition") return el("li", { text: `Written as a request: ${x.request} (nothing ran)` });
      return el("li", { text: t.tool });
    });
    return el("div", { class: "audit-msg ai" },
      r.reply ? el("div", { text: r.reply }) : el("div", { class: "warn-text", text: r.error || "No answer." }),
      (r.unverified_numbers || []).length ? el("div", { class: "warn-text", text: `Numbers not found in the findings: ${r.unverified_numbers.join(", ")} — check them before relying on them.` }) : null,
      tools.length ? el("ul", { class: "np" }, tools) : null);
  }

  async function run(params) {
    const btn = $("audit-run");
    if (btn) { btn.disabled = true; btn.textContent = "Running…"; }
    const prev = CUR ? CUR.manifest.run_id : null;
    try {
      const man = await job(`/api/sessions/${SID}/audit/start`, { params }, (what, sec) => {
        const b = $("audit-run"); if (b) b.textContent = `Running · ${what} · ${Math.round(sec)} s`; });
      let diff = null;
      if (prev) {
        const cmp = await api(`/api/sessions/${SID}/audit/${man.run_id}/compare/${prev}`).catch(() => null);
        if (cmp) diff = R().diffPanel(cmp);
      }
      await show(man.run_id, diff);
      window.PRISM_SESSION && window.PRISM_SESSION.refresh();
    } catch (e) { error(e.message); if (btn) { btn.disabled = false; btn.textContent = "Run audit"; } }
  }

  function paramsDrawer(b) {
    const d = (DEFAULTS && DEFAULTS.defaults) || {}, heur = new Set((DEFAULTS && DEFAULTS.heuristic) || []);
    const cur = b ? b.manifest.params : d;
    const inputs = {};
    const rows = Object.keys(d).map((k) => {
      const v = cur[k] !== undefined ? cur[k] : d[k];
      inputs[k] = el("input", { class: "input input-sm", value: Array.isArray(v) ? JSON.stringify(v) : String(v), "aria-label": k });
      return el("div", { class: "audit-param" }, el("label", {}, el("code", { text: k }), heur.has(k) ? el("span", { class: "ar-pill heuristic", text: "heuristic" }) : null,
        el("span", { class: "item-sub", text: ` default ${Array.isArray(d[k]) ? JSON.stringify(d[k]) : d[k]}` })), inputs[k]);
    });
    const go = () => {
      const params = {};
      for (const [k, inp] of Object.entries(inputs)) {
        let v; try { v = JSON.parse(inp.value); } catch (_) { v = inp.value; }
        if (JSON.stringify(v) !== JSON.stringify(d[k])) params[k] = v;
      }
      run(params);
    };
    return el("div", { id: "audit-params", class: "audit-drawer hidden" },
      el("p", { class: "item-sub", text: "Changed values are recorded in the run manifest (its parameter hash changes). The new run is compared with the one shown." }),
      el("div", { class: "audit-params" }, rows),
      el("div", { class: "guide-actions" }, el("button", { class: "btn btn-sm", type: "button", text: "Re-run with these parameters", onclick: go })));
  }

  function overridesDrawer() {
    const kind = el("select", { class: "input input-sm" }, ["sample_role", "exclude_from_audit", "batch_variable", "design_variable", "source_variable", "param",
      "processing_state", "control_feature", "feature_level", "zeros_are_missing", "floor_censored"].map((k) => el("option", { value: k, text: k })));
    const target = el("input", { class: "input input-sm", placeholder: "sample, column or parameter" });
    const value = el("input", { class: "input input-sm", placeholder: "role (qc, blank, pool, study) or value" });
    const reason = el("input", { class: "input input-sm", placeholder: "reason (optional)" });
    const add = async () => {
      const k = kind.value, body = { kind: k, reason: reason.value };
      if (k === "sample_role" || k === "exclude_from_audit") body.sample = target.value.trim();
      else if (k === "param") { body.key = target.value.trim(); try { body.value = JSON.parse(value.value); } catch (_) { body.value = value.value; } }
      else if (k === "processing_state") { body.item = target.value.trim(); body.value = value.value.trim(); }
      else if (k === "control_feature") body.feature = target.value.trim();
      else if (k === "feature_level") body.level = value.value.trim();
      else if (k === "zeros_are_missing" || k === "floor_censored") body.value = value.value.trim();
      else body.column = target.value.trim();
      if (k === "sample_role") body.role = value.value.trim();
      try { await api(`/api/sessions/${SID}/overrides`, body); target.value = value.value = reason.value = ""; loadOverrides(); }
      catch (e) { error(e.message); }
    };
    return el("div", { id: "audit-overrides", class: "audit-drawer hidden" },
      el("p", { class: "item-sub", text: "Overrides are inputs to the next run, never edits to results. Each one records who and when." }),
      el("div", { id: "audit-ov-list" }),
      el("div", { class: "audit-ov-form" }, kind, target, value, reason, el("button", { class: "btn btn-sm", type: "button", text: "Add", onclick: add })));
  }
  async function loadOverrides() {
    const list = $("audit-ov-list");
    if (!list) return;
    try {
      const { overrides } = await api(`/api/sessions/${SID}/overrides`);
      list.replaceChildren(overrides.length ? el("ul", { class: "np" }, overrides.map((o) => el("li", {},
        el("code", { text: o.override_id }), ` ${o.kind} · ${o.sample || o.column || o.key || ""}${o.role ? ` → ${o.role}` : ""}${o.value != null ? ` = ${JSON.stringify(o.value)}` : ""}${o.reason ? ` (${o.reason})` : ""} `,
        el("button", { class: "linkbtn", type: "button", text: "remove", onclick: async () => {
          try { await api(`/api/sessions/${SID}/overrides/${o.override_id}`, undefined, "DELETE"); loadOverrides(); } catch (e) { error(e.message); } } }))))
        : el("p", { class: "item-sub", text: "No overrides." }));
    } catch (e) { list.replaceChildren(el("p", { class: "item-sub", text: e.message })); }
  }

  // ---------------------------------------------------------------- v4: intake and the plan
  function intakeDrawer() {
    return el("div", { id: "audit-intake", class: "audit-drawer hidden" },
      el("p", { class: "item-sub", text: "Four answers unlock the study-dependent steps. An empty field is fine: the steps that need it say so." }),
      el("div", { id: "audit-intake-form" }));
  }
  async function loadIntake() {
    const box_ = $("audit-intake-form");
    if (!box_) return;
    let doc;
    try { doc = await api(`/api/sessions/${SID}/intake`); } catch (e) { box_.replaceChildren(el("p", { class: "item-sub", text: e.message })); return; }
    const cur = doc.intake || {}, F = doc.fields || {};
    const sel = (name, options) => el("select", { class: "input input-sm", name },
      [el("option", { value: "", text: "— not answered —" }), ...options.map((o) => el("option", { value: o, text: o, selected: cur[name] === o ? "selected" : null }))]);
    const txt = (name) => el("input", { class: "input input-sm", name, value: (cur[name] || []).join ? (cur[name] || []).join(", ") : (cur[name] || ""), placeholder: "comma-separated columns" });
    const fields = { contrast_variable: sel("contrast_variable", ["none", ...(doc.columns || [])]), protected_variables: txt("protected_variables"),
      nuisance_variables: txt("nuisance_variables"), question_type: sel("question_type", doc.question_types || []),
      downstream_family: sel("downstream_family", doc.downstream_families || []) };
    const save = async () => {
      const body = {};
      for (const [k, n] of Object.entries(fields)) if (n.value.trim()) body[k] = n.value.trim();
      try { await api(`/api/sessions/${SID}/intake`, body); loadIntake(); loadPlan(); } catch (e) { error(e.message); }
    };
    box_.replaceChildren(el("div", { class: "audit-params" }, Object.entries(fields).map(([k, n]) =>
      el("div", { class: "audit-param" }, el("label", {}, el("code", { text: k }), el("span", { class: "item-sub", text: ` ${F[k] || ""}` })), n))),
      el("div", { class: "guide-actions" }, el("button", { class: "btn btn-sm", type: "button", text: "Save intake", onclick: save }),
        el("span", { class: "item-sub", text: doc.version ? ` version ${doc.version}` : " not answered yet" })));
  }
  function planDrawer() {
    return el("div", { id: "audit-plan", class: "audit-drawer hidden" }, el("div", { id: "audit-plan-body" },
      el("p", { class: "item-sub", text: "Building the plan…" })));
  }
  let PLAN = null, POS = 0;
  async function loadPlan() {
    const body = $("audit-plan-body");
    if (!body) return;
    try {
      PLAN = await job(`/api/sessions/${SID}/plan/start`, {}, (what, sec) => body.replaceChildren(
        el("p", { class: "item-sub", text: `Building the plan · ${what} · ${Math.round(sec)} s` })));
    } catch (e) { body.replaceChildren(el("p", { class: "alert alert-error", text: e.message })); return; }
    const actions = PLAN.entries.filter((e) => !e.checkpoint);
    POS = Math.min(POS, Math.max(0, actions.length - 1));
    renderPlan();
  }
  function renderPlan() {
    const body = $("audit-plan-body");
    const actions = PLAN.entries.filter((e) => !e.checkpoint);
    const qs = (PLAN.questions || []).map((q) => el("div", { class: "audit-q" }, el("b", { text: `${q.dataset}: ${q.text}` }),
      el("p", { class: "item-sub", text: [q.reason, ...(q.evidence || []), ...(q.platform_notes || []).map((n) => `Note: ${n.text}`), q.consequence].filter(Boolean).join(" ") }),
      el("div", { class: "guide-actions" }, (q.options || ["yes", "no"]).map((o) => el("button", { class: "btn btn-light btn-sm", type: "button", text: o,
        onclick: async () => { try { await api(`/api/sessions/${SID}/overrides`, Object.assign({}, q.override, { value: o, reason: "answered in the plan" })); await run({}); loadPlan(); } catch (e) { error(e.message); } } })))));
    const e = actions[POS];
    const STATUS = { "Recommended": "rec", "Alternative": "alt", "Do not automate": "dna", "Insufficient metadata": "ins", "Not applicable": "na" };
    const card = e ? el("article", { class: "v4-entry" },
      el("header", {}, el("span", { class: "v4-id", text: e.step }), el("b", { text: `${e.step_name} · ${e.dataset}/${e.assay}` }), el("span", { class: `v4-st ${STATUS[e.status] || ""}`, text: e.status })),
      el("p", { text: (e.because || []).join(" ") }),
      el("p", { class: "item-sub", text: `Evidence: ${Object.entries(e.evidence_levels || {}).map(([k, v]) => `${k} (${v})`).join(", ") || "—"}${(e.evidence || []).length ? ` · ${e.evidence.join(", ")}` : ""}` }),
      (e.blocked_by || []).length ? el("p", { class: "item-sub", text: `Blocked by: ${e.blocked_by.join(", ")}` }) : null,
      el("ul", { class: "v4-cands" }, (e.candidates || []).map((c, i) => el("li", {},
        el("span", { class: `v4-st ${STATUS[c.status] || ""}`, text: c.status }), ` ${c.label} `,
        ["Recommended", "Alternative"].includes(e.status) && ["Recommended", "Alternative"].includes(c.status)
          ? el("button", { class: "btn btn-sm", type: "button", text: c.op ? "Approve and run" : "Approve", onclick: (ev) => approve(e, i, ev.target) }) : null))),
      el("div", { class: "guide-actions" },
        el("button", { class: "linkbtn", type: "button", text: "Reject", onclick: async () => { try { await api(`/api/sessions/${SID}/plan/${encodeURIComponent(e.entry_id)}/reject`, { reason: "rejected in the plan view" }); POS += 1; renderPlan(); } catch (er) { error(er.message); } } }))) : el("p", { class: "item-sub", text: "Nothing to decide." });
    body.replaceChildren(...[
      qs.length ? el("div", {}, el("h3", { text: "Questions" }), qs) : null,
      el("div", { class: "audit-bar" },
        el("button", { class: "linkbtn", type: "button", text: "← previous", disabled: POS <= 0, onclick: () => { POS -= 1; renderPlan(); } }),
        el("span", { class: "item-sub", text: actions.length ? `entry ${POS + 1} of ${actions.length}` : "" }),
        el("button", { class: "linkbtn", type: "button", text: "next →", disabled: POS >= actions.length - 1, onclick: () => { POS += 1; renderPlan(); } })),
      card,
      el("p", { class: "item-sub", text: "Nothing runs without approval. Approving calls the executor on a new data state, then re-runs that checkpoint's audit." })]
      .filter(Boolean));
  }
  async function approve(e, i, btn) {
    btn.disabled = true; btn.textContent = "Running…";
    try {
      const out = await job(`/api/sessions/${SID}/plan/${encodeURIComponent(e.entry_id)}/approve/start`, { candidate: i },
        (what, sec) => { btn.textContent = `${what} · ${Math.round(sec)} s`; });
      if (out.run_id) await show(out.run_id);
      const d = $("audit-plan"); if (d) d.classList.remove("hidden");
      POS += 1;
      await loadPlan();
    } catch (er) { error(er.message); btn.disabled = false; btn.textContent = "Approve and run"; }
  }

  window.PRISM_AUDIT = { open, show, run };
})();
