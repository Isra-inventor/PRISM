// PRISM Step 0 — guided confirmation wizard.
// The backend is the source of truth (the "draft"); every step sends a
// decision to /api/confirm-step and re-renders from the returned draft.

(() => {
  "use strict";

  const API = "";
  const ACCEPTED = [".csv", ".tsv", ".txt", ".tab"];
  const STEPS = [
    ["layout", "Layout"], ["feature_id", "Feature ID"], ["annotations", "Annotations"], ["values", "Values"],
    ["samples", "Samples"], ["sample_info", "Sample info"], ["design", "Design"], ["history", "History"], ["review", "Review"],
  ];
  const PREVIEW_COLS = 30;
  const S = { vocab: null, defs: {}, historyQ: [], scope: "", session: null, draft: null, digests: null, step: "layout",
              local: {}, aiOn: true, finalized: null, busy: false };

  // ------------------------------------------------------------ helpers
  const $ = (id) => document.getElementById(id);
  function el(tag, attrs = {}, ...children) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else if (k === "checked" || k === "selected" || k === "disabled") n[k] = !!v;
      else if (k === "value") n.value = v;
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat(Infinity)) if (c != null && c !== false) n.append(c.nodeType ? c : document.createTextNode(String(c)));
    return n;
  }
  const code = (t) => el("code", { text: t });
  const pretty = (s) => (s || "").replace(/_/g, " ");
  function fmtNum(x) {
    if (x == null || Number.isNaN(x)) return "—";
    const a = Math.abs(x);
    if (a !== 0 && (a >= 1e5 || a < 1e-3)) return x.toExponential(1).replace("e+", "e");
    return String(+x.toPrecision(3));
  }
  const pct = (x) => (x == null ? "—" : `${Math.round(x * 100)}%`);
  const show = (node, on = true) => node.classList.toggle("hidden", !on);
  let _gbiSrc = null, _gbi = {};
  const groupsById = () => {   // cached per groups array (every change replaces the array)
    const src = S.session?.groups || [];
    if (src !== _gbiSrc) { _gbiSrc = src; _gbi = Object.fromEntries(src.map((g) => [g.group_id, g])); }
    return _gbi;
  };
  const G = (gid) => groupsById()[gid];
  const D = (gid) => S.draft.groups[gid];
  const layout = () => S.draft?.layout?.value;
  const header = (i) => S.session.header[i];

  async function api(path, body, method = "POST") {
    const opts = { method, headers: {} };
    if (body instanceof FormData) opts.body = body;
    else if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
    const r = await fetch(API + path, opts);
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : (JSON.stringify(j.detail || j) || `HTTP ${r.status}`));
    return j;
  }

  // ------------------------------------------------------------ progress
  function startProgress(slot, label) {
    const id = (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : `p${Date.now()}${Math.random().toString(16).slice(2)}`;
    const node = $("progress-template").content.firstElementChild.cloneNode(true);
    slot.replaceChildren(node);
    const q = (c) => node.querySelector(c);
    const started = Date.now();
    let last = Date.now(), lastKey = "", server = null, upload = null, stopped = false;
    const fmtS = (s) => (s < 60 ? `${Math.round(s)}s` : `${Math.floor(s / 60)}m ${String(Math.round(s % 60)).padStart(2, "0")}s`);
    function render() {
      const idle = (Date.now() - last) / 1000;
      let p = server ? server.percent : Math.round((upload || 0) * 2);
      let lab = label;
      if (upload != null && upload < 1) lab = `Uploading (${Math.round(upload * 100)}%)`;
      else if (server) lab = { reading: "Reading the file", detecting: "Grouping columns", ai: "AI is labelling column groups",
                               done: "Done", error: "Stopped with an error" }[server.stage] || label;
      q(".p-label").textContent = lab;
      q(".progress-pct").textContent = `${p}%`;
      q(".progress-track i").style.width = `${p}%`;
      q(".p-detail").textContent = server && server.batches_total ? `${server.batches_done} of ${server.batches_total} AI batch(es) done` : "";
      q(".p-elapsed").textContent = `${fmtS((Date.now() - started) / 1000)} elapsed`;
      const idleEl = q(".p-idle");
      idleEl.textContent = idle < 3 ? "active" : `last update ${fmtS(idle)} ago`;
      const stalled = idle >= 60;
      idleEl.classList.toggle("warn", stalled);
      node.classList.toggle("stalled", stalled);
      q(".progress-stall").textContent = `No progress for ${fmtS(idle)}. This is the tool or the AI service, not your data: `
        + "check the [PRISM AI] lines in the server terminal.";
      show(q(".progress-stall"), stalled);
    }
    async function poll() {
      if (stopped) return;
      try {
        const r = await fetch(`${API}/api/progress/${id}`, { cache: "no-store" });
        if (r.ok) {
          const p = await r.json();
          const key = `${p.stage}|${p.percent}|${p.batches_done}|${p.log.length ? p.log[p.log.length - 1].t : ""}`;
          if (key !== lastKey) {
            lastKey = key; last = Date.now();
            q(".progress-log").replaceChildren(...p.log.map((e) => el("li", { title: e.msg }, el("b", { text: `${Math.round(e.t)}s` }), e.msg)));
          }
          server = p;
        }
      } catch (_) { /* keep polling */ }
      if (!stopped) setTimeout(poll, 1000);
    }
    const tick = setInterval(render, 500);
    render(); setTimeout(poll, 300);
    return { id, onUpload(f) { upload = f; last = Date.now(); render(); },
             stop() { stopped = true; clearInterval(tick); slot.replaceChildren(); } };
  }

  async function busy(label, fn) {
    if (S.busy) return;
    S.busy = true;
    document.body.classList.add("is-busy");
    const prog = startProgress($("busy-slot"), label);
    try { return await fn(prog.id); }
    finally { prog.stop(); S.busy = false; document.body.classList.remove("is-busy"); }
  }

  function postWithUpload(url, fd, onUpload) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", url);
      xhr.upload.onprogress = (e) => { if (e.lengthComputable) onUpload(e.loaded / e.total); };
      xhr.upload.onload = () => onUpload(1);
      xhr.onload = () => { let b = {}; try { b = JSON.parse(xhr.responseText); } catch (_) {} resolve({ status: xhr.status, body: b }); };
      xhr.onerror = () => reject(new Error("Network error: is the server still running?"));
      xhr.send(fd);
    });
  }

  // ------------------------------------------------------------ boot
  Promise.all([api("/api/vocabulary", undefined, "GET"), api("/api/health", undefined, "GET")]).then(([v, h]) => {
    S.vocab = v.vocabulary; S.defs = v.definitions; S.historyQ = v.history_questions; S.scope = v.scope_description || "";
    $("ai-status").textContent = h.ai_available ? `AI: ${h.ai_provider} · ${h.ai_models[0] || ""}` : "AI unavailable · manual mode";
    $("ai-status").title = h.ai_available ? "Used to label column groups; only statistics are sent, never raw rows."
      : (h.ai_unavailable_reason || "");
    if (!h.ai_available) { $("ai-toggle").checked = false; S.aiOn = false; }
  }).catch(() => { $("ai-status").textContent = "Backend unreachable"; });

  $("ai-toggle").addEventListener("change", async (e) => {
    S.aiOn = e.target.checked;
    if (!S.session) return;
    if (!confirm(`Turn AI suggestions ${S.aiOn ? "on" : "off"} and rebuild the proposal? Steps you already confirmed will be asked again.`)) {
      e.target.checked = !S.aiOn; S.aiOn = !S.aiOn; return;
    }
    await propose();
  });

  const input = $("file-input"), dz = $("dropzone");
  input.addEventListener("change", () => input.files[0] && upload(input.files[0]));
  ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));
  $("restart").addEventListener("click", reset);

  function reset() {
    Object.assign(S, { session: null, draft: null, digests: null, step: "layout", local: {}, finalized: null });
    show($("workspace"), false); show($("panel-upload")); show($("restart"), false); show($("upload-error"), false);
    input.value = "";
    window.scrollTo({ top: 0 });
  }

  async function upload(file) {
    show($("upload-error"), false);
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!ACCEPTED.includes(ext)) {
      $("upload-error").textContent = `"${file.name}" is not a CSV/TSV file. PRISM accepts quantified tables only: export a protein, `
        + "peptide or feature table from your analysis software as .csv or .tsv. Raw spectra are not processed.";
      show($("upload-error")); input.value = ""; return;
    }
    dz.style.pointerEvents = "none";
    const prog = startProgress($("upload-progress-slot"), "Uploading");
    try {
      const fd = new FormData();
      fd.append("file", file); fd.append("progress_id", prog.id);
      const { status, body } = await postWithUpload(`${API}/api/upload`, fd, prog.onUpload);
      if (status < 200 || status >= 300) throw new Error(body.detail || `Upload failed (HTTP ${status}).`);
      S.session = body;
      prog.stop();
      show($("panel-upload"), false); show($("workspace")); show($("restart"));
      $("ws-file").replaceChildren(el("b", { text: body.filename }), ` · ${body.n_rows.toLocaleString()} rows × ${body.n_columns.toLocaleString()} columns`);
      renderParseNotes();
      await propose();
    } catch (err) {
      prog.stop();
      $("upload-error").textContent = err.message || String(err);
      show($("upload-error"));
    } finally { dz.style.pointerEvents = ""; input.value = ""; }
  }

  async function propose() {
    $("guide").replaceChildren(el("p", { class: "q", text: "Preparing the proposal…" }));
    await busy(S.aiOn ? "Asking the AI to label column groups" : "Preparing manual mode", async (pid) => {
      try {
        const r = await api("/api/propose", { session_id: S.session.session_id, ai: S.aiOn, progress_id: pid });
        S.session = r.session; S.draft = r.draft; S.digests = r.digests; S.finalized = null;
        S.step = "layout"; S.local = {};
      } catch (err) {
        $("guide").replaceChildren(el("div", { class: "alert alert-error", text: err.message }));
        throw err;
      }
    });
    renderAll(true);
  }

  // ------------------------------------------------------------ roles, steps
  function roleKey(it) {
    if (!it) return "pending";
    if (it.role === "value") return it.keep === false ? "value_excluded" : "value";
    return it.role;
  }
  function stepFor(gid) {
    const it = D(gid), g = G(gid), lay = layout();
    if (!it || !g) return "review";
    const r = it.role;
    if (r === "feature_id") return "feature_id";
    if (r === "feature_annotation") return "annotations";
    if (r === "value") return "values";
    if (r === "sample_id") return "samples";
    if (r === "sample_metadata") return "sample_info";
    if (g.kind === "numeric_block") return "values";
    if (lay === "samples_in_rows") return g.type === "numeric" ? "values" : "sample_info";
    return "annotations";
  }
  function stepStatus(st) { return S.draft?.steps?.[st] || "pending"; }
  function chipClass(gid) {
    const it = D(gid);
    if (!it || it.role === "unresolved") return "k-unresolved";
    const st = stepFor(gid);
    const confirmed = stepStatus(st) === "confirmed";
    return confirmed ? `k-${roleKey(it)}` : "k-pending";
  }
  function roleLabel(it) {
    if (!it) return "pending";
    const lab = it.label ? ` · ${it.label}` : "";
    return {
      feature_id: "feature ID" + lab, feature_annotation: (it.marks_rows_as_suspect ? "flag" : "annotation") + lab,
      sample_id: "sample ID", sample_metadata: `sample info${it.audit_kind ? " · " + pretty(it.audit_kind) : ""}${lab}`,
      ignore: "ignored", unresolved: "unresolved",
      value: `value${it.keep === false ? " · excluded" : ""}${lab}`,
    }[it.role] || it.role;
  }
  function focusGroups(step) {
    if (!S.draft) return [];
    const all = Object.keys(S.draft.groups), lay = layout();
    const byRole = (r) => all.filter((g) => D(g).role === r);
    switch (step) {
      case "layout": return byRole("value").length ? byRole("value") : all.filter((g) => G(g)?.kind === "numeric_block");
      case "feature_id": return lay === "samples_in_rows" ? byRole("value")
        : [...S.draft.feature_identity.group_ids, ...(lay === "long" && S.draft.sample_id_group.value ? [S.draft.sample_id_group.value] : [])];
      case "samples": return lay === "samples_in_columns" ? all.filter((g) => D(g).role === "value" && D(g).keep !== false) : byRole("sample_id");
      case "history": return all.filter((g) => D(g).role === "value" && D(g).keep !== false);
      case "review": return [];
      case "values": {
        const blocks = all.filter((g) => D(g).role === "value");
        return [...blocks, ...all.filter((g) => !blocks.includes(g) && (stepFor(g) === "values" || (lay === "samples_in_rows"
          && G(g)?.type === "numeric" && G(g).n_columns === 1 && ["sample_metadata", "ignore"].includes(D(g).role))))];
      }
      default: return all.filter((g) => stepFor(g) === step);
    }
  }

  // ------------------------------------------------------------ render
  function renderAll(scroll) {
    renderChat();
    renderStepper();
    renderLegend();
    renderNameParts();
    renderPreview();
    renderGuide();
    renderSaw();
    if (scroll) $("workspace").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderStepper() {
    const needs = new Set((S.draft.unresolved || []).filter((u) => !/not confirmed yet/.test(u.what) && u.step !== "history").map((u) => u.step));
    $("stepper").replaceChildren(...STEPS.map(([id, label], k) => el("li", {
      class: [stepStatus(id), id === S.step ? "current" : "", needs.has(id) ? "needs" : ""].join(" ") },
      el("button", { type: "button", onclick: () => go(id), title: stepStatus(id).replace("_", " ") },
        el("b", { text: String(k + 1) }), el("span", { text: label })))));
    const hint = S.draft.signature_hint, aiUsed = S.draft.ai.used, gr = S.draft.grouping || {};
    const tpl = gr.n_templates ? ` from ${gr.n_templates} name template${gr.n_templates > 1 ? "s" : ""}` : "";
    const how = { ai: `grouped by the AI${tpl}${gr.chunks > 1 ? ` in ${gr.chunks} chunks + 1 final call` : ""}`,
                  signature: "grouped from the known format",
                  manual: S.session.groups.length < S.session.n_columns ? "grouped by you" : "not grouped yet: use shared name parts" }[gr.source] || "";
    $("ws-sig").textContent = (aiUsed ? "AI-assisted" : "Manual mode") + ` · ${S.session.groups.length} groups, ${how}`
      + (hint ? ` · ${hint.charAt(0).toUpperCase() + hint.slice(1)}` : "");
  }

  function go(step) {
    S.step = step; S.local = {};
    renderStepper(); renderPreview(); renderGuide();
  }

  function nextStep(from) {
    const idx = STEPS.findIndex(([id]) => id === from);
    for (let k = idx + 1; k < STEPS.length; k++) if (stepStatus(STEPS[k][0]) === "pending" || STEPS[k][0] === "review") return STEPS[k][0];
    return "review";
  }

  function renderLegend() {
    const items = [["feature ID", "var(--c-fid)"], ["annotation", "var(--c-ann)"], ["value", "var(--c-val)"],
      ["value (excluded)", "transparent;border:1px solid #333"], ["sample ID", "var(--c-sid)"], ["sample info", "var(--c-smd)"],
      ["ignored", "transparent;border:1px solid #333"], ["unresolved", "transparent;border:1px solid var(--c-unres)"],
      ["pending", "transparent;border:1px dashed #555"]];
    $("legend").replaceChildren(...items.map(([t, c]) => el("span", {}, el("i", { style: `background:${c}` }), t)));
  }

  function renderParseNotes() {
    const r = S.session.parse_report;
    const notes = [];
    if (r.duplicate_column_names.length) notes.push(`${r.duplicate_column_names.length} duplicated column name(s)`);
    if (r.empty_or_unnamed_headers.length) notes.push(`${r.empty_or_unnamed_headers.length} empty/unnamed header(s)`);
    if (r.decimal_comma_columns.length) notes.push(`decimal commas in ${r.decimal_comma_columns.length} column(s)`);
    if (r.thousands_separator_columns.length) notes.push("thousands separators");
    if (r.whitespace_padded_columns.length) notes.push("padded values");
    if (r.fully_empty_rows) notes.push(`${r.fully_empty_rows} empty line(s)`);
    if (r.ragged_rows.count) notes.push(`${r.ragged_rows.count} ragged row(s)`);
    const tokens = Object.keys(r.missing_value_tokens || {});
    if (tokens.length) notes.push(`missing values written as ${tokens.slice(0, 4).join(", ")}`);
    $("pv-note").textContent = notes.length ? `Parse notes: ${notes.join(" · ")} (nothing changed)` : "No parsing issues found.";
    $("parse-pre").textContent = JSON.stringify(r, null, 2);
  }

  function renderSaw() {
    const d = S.draft;
    const head = d.ai.enabled && d.ai.available
      ? `Provider: ${d.ai.provider} · model: ${(d.ai.models_used || []).join(", ") || d.ai.model || "—"} · prompt ${d.ai.prompt_version} · temperature ${d.ai.temperature}${d.ai.cached ? " · cached proposal" : ""}\nRaw data rows sent: no. Only the digest below.\n\n`
      : "The AI is off: nothing was sent.\n\n";
    $("saw-pre").textContent = head + (S.digests && S.digests.length ? JSON.stringify(S.digests.length === 1 ? S.digests[0] : S.digests, null, 2) : "")
      + (d.last_chat_context ? "\n\nLast chat turn (what the AI received):\n" + JSON.stringify(d.last_chat_context, null, 2) : "");
  }

  // ------------------------------------------------------------ preview table
  function displayEntries(focus) {
    const groups = [...S.session.groups].sort((a, b) => a.indices[0] - b.indices[0]);
    const cond = (g) => {
      const idx = [...g.indices].sort((a, b) => a - b);
      if (idx.length <= 5) return idx.map((i) => ({ i, gid: g.group_id }));
      return [...idx.slice(0, 3).map((i) => ({ i, gid: g.group_id })),
              { i: idx[2] + 0.5, gid: g.group_id, more: idx.length - 4 }, { i: idx[idx.length - 1], gid: g.group_id }];
    };
    const out = [];
    const used = new Set();
    for (const gid of focus) { const g = G(gid); if (g) { out.push(...cond(g)); used.add(gid); } }
    for (const g of groups) {
      if (out.length >= PREVIEW_COLS) break;
      if (!used.has(g.group_id)) { out.push(...cond(g)); used.add(g.group_id); }
    }
    return out.sort((a, b) => a.i - b.i);
  }

  function renderPreview() {
    const focus = new Set(focusGroups(S.step));
    const entries = displayEntries([...focus]);
    const chips = el("tr", { class: "chips" }, el("th", { class: "rn" }));
    for (let k = 0; k < entries.length;) {
      let j = k;
      while (j + 1 < entries.length && entries[j + 1].gid === entries[k].gid) j++;
      const gid = entries[k].gid, it = D(gid);
      const cls = chipClass(gid);
      chips.append(el("th", { colspan: j - k + 1, class: focus.has(gid) ? "focus" : "", "data-gid": gid,
        onclick: () => { if (S.chatOpen) openChat(G(gid).columns.slice(0, 200)); } },
        el("span", { class: `chip ${cls} ${S.hop?.has(gid) ? "hop" : ""}`, title: `${roleLabel(it)} — ${G(gid).n_columns} column(s)` },
          cls === "k-pending" ? `? ${roleLabel(it)}` : roleLabel(it))));
      k = j + 1;
    }
    const names = el("tr", { class: "names" }, el("th", { class: "rn", text: "#" }),
      entries.map((e) => e.more ? el("th", { class: "more", text: `… ${e.more} more` })
        : el("th", { class: focus.has(e.gid) ? "focus" : "", title: S.session.labels[e.i], text: S.session.labels[e.i] })));
    const rows = S.session.preview_rows.map((r, ri) => el("tr", {}, el("td", { class: "rn", text: ri + 1 }),
      entries.map((e) => {
        if (e.more) return el("td", { class: "more", text: "…" });
        const it = D(e.gid);
        const conf = chipClass(e.gid) !== "k-pending";
        return el("td", { class: `${conf ? "t-" + roleKey(it) : ""} ${focus.has(e.gid) ? "focus" : ""}`, title: r[e.i] ?? "", text: r[e.i] ?? "" });
      })));
    $("pv").replaceChildren(el("thead", {}, chips, names), el("tbody", {}, rows));
    const shown = entries.filter((e) => !e.more).length;
    $("pv-meta").textContent = `First ${S.session.preview_rows.length} rows · ${shown} of ${S.session.n_columns} columns shown`;
    // hop: scroll the highlighted group into view, then draw the callout
    const firstGid = [...focus][0];
    const first = (firstGid && $("pv").querySelector(`tr.chips th[data-gid="${firstGid}"]`)) || $("pv").querySelector("tr.chips th.focus");
    const sc = $("pv-scroll");
    if (first) sc.scrollTo({ left: Math.max(0, first.offsetLeft - 60), behavior: "smooth" });
    setTimeout(drawCallout, 380);
    if (S.hop) setTimeout(() => { S.hop = null; }, 1200);
  }

  function drawCallout() {
    const svg = $("callout"), grid = $("ws-grid");
    svg.replaceChildren();
    if (window.innerWidth < 960) return;
    const fg = focusGroups(S.step)[0];
    const target = (fg && $("pv").querySelector(`tr.chips th[data-gid="${fg}"] .chip`)) || $("pv").querySelector("tr.chips th.focus .chip");
    const card = $("guide");
    if (!target || !card) return;
    const gr = grid.getBoundingClientRect(), tr = target.getBoundingClientRect(), cr = card.getBoundingClientRect(),
          sr = $("pv-scroll").getBoundingClientRect();
    let tx = Math.min(Math.max(tr.left + tr.width / 2, sr.left + 8), sr.right - 8) - gr.left;
    const ty = tr.bottom - gr.top + 2;
    const sx = cr.left - gr.left, sy = Math.min(cr.top - gr.top + 70, cr.bottom - gr.top - 20);
    const mx = (sx + tx) / 2;
    const ns = "http://www.w3.org/2000/svg";
    const path = document.createElementNS(ns, "path");
    path.setAttribute("d", `M ${sx} ${sy} C ${mx} ${sy}, ${tx} ${ty + 60}, ${tx} ${ty}`);
    const dot = document.createElementNS(ns, "circle");
    dot.setAttribute("cx", tx); dot.setAttribute("cy", ty); dot.setAttribute("r", 2.5);
    svg.setAttribute("width", gr.width); svg.setAttribute("height", gr.height);
    svg.append(path, dot);
  }
  $("pv-scroll").addEventListener("scroll", () => requestAnimationFrame(drawCallout));
  window.addEventListener("resize", () => requestAnimationFrame(drawCallout));
  window.addEventListener("scroll", () => requestAnimationFrame(drawCallout), { passive: true });

  // ------------------------------------------------------------ guide building blocks
  // Every proposal is editable in place: closed fields are dropdowns built from
  // /api/vocabulary, open fields are text inputs with suggestions (labels already
  // used in this session). Local edits live in S.local until the step is confirmed.
  function provBadge(p, source) {
    const map = { computed: ["Computed", "p-computed"], ai_proposed_confirmed: ["AI suggestion", "p-ai"],
                  ai_proposed_corrected: ["You (corrected AI)", "p-you"], user_set: ["You", "p-you"] };
    if (source === "none" && p === "user_set") return el("span", { class: "badge p-none", text: "No proposal" });
    const [t, c] = map[p] || ["—", "p-none"];
    return el("span", { class: `badge ${c}`, text: t });
  }
  function confBar(c) {
    if (c == null) return null;
    return el("span", { class: `conf ${c < 0.6 ? "low" : ""}`, title: "confidence" },
      el("span", { class: "conf-bar" }, el("i", { style: `width:${Math.round(c * 100)}%` })), el("span", { class: "conf-num", text: pct(c) }));
  }
  function warnList(v, claimed) {
    const msgs = (v?.messages || []).map((m) => el("li", { class: v.status === "contradicted" ? "bad" : "", text: m }));
    if (claimed) {
      const what = [pretty(claimed.role), claimed.audit_kind && pretty(claimed.audit_kind), claimed.label && `“${claimed.label}”`].filter(Boolean).join(" / ");
      msgs.unshift(el("li", { class: "bad", text: `The AI proposed ${what}, but the data contradicts it. Please choose.` }));
    }
    return msgs.length ? el("ul", { class: "warns" }, msgs) : null;
  }
  function metaRow(it) {
    return el("div", { class: "meta-row" }, provBadge(it.provenance, it.source), confBar(it.source === "none" ? null : it.confidence),
      it.validation && it.validation.status !== "ok" ? el("span", { class: `badge v-${it.validation.status}`, text: it.validation.status }) : null);
  }
  function evidence(t) { return t ? el("div", { class: "evidence", text: t }) : null; }
  function select(options, value, onchange, { placeholder, labels, bad } = {}) {
    const s = el("select", { class: `select ${bad ? "bad" : ""}` },
      placeholder && (value == null || !options.includes(value)) ? el("option", { value: "", text: placeholder, selected: true, disabled: true }) : null,
      options.map((o) => el("option", { value: o, selected: o === value, title: S.defs[o] || "", text: labels ? labels(o) : pretty(o) })));
    s.addEventListener("change", () => onchange(s.value));
    return s;
  }
  // free-text input with suggestions; typing never re-renders (keeps focus)
  function textInput(value, onInput, { placeholder = "", list, cls = "", onCommit } = {}) {
    const i = el("input", { class: `input ${cls}`, type: "text", value: value ?? "", placeholder, list: list || null, spellcheck: "false" });
    i.addEventListener("input", () => onInput(i.value));
    if (onCommit) i.addEventListener("change", () => onCommit(i.value));
    return i;
  }
  function datalists() {
    const sug = S.draft.suggestions || {};
    const mk = (id, vals) => el("datalist", { id }, (vals || []).map((v) => el("option", { value: v })));
    return el("div", { class: "hidden" }, mk("dl-label", sug.label), mk("dl-assay", sug.assay_label), mk("dl-omics", sug.omics_type),
      mk("dl-software", sug.source_software), mk("dl-sample-label", sug.sample_label));
  }
  function fieldBox(label, node, cls = "") { return el("label", { class: `field ${cls}` }, el("span", { class: "field-k", text: label }), node); }
  function local(gid) { S.local.items = S.local.items || {}; return (S.local.items[gid] = S.local.items[gid] || {}); }
  function cur(gid, f) { const l = S.local.items?.[gid]; return l && f in l ? l[f] : D(gid)[f]; }
  function colsText(g) { return g.n_columns <= 3 ? g.columns.join(", ") : `${g.columns.slice(0, 2).join(", ")} … ${g.columns[g.columns.length - 1]} (${g.n_columns} columns)`; }
  const aiReady = () => S.draft.ai.enabled && S.draft.ai.available;
  const assayLabels = () => S.draft.assays.map((a, k) => S.local.assays?.[k]?.assay_label ?? a.assay_label);

  function roleSelect(gid, allowed) {
    const roles = allowed || S.vocab.column_role.filter((r) => r !== "unresolved");
    const v = cur(gid, "role");
    return select(roles.includes(v) || v === "unresolved" ? roles : [...roles, v], v === "unresolved" ? null : v,
      (x) => { local(gid).role = x; renderGuide(); }, { placeholder: "choose a role…", bad: v === "unresolved", labels: (r) => pretty(r) });
  }
  function auditKindSelect(gid) {
    const v = cur(gid, "audit_kind");
    return select(S.vocab.audit_kind, S.vocab.audit_kind.includes(v) ? v : null, (x) => { local(gid).audit_kind = x; renderGuide(); },
      { placeholder: "what is it for the audit?", bad: !S.vocab.audit_kind.includes(v) });
  }
  function keepBox(gid) {
    const cb = el("input", { type: "checkbox", checked: cur(gid, "keep") !== false });
    cb.addEventListener("change", () => { local(gid).keep = cb.checked; });
    return el("label", { class: "check" }, cb, "keep in the outputs");
  }
  // suspect flag: which value means "flagged"?
  
  function suspectControls(gid) {
    const it = D(gid), on = !!cur(gid, "marks_rows_as_suspect");
    const cb = el("input", { type: "checkbox", checked: on });
    cb.addEventListener("change", () => { local(gid).marks_rows_as_suspect = cb.checked; renderGuide(); });
    const out = [el("label", { class: "check" }, cb, "marks rows as suspect (decoy, contaminant…)")];
    if (!on) return out;
    const counts = it.flag_values || it.value_counts;
    if (!counts) {
      out.push(el("div", { class: "item-sub", text: G(gid).n_columns > 1 ? "Several columns: record as a flag group; nothing is removed."
        : "This column has more than 5 distinct values, so no single ‘flagged’ value can be chosen. It is recorded as a flag column; nothing is removed." }));
      return out;
    }
    const fv = cur(gid, "flagged_values") || [];
    const opts = Object.keys(counts);
    out.push(el("div", { class: `full flagpick ${fv.length ? "" : "bad"}` }, el("span", { class: "field-k", text: "Which values mark a row as flagged? (one or more)" }),
      opts.map((v) => {
        const cb = el("input", { type: "checkbox", checked: fv.includes(v) });
        cb.addEventListener("change", () => { const now = new Set(cur(gid, "flagged_values") || []); cb.checked ? now.add(v) : now.delete(v); local(gid).flagged_values = opts.filter((o) => now.has(o)); renderGuide(); });
        return el("label", { class: "check" }, cb, `${v === "" ? "(empty)" : v} — ${counts[v]} row(s)`);
      })));
    const n = fv.reduce((a, v) => a + (counts[v] || 0), 0);
    if (fv.length) out.push(el("div", { class: "notice", text: `${n} row(s) flagged — nothing is removed now; recorded for the audit step.` }));
    return out;
  }

  function reconsiderOne(gid) {
    return el("button", { class: "linkbtn", type: "button", text: "Ask the AI about this one", onclick: () => openChat(G(gid).columns.slice(0, 200)) });
  }

  // generic item editor (annotations, sample info, other columns)
  function itemCard(gid, { roles } = {}) {
    const it = D(gid), g = G(gid), role = cur(gid, "role");
    const unres = role === "unresolved" || (role === "sample_metadata" && !S.vocab.audit_kind.includes(cur(gid, "audit_kind")))
      || (cur(gid, "marks_rows_as_suspect") && (it.flag_values || it.value_counts) && !(cur(gid, "flagged_values") || []).length);
    const ctr = el("div", { class: "item-controls" },
      fieldBox("Role", roleSelect(gid, roles)),
      fieldBox("What it is (your words)", textInput(cur(gid, "label"), (v) => { local(gid).label = v; }, { placeholder: "e.g. gene symbol", list: "dl-label" })));
    if (role === "sample_metadata") {
      ctr.append(fieldBox("Audit kind", auditKindSelect(gid)),
        fieldBox("Detail (optional)", textInput(cur(gid, "detail"), (v) => { local(gid).detail = v; }, { placeholder: cur(gid, "audit_kind") === "timepoint" ? "e.g. ordinal label, days, date" : "optional" })));
    }
    if (role === "feature_annotation") ctr.append(el("div", { class: "full" }, suspectControls(gid)));
    if (role === "value") {
      ctr.append(fieldBox("Assay", textInput(cur(gid, "assay_label") || assayLabels()[0], (v) => { local(gid).assay_label = v; }, { list: "dl-assay" })));
    }
    if (["feature_annotation", "sample_metadata"].includes(role)) ctr.append(el("div", { class: "full" }, keepBox(gid)));
    return el("div", { class: `item ${unres ? "unres" : ""}`, onmouseenter: () => hoverGroup(gid, true), onmouseleave: () => hoverGroup(gid, false) },
      el("div", { class: "item-head" }, el("span", { class: "item-name", text: colsText(g) }), metaRow(it)),
      ctr,
      el("div", { class: "item-sub", text: it.hint || "" }), evidence(it.evidence), warnList(it.validation, it.claimed),
      el("div", { class: "item-foot" }, reconsiderOne(gid), structureTools(gid)));
  }
  function hoverGroup(gid, on) {
    $("pv").querySelectorAll(`th[data-gid="${gid}"] .chip`).forEach((c) => c.style.outline = on ? "1px solid #fff" : "");
  }
  const ITEM_FIELDS = ["role", "label", "assay_label", "audit_kind", "marks_rows_as_suspect", "flagged_values", "detail", "keep", "family"];
  function itemDecisions(gids) {
    return gids.map((gid) => {
      const l = S.local.items?.[gid] || {};
      const out = { group_id: gid };
      for (const f of ITEM_FIELDS) if (f in l) out[f] = l[f];
      return out;
    }).filter((d) => Object.keys(d).length > 1);
  }
  function checkItems(gids) {
    for (const gid of gids) {
      const role = cur(gid, "role"), name = G(gid).columns[0];
      if (role === "unresolved") throw new Error(`Choose a role for ${name}.`);
      if (role === "sample_metadata" && !S.vocab.audit_kind.includes(cur(gid, "audit_kind"))) throw new Error(`Choose an audit kind for ${name}.`);
      if (cur(gid, "marks_rows_as_suspect") && (D(gid).flag_values || D(gid).value_counts) && !(cur(gid, "flagged_values") || []).length)
        throw new Error(`Choose which value(s) of ${name} mean ‘flagged’.`);
    }
  }

  // ------------------------------------------------------------ grouping: your decisions
  // Groups are the AI's proposal (or the known format's, or yours). You can merge
  // groups ("these are the same thing"), take columns out, or group columns that
  // share a name part. Code applies only what you confirm here.
  const CAP = 60;
  function capped(gids, card) {
    const key = `cap:${S.step}`;
    const all = S.local[key] || gids.length <= CAP;
    const shown = all ? gids : gids.slice(0, CAP);
    return el("div", {}, el("div", { class: "items" }, shown.map(card)),
      all ? null : el("button", { class: "linkbtn", type: "button", style: "margin-top:8px", text: `Show all ${gids.length} (${gids.length - CAP} more)`,
        onclick: () => { S.local[key] = true; renderGuide(); } }));
  }
  async function structOp(path, body, label) {
    let ok = false;
    await busy(label, async () => {
      try {
        const r = await api(path, { session_id: S.session.session_id, ...body });
        if (r.session) S.session = r.session;
        if (r.draft) S.draft = r.draft;
        S.local = { items: S.local.items && Object.fromEntries(Object.entries(S.local.items).filter(([g]) => S.draft.groups[g])) };
        ok = true;
      } catch (e) { S.local.error = e.message; }
    });
    renderAll();
    return ok;
  }
  function groupName(gid) {
    const g = G(gid), it = D(gid);
    return `${it.label || g.columns[0]}${g.n_columns > 1 ? ` (${g.n_columns} columns)` : ""}`;
  }
  function structureTools(gid) {
    const g = G(gid);
    if (!g) return null;
    const open = S.local.struct === gid;
    if (!open) return el("button", { class: "linkbtn", type: "button", style: "margin-left:12px",
      text: g.n_columns > 1 ? "Same thing as… / take columns out" : "Same thing as another group…",
      onclick: () => { S.local.struct = gid; renderGuide(); } });
    const others = S.session.groups.filter((x) => x.group_id !== gid).map((x) => x.group_id);
    const near = focusGroups(S.step).filter((x) => x !== gid);
    const opts = [...near, ...others.filter((x) => !near.includes(x))];
    const sel = S.local.mergeWith && opts.includes(S.local.mergeWith) ? S.local.mergeWith : null;
    const pick = select(opts, sel, (v) => { S.local.mergeWith = v; renderGuide(); },
      { placeholder: "choose the group it is the same as…", labels: groupName });
    const box = el("div", { class: "struct" },
      el("div", { class: "disagree-head" }, el("strong", { text: "These are actually the same thing" })),
      fieldBox("Same as", pick),
      sel && aiReady() ? el("button", { class: "btn btn-sm", type: "button", text: "Ask the AI to check", onclick: () => {
        S.chatDraft = "Are these two groups the same thing (one family)?"; openChat([...G(gid).columns.slice(0, 100), ...G(sel).columns.slice(0, 100)]);
      } }) : null,
      sel ? el("button", { class: "btn btn-sm btn-light", type: "button", style: "margin-left:6px", text: "Merge them",
        onclick: () => structOp("/api/merge", { group_ids: [gid, sel] }, "Merging") }) : null);
    if (g.n_columns > 1) {
      S.local.take = S.local.take || {};
      box.append(el("details", { class: "saw", open: g.n_columns <= 30 },
        el("summary", { text: `Columns in this group (${g.n_columns}) — tick the ones that don't belong` }),
        el("div", { class: "colpick" }, g.columns.slice(0, 400).map((c) => {
          const cb = el("input", { type: "checkbox", checked: !!S.local.take[c] });
          cb.addEventListener("change", () => { S.local.take[c] = cb.checked; });
          return el("label", { class: "check" }, cb, c);
        })),
        el("button", { class: "btn btn-sm", type: "button", style: "margin:8px 12px", text: "Take ticked columns out",
          onclick: () => {
            const cols = Object.keys(S.local.take).filter((c) => S.local.take[c] && g.columns.includes(c));
            if (!cols.length) { S.local.error = "Tick at least one column."; renderGuide(); return; }
            structOp("/api/split", { group_id: gid, columns: cols }, "Taking columns out");
          } })));
    }
    box.append(el("button", { class: "linkbtn", type: "button", text: "close", onclick: () => { S.local.struct = null; renderGuide(); } }));
    return box;
  }
  function groupingTrouble() {
    const d = S.draft, gr = d.grouping || {}, left = d.ai_ungrouped || [];
    if (!(left.length || gr.consolidation_error) || !aiReady()) return null;
    return el("div", { class: "notice accent" },
      left.length ? el("div", {}, `The AI could not answer for ${left.length} column(s) (${(gr.failed_chunks || []).length} of ${gr.chunks} chunk(s)); `
        + "they are unresolved single columns. ",
        el("button", { class: "linkbtn", type: "button", text: "Ask the AI again for these columns",
          onclick: async () => {
            await busy("Asking the AI to group the remaining columns", async (pid) => {
              try {
                const r = await api("/api/reconsider", { session_id: S.session.session_id, group_ids: left, progress_id: pid,
                  user_hint: "The AI could not answer for these columns earlier: group and label them." });
                S.draft = r.draft; if (r.session) S.session = r.session; S.local = {};
              } catch (e) { S.local.error = e.message; }
            });
            if (!S.local.error && gr.chunks > 1) await structOp("/api/consolidate", {}, "Asking the AI for the final grouping");
            else renderAll();
          } })) : null,
      gr.consolidation_error ? el("div", {}, "The AI's final answer for the whole file failed (" + gr.consolidation_error.slice(0, 120)
        + "). Chunk proposals are drafts, so their labels and assays are not used. ",
        el("button", { class: "linkbtn", type: "button", text: "Retry the final answer",
          onclick: () => structOp("/api/consolidate", {}, "Asking the AI for the final grouping") })) : null);
  }
  // ------------------------------------------------------------ question queue (v2.4 §5)
  async function answerQuestion(q, optionIds) {
    await structOp("/api/question/answer", { question_id: q.question_id, option_ids: optionIds }, "Applying your answer");
  }
  async function askChat(text) {
    await busy("The AI is reading your message", async (pid) => {
      try {
        const r = await api("/api/chat", { session_id: S.session.session_id, message: text, step: S.step, progress_id: pid });
        S.draft = r.draft; if (r.session) S.session = r.session;
      } catch (e) { S.local.error = e.message; }
    });
    renderAll();
  }
  function questionCard(q) {
    if (q.status !== "open") {
      const a = q.answer || {};
      return el("div", { class: "question done" }, q.status === "answered" ? "✓ " : "— ", el("span", { text: q.text }), " ",
        el("b", { text: q.status === "answered" ? (a.labels || []).join("; ") : "dismissed" }), " ",
        a.edit_id ? el("button", { class: "linkbtn", type: "button", text: "Undo", onclick: () => undoTo(a.edit_id) }) : null);
    }
    S.local.qpick = S.local.qpick || {};
    const pick = (S.local.qpick[q.question_id] = S.local.qpick[q.question_id] || {});
    const optBtn = (o) => {
      const n = o.n_columns ? ` (${o.n_columns} column${o.n_columns > 1 ? "s" : ""})` : "";
      if (o.chat) return el("button", { class: "btn btn-sm", type: "button", text: o.label, disabled: !aiReady(), onclick: () => askChat(o.chat) });
      if (q.type === "multi") {
        const cb = el("input", { type: "checkbox", checked: !!pick[o.option_id], disabled: !!(o.problems || []).length });
        cb.addEventListener("change", () => { pick[o.option_id] = cb.checked; });
        return el("label", { class: "check" }, cb, o.label + n, (o.problems || []).length ? el("span", { class: "item-sub warn-text", text: ` — not possible: ${o.problems.join(" ")}` }) : null);
      }
      return el("button", { class: "btn btn-sm btn-light", type: "button", text: o.label + n, disabled: !!(o.problems || []).length,
        title: (o.problems || []).join(" "), onclick: () => answerQuestion(q, [o.option_id]) });
    };
    const other = q.allow_free_text && aiReady() ? (() => {
      const inp = textInput("", (v) => { pick._text = v; }, { placeholder: "Other… (sent to the AI chat)", cls: "grow" });
      return el("div", { class: "reconsider" }, inp, el("button", { class: "btn btn-sm", type: "button", text: "Send",
        onclick: () => pick._text && askChat(`About the question “${q.text}”: ${pick._text}`) }));
    })() : null;
    return el("div", { class: "question" },
      el("div", { class: "question-head" }, el("span", { class: `badge ${q.source === "ai" ? "p-ai" : "p-computed"}`, text: q.source === "ai" ? "AI asks" : "Computed check" }),
        el("b", { text: " " + q.text })),
      (q.evidence || []).length ? el("details", {}, el("summary", { text: "Evidence (computed)" }), el("ul", { class: "np" }, q.evidence.map((e) => el("li", { text: e })))) : null,
      el("div", { class: `question-opts ${q.type === "multi" ? "multi" : ""}` }, q.options.map(optBtn)),
      q.type === "multi" ? el("button", { class: "btn btn-sm btn-light", type: "button", text: "Answer",
        onclick: () => { const ids = q.options.filter((o) => pick[o.option_id]).map((o) => o.option_id); if (ids.length) answerQuestion(q, ids); else { S.local.error = "Tick at least one option."; renderGuide(); } } }) : null,
      other,
      el("button", { class: "linkbtn", type: "button", text: "Dismiss this question", onclick: () => structOp("/api/question/dismiss", { question_id: q.question_id }, "Recording") }));
  }
  function stepQuestions(step) {
    const all = (S.draft.questions || []).filter((q) => (q.step || "review") === step);
    const open = all.filter((q) => q.status === "open"), done = all.filter((q) => q.status !== "open");
    if (!all.length) return null;
    return el("div", { class: "questions" }, open.map(questionCard), done.length ? el("details", {}, el("summary", { text: `${done.length} answered here` }), done.map(questionCard)) : null);
  }
  function questionTray() {
    const open = (S.draft.questions || []).filter((q) => q.status === "open");
    if (!open.length) return null;
    const on = !!S.local.trayOpen;
    const label = (st) => (STEPS.find(([id]) => id === st) || [st, st])[1];
    return el("div", { class: "qtray" },
      el("button", { class: "linkbtn", type: "button", onclick: () => { S.local.trayOpen = !on; renderGuide(); } },
        el("span", { class: "qbadge", text: String(open.length) }), ` open question${open.length > 1 ? "s" : ""} — finishing waits for them`),
      on ? el("ul", { class: "np" }, open.map((q) => el("li", {}, el("span", { class: "item-sub", text: `${label(q.step || "review")}: ` }), q.text, " ",
        q.step !== S.step ? el("button", { class: "linkbtn", type: "button", text: "go there", onclick: () => go(q.step || "review") }) : null))) : null);
  }
  // ------------------------------------------------------------ AI patches: diff cards (v2.4 §4.8)
  const fmtVal = (v) => v == null ? "—" : Array.isArray(v) ? (v.map((x) => x === "" ? "(empty)" : x).join(", ") || "none") : typeof v === "boolean" ? (v ? "yes" : "no") : String(v) || "—";
  async function applyPatches(ids, confirmLarge = [], overrides = {}) {
    let res = null;
    const before = new Set(Object.keys(S.draft.groups));
    await busy("Applying", async () => {
      try {
        res = await api("/api/patch/apply", { session_id: S.session.session_id, patch_ids: ids, confirm_large: confirmLarge, overrides });
        S.draft = res.draft; if (res.session) S.session = res.session;
      } catch (e) { S.local.error = e.message; }
    });
    if (res) {
      const bad = res.results.filter((r) => r.status !== "applied");
      S.local.note = `Applied ${res.results.length - bad.length} change(s)` + (bad.length ? `; ${bad.map((r) => `${r.status}: ${r.reason}`).join("; ")}` : "") + ".";
      S.hop = new Set(res.results.filter((r) => r.status === "applied").flatMap((r) => {
        const p = (S.draft.patches || []).find((x) => x.patch_id === r.patch_id);
        const cols = new Set(p?.resolved?.columns || []);
        return S.session.groups.filter((g) => g.columns.some((c) => cols.has(c)) || !before.has(g.group_id)).map((g) => g.group_id);
      }));
    }
    S.local.patchEdits = {};
    renderAll();
  }
  async function dismissPatches(ids) { await structOp("/api/patch/dismiss", { patch_ids: ids }, "Dismissing"); }
  function patchCard(p, multi) {
    const n = p.n_columns || p.resolved?.n_columns || 0, ns = p.resolved?.n_samples || 0;
    const count = n ? `${n} column${n > 1 ? "s" : ""}` : ns ? `${ns} sample${ns > 1 ? "s" : ""}` : "";
    if (p.status === "applied") return el("div", { class: "patch done" }, "✓ ", el("b", { text: p.op_text }), count ? ` · ${count}` : "", " ",
      el("button", { class: "linkbtn", type: "button", text: "Undo", onclick: () => undoTo(p.edit_id) }));
    if (p.status === "dismissed") return null;
    if (p.status === "rejected") return el("div", { class: "patch rejected" }, el("b", { text: `${p.op_text}: not possible` }), " — ", (p.problems || []).join(" "),
      p.reason ? el("div", { class: "item-sub", text: `The AI wanted: ${p.reason}` }) : null);
    S.local.patchEdits = S.local.patchEdits || {};
    S.local.pick = S.local.pick || {};
    if (!(p.patch_id in S.local.pick)) S.local.pick[p.patch_id] = true;
    const pv = p.preview || {};
    const cols = p.resolved?.columns || [];
    const args = p.args || {};
    const edit = S.local.patchEdits[p.patch_id];
    const editable = "label" in args && args.label != null
      ? fieldBox("Label (you can edit it before applying)", textInput(edit?.label ?? args.label, (v) => { S.local.patchEdits[p.patch_id] = { ...args, label: v }; })) : null;
    const rows = (pv.groups || []).map((g) => el("li", {}, code(g.columns[0] + (g.n_columns > 1 ? ` … (${g.n_columns})` : "")), " ",
      Object.entries(g.changes).map(([k, [b, a]]) => el("span", { class: "chg" }, `${pretty(k)}: `, el("s", { text: fmtVal(b) }), " → ", el("b", { text: fmtVal(a) }))))).concat(
      (pv.metadata || []).map((m) => el("li", {}, code(m.column), " (metadata) ", Object.entries(m.changes).map(([k, [b, a]]) => el("span", { class: "chg" }, `${pretty(k)}: `, el("s", { text: fmtVal(b) }), " → ", el("b", { text: fmtVal(a) }))))));
    const der = p.derivation;
    const cb = multi ? el("input", { type: "checkbox", checked: !!S.local.pick[p.patch_id] }) : null;
    if (cb) cb.addEventListener("change", () => { S.local.pick[p.patch_id] = cb.checked; });
    const large = n > 25;
    return el("div", { class: `patch ${p.status === "held" ? "held" : ""}` },
      el("div", { class: "patch-head" }, cb, el("b", { text: p.op_text }), count ? el("span", { class: "badge", text: count }) : null,
        p.source === "question" ? el("span", { class: "item-sub", text: " from a question" }) : null),
      p.reason ? el("div", { class: "item-sub", text: p.reason }) : null,
      cols.length ? el("details", { class: "saw" }, el("summary", { text: `Which columns (${cols.length})` }), el("div", { class: "ids", text: cols.join(", ") })) : null,
      rows.length ? el("ul", { class: "np" }, rows, (pv.n_groups_changed || 0) > rows.length ? el("li", { class: "item-sub", text: `… and ${pv.n_groups_changed - rows.length} more group(s)` }) : null) : null,
      pv.assays ? el("div", { class: "item-sub" }, "Assays: ", el("s", { text: pv.assays[0].join(", ") }), " → ", el("b", { text: pv.assays[1].join(", ") })) : null,
      pv.n_groups_after !== pv.n_groups_before && pv.n_groups_after != null ? el("div", { class: "item-sub", text: `Groups: ${pv.n_groups_before} → ${pv.n_groups_after}` }) : null,
      der ? el("div", { class: "item-sub" }, `Parsed ${der.n_parsed} of ${der.n_total} names. `,
        Object.entries(der.parts).map(([k, v]) => el("div", {}, code(k), `: ${v.n_distinct} distinct — `, v.values.slice(0, 12).map((x) => `${x.label} (${x.count})`).join(", ")))) : null,
      editable,
      (p.warnings || []).length ? el("ul", { class: "warns" }, p.warnings.map((w) => el("li", { text: w }))) : null,
      p.status === "held" ? el("div", { class: "notice accent", text: (p.problems || []).join(" ") }) : null,
      (p.consequences || []).length ? el("div", { class: "notice" }, el("b", { text: "Worth knowing: " }), p.consequences.join(" ")) : null,
      el("div", { class: "disagree-actions", style: "justify-content:flex-start" },
        el("button", { class: "btn btn-sm btn-light", type: "button", text: count ? `Apply to ${count}` : "Apply",
          onclick: () => applyPatches([p.patch_id], large ? [p.patch_id] : [], S.local.patchEdits[p.patch_id] ? { [p.patch_id]: S.local.patchEdits[p.patch_id] } : {}) }),
        el("button", { class: "btn btn-sm", type: "button", style: "margin-left:6px", text: "Dismiss", onclick: () => dismissPatches([p.patch_id]) })));
  }
  // ------------------------------------------------------------ chat drawer (v2.4 §4.1): the one way to talk to the AI
  const EXAMPLES = ["Exclude all annotation columns except the ID", "The animal is the subject; the number after the underscore is time",
                    "Are these two columns the same?"];
  function openChat(sel) {
    S.chatOpen = true;
    if (sel) S.chatSel = [...new Set([...(S.chatSel || []), ...sel])];
    renderChat();
    setTimeout(() => $("chat-drawer").querySelector("textarea")?.focus(), 30);
  }
  function closeChat() { S.chatOpen = false; renderChat(); }
  async function sendChat(text) {
    if (!text.trim()) return;
    await busy("The AI is reading your message", async (pid) => {
      try {
        const r = await api("/api/chat", { session_id: S.session.session_id, message: text, step: S.step, selection: S.chatSel || [], progress_id: pid });
        S.draft = r.draft; if (r.session) S.session = r.session; S.chatDraft = ""; S.chatSel = [];
      } catch (e) { S.chatError = e.message; }
    });
    renderAll();
    const box = $("chat-drawer").querySelector(".chat-msgs");
    if (box) box.scrollTop = box.scrollHeight;
  }
  function renderChat() {
    const dr = $("chat-drawer"), fab = $("chat-fab");
    if (!S.draft) { show(dr, false); show(fab, false); return; }
    show(fab, !S.chatOpen); show(dr, !!S.chatOpen);
    if (!S.chatOpen) return;
    const byP = Object.fromEntries((S.draft.patches || []).map((p) => [p.patch_id, p]));
    const byQ = Object.fromEntries((S.draft.questions || []).map((q) => [q.question_id, q]));
    const msgs = S.draft.chat || [];
    const stepName = (STEPS.find(([id]) => id === S.step) || ["", ""])[1];
    const auto = el("input", { type: "checkbox", checked: !!S.draft.settings?.auto_apply });
    auto.addEventListener("change", () => structOp("/api/settings", { auto_apply: auto.checked }, "Saving"));
    const ta = el("textarea", { class: "input", rows: 2, placeholder: aiReady() ? "Ask, or tell the AI what to change…" : "The AI is off", disabled: !aiReady() });
    ta.value = S.chatDraft || "";
    ta.addEventListener("input", () => { S.chatDraft = ta.value; });
    ta.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(ta.value); } });
    const sel = S.chatSel || [];
    dr.replaceChildren(...[
      el("div", { class: "chat-head" }, el("b", { text: "Talk to the AI" }), el("span", { class: "item-sub", text: ` · step ${STEPS.findIndex(([id]) => id === S.step) + 1}: ${stepName}` }),
        el("span", { class: "spacer" }), el("button", { class: "linkbtn", type: "button", text: "close ✕", onclick: closeChat })),
      aiReady() ? null : el("div", { class: "notice", text: "The AI is off or unavailable. Manual editing and all computed questions still work." }),
      el("label", { class: "check chat-auto", title: "Still logged and undoable. Large changes (over 25 columns), changes with warnings or near an invariant always wait for your click." },
        auto, "Apply what I ask for automatically"),
      el("div", { class: "chat-msgs" },
        msgs.length ? msgs.map((m) => el("div", { class: "chat-msg" },
          el("div", { class: "bubble user" }, m.text, (m.selection || []).length ? el("div", { class: "item-sub", text: `about: ${m.selection.slice(0, 6).join(", ")}${m.selection.length > 6 ? " …" : ""}` }) : null),
          el("div", { class: "bubble ai", text: m.reply || "" }),
          (m.auto_applied || []).length ? el("div", { class: "item-sub", text: `Applied automatically: ${m.auto_applied.length} change(s) (undo below or in Changes).` }) : null,
          (m.patch_ids || []).map((id) => byP[id] && patchCard(byP[id], (m.patch_ids || []).filter((x) => ["pending", "held"].includes(byP[x]?.status)).length > 1)),
          (m.question_ids || []).map((id) => byQ[id] && questionCard(byQ[id])),
          (() => {
            const pend = (m.patch_ids || []).map((id) => byP[id]).filter((p) => p && ["pending", "held"].includes(p.status));
            return pend.length > 1 ? el("button", { class: "btn btn-sm btn-light", type: "button", text: "Apply all ticked",
              onclick: () => applyPatches(pend.filter((p) => S.local.pick?.[p.patch_id] !== false).map((p) => p.patch_id), [], S.local.patchEdits || {}) }) : null;
          })()))
          : el("div", { class: "chat-empty" }, el("p", { class: "item-sub", text: "The AI can only propose changes; you apply them. Try:" }),
            EXAMPLES.map((x) => el("button", { class: "linkbtn", type: "button", text: `“${x}”`, onclick: () => { S.chatDraft = x; renderChat(); } })))),
      S.chatError ? el("div", { class: "alert alert-error", text: S.chatError }) : null,
      sel.length ? el("div", { class: "chat-sel" }, el("span", { class: "item-sub", text: "About: " }),
        sel.slice(0, 12).map((c) => el("span", { class: "pill" }, c, el("button", { class: "linkbtn", type: "button", text: " ✕",
          onclick: () => { S.chatSel = sel.filter((x) => x !== c); renderChat(); } }))), sel.length > 12 ? ` +${sel.length - 12}` : "") : 
        el("div", { class: "item-sub chat-hint", text: "Tip: click a column group in the table to ask about it." }),
      el("div", { class: "command-row" }, ta, el("button", { class: "btn btn-sm", type: "button", text: "Send", disabled: !aiReady(), onclick: () => sendChat(ta.value) }))].filter(Boolean));
    S.chatError = null;
    const box = dr.querySelector(".chat-msgs");
    if (box) box.scrollTop = box.scrollHeight;
  }
  $("chat-fab").addEventListener("click", () => openChat());

  // ------------------------------------------------------------ changes + undo (one edit layer, v2.4)
  async function undoTo(editId) {
    await structOp("/api/undo", editId ? { edit_id: editId } : {}, "Undoing");
  }
  function changesBox() {
    const ch = S.draft.changes || [];
    if (!ch.length || S.finalized) return null;
    const who = { user: "you", ai_patch: "AI patch", question_option: "answer" };
    const open = !!S.local.changesOpen;
    return el("div", { class: "changes" },
      el("div", { class: "changes-head" },
        el("button", { class: "linkbtn", type: "button", text: `Undo: ${ch[0].summary}`, title: "Undo the latest change", onclick: () => undoTo(null) }),
        el("button", { class: "linkbtn", type: "button", style: "margin-left:12px", text: open ? "hide changes" : `Changes (${ch.length})`,
          onclick: () => { S.local.changesOpen = !open; renderGuide(); } })),
      open ? el("ol", { class: "changes-list" }, ch.map((c, k) => el("li", {},
        el("span", { class: "item-sub", text: `${c.at.slice(11, 19)} · ${who[c.actor] || c.actor}` }), " ", c.summary,
        c.n_columns ? el("span", { class: "item-sub", text: ` (${c.n_columns} column${c.n_columns > 1 ? "s" : ""})` }) : null, " ",
        el("button", { class: "linkbtn", type: "button", text: k ? `undo this and ${k} later` : "undo",
          onclick: () => undoTo(c.edit_id) })))) : null);
  }
  function ledgerTable() {
    const led = S.draft.column_ledger || {};
    const cats = ["value", "feature_id", "annotation", "sample_id", "sample_metadata", "excluded", "unresolved", "unaccounted"];
    return el("table", { class: "ledger" },
      el("thead", {}, el("tr", {}, el("th", { text: "file" }), el("th", { text: "total" }), cats.map((c) => el("th", { text: pretty(c) })))),
      el("tbody", {}, Object.entries(led).map(([f, x]) => el("tr", { class: x.unaccounted || x.unresolved ? "bad" : "" },
        el("td", { text: f }), el("td", { text: x.total }), cats.map((c) => el("td", { text: x[c] || 0 }))))));
  }

  function renderNameParts() {
    const parts = S.draft?.name_parts || [];
    const box = $("name-parts");
    if (!box) return;
    box.classList.toggle("hidden", !parts.length);
    box.open = S.draft?.grouping?.source === "manual" || box.open;
    $("name-parts-list").replaceChildren(...parts.map((p) => el("li", {},
      el("code", { text: p.side === "prefix" ? `${p.text}…` : `…${p.text}` }),
      el("span", { class: "item-sub", text: ` ${p.n} columns, now in ${p.n_groups} groups ` }),
      el("button", { class: "linkbtn", type: "button", text: `group these ${p.n}`,
        onclick: () => structOp("/api/group-columns", { columns: p.columns, reason: `columns sharing '${p.text}'` }, "Grouping columns") }))));
  }

  // ------------------------------------------------------------ steps
  const LAYOUT_TEXT = {
    samples_in_columns: ["Samples in columns", "Each row looks like one feature, and each sample has its own column."],
    samples_in_rows: ["Samples in rows", "Each row looks like one sample, and each feature has its own column."],
    long: ["Long table", "Each row looks like one measurement: a feature, a sample and a single value."],
  };
  function layoutIcon(kind) {
    const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg");
    svg.setAttribute("width", "44"); svg.setAttribute("height", "30"); svg.setAttribute("viewBox", "0 0 44 30");
    const rect = (x, y, w, h, f) => { const r = document.createElementNS(ns, "rect"); Object.entries({ x, y, width: w, height: h, fill: f, rx: 1 }).forEach(([k, v]) => r.setAttribute(k, v)); svg.append(r); };
    if (kind === "samples_in_columns") { rect(0, 0, 12, 30, "#fff"); for (let c = 0; c < 4; c++) rect(15 + c * 7.5, 0, 6, 30, "#ff8a2a"); }
    else if (kind === "samples_in_rows") { for (let r = 0; r < 5; r++) { rect(0, r * 6.2, 10, 5, "#d9c7a3"); rect(13, r * 6.2, 31, 5, "#ff8a2a"); } }
    else { for (let r = 0; r < 5; r++) { rect(0, r * 6.2, 14, 5, "#fff"); rect(16, r * 6.2, 14, 5, "#d9c7a3"); rect(32, r * 6.2, 12, 5, "#ff8a2a"); } }
    return svg;
  }
  function scopeNotice(a) {
    if (!a || !["no", "unsure"].includes(a.in_supported_scope)) return null;
    return el("div", { class: "notice accent" }, `Recognized as ${a.omics_type || "unknown"} — not yet supported by PRISM's audit`,
      a.in_supported_scope === "unsure" ? " (the AI is not sure)" : "", ". This is a notice, not a block: you can finish this step and download the files.",
      a.scope_reason ? el("div", { class: "item-sub", text: a.scope_reason }) : null);
  }
  function assayCard(a, k) {
    S.local.assays = S.local.assays || {};
    const l = (S.local.assays[k] = S.local.assays[k] || {});
    const v = (f) => (f in l ? l[f] : a[f]);
    const set = (f) => (x) => { l[f] = x; };
    const nBlocks = Object.values(S.draft.groups).filter((it) => it.role === "value" && it.assay_label === a.assay_label).length;
    return el("div", { class: "block-card" },
      el("div", { class: "item-head" }, el("h4", { text: `Assay ${k + 1}` }, el("span", { class: "item-sub", text: `  ${nBlocks} value block(s)` })), metaRow(a)),
      el("div", { class: "item-controls" },
        fieldBox("Assay label", textInput(v("assay_label"), set("assay_label"), { list: "dl-assay" })),
        fieldBox("Omics type (your words)", textInput(v("omics_type") === "unknown" ? "" : v("omics_type"), set("omics_type"), { placeholder: "e.g. proteomics, 16S microbiome", list: "dl-omics" })),
        fieldBox("Source software", textInput(v("source_software") === "unknown" ? "" : v("source_software"), set("source_software"), { placeholder: "unknown", list: "dl-software" })),
        fieldBox("In PRISM's supported scope?", select(S.vocab.in_supported_scope, v("in_supported_scope"), (x) => { l.in_supported_scope = x; renderGuide(); }))),
      scopeNotice({ ...a, ...l }), evidence(a.evidence));
  }
  function stepLayout() {
    const d = S.draft, lay = S.local.layout ?? d.layout.value, known = LAYOUT_TEXT[lay];
    const body = [];
    if (d.signature_hint) body.push(el("div", { class: "notice", text: `Known format: ${d.signature_hint}. This is only a hint${d.ai.used ? " given to the AI" : "; it pre-filled the proposal below"}.` }));
    body.push(el("div", { class: "opt-cards" }, Object.entries(LAYOUT_TEXT).map(([k, [t, q]]) => el("button", {
      type: "button", class: `opt-card ${lay === k ? "sel" : ""}`, onclick: () => { S.local.layout = k; renderGuide(); } },
      layoutIcon(k), el("strong", { text: t }), q))));
    if (S.local.layout && S.local.layout !== d.layout.value) body.push(el("div", { class: "notice", text: "Changing the layout re-proposes every column: steps you confirmed are asked again." }));
    body.push(el("div", { class: "meta-row" }, metaRow(d.layout)), evidence(d.layout.evidence), warnList(d.layout.validation));
    if (d.layout.hint) body.push(el("div", { class: "item-sub", text: "From the data: " + d.layout.hint }));
    body.push(el("div", { class: "section-label", text: d.assays.length > 1 ? `${d.assays.length} assays in this file` : "What was measured" }));
    body.push(el("div", {}, d.assays.map(assayCard)));
    if (S.scope) body.push(el("details", { class: "saw" }, el("summary", { text: "What PRISM currently supports" }), el("p", { class: "item-sub", text: S.scope })));
    return {
      title: known ? `${known[0]}: is that right?` : "How is this table organised?",
      question: known ? "Check the layout and describe what was measured, in your own words. Everything here is editable." : "Pick the layout that matches your file. Features usually outnumber samples.",
      body,
      decision: () => {
        if (!LAYOUT_TEXT[lay]) throw new Error("Choose one of the three layouts.");
        const assays = d.assays.map((a, k) => {
          const l = S.local.assays?.[k] || {};
          const x = { assay_label: l.assay_label ?? a.assay_label, omics_type: l.omics_type ?? a.omics_type,
                      source_software: l.source_software ?? a.source_software, in_supported_scope: l.in_supported_scope ?? a.in_supported_scope,
                      scope_reason: a.scope_reason };
          if (!String(x.assay_label || "").trim()) throw new Error(`Give assay ${k + 1} a label.`);
          return x;
        });
        return { layout: lay, assays };
      },
    };
  }

  function candidateIdGroups() {
    return S.session.groups.filter((g) => g.n_columns === 1 && D(g.group_id).role !== "value");
  }
  function stepFeatureId() {
    const d = S.draft, lay = layout();
    if (lay === "samples_in_rows") {
      const blocks = Object.keys(d.groups).filter((g) => D(g).role === "value");
      const names = blocks.flatMap((b) => G(b).columns);
      return { title: "Feature names are the column headers of the measurement block.",
        question: el("span", {}, `${names.length} features, e.g. `, code(names.slice(0, 3).join(", ")), ". Nothing to choose here; confirm to continue."),
        body: [], decision: () => ({}) };
    }
    const sel = S.local.fid ?? d.feature_identity.group_ids;
    const fi = d.feature_identity;
    const labels = sel.map((g) => G(g)?.columns[0]).filter(Boolean);
    const body = [el("div", { class: "meta-row" }, metaRow(fi)), evidence(fi.evidence), warnList(fi.validation),
      el("div", { class: "section-label", text: "Tick one column, or several for a composite key (e.g. m/z + RT)" })];
    body.push(el("div", { class: "items" }, candidateIdGroups().map((g) => {
      const cb = el("input", { type: "checkbox", checked: sel.includes(g.group_id) });
      cb.addEventListener("change", () => { const s = new Set(S.local.fid ?? d.feature_identity.group_ids); cb.checked ? s.add(g.group_id) : s.delete(g.group_id); S.local.fid = [...s]; renderGuide(); });
      const shape = (g.profile?.value_shapes || [])[0];
      return el("label", { class: `item pick ${sel.includes(g.group_id) ? "on" : ""}` }, cb,
        el("span", {}, el("span", { class: "item-name", text: g.columns[0] }), D(g.group_id).label ? el("span", { class: "item-sub", text: ` · ${D(g.group_id).label}` }) : null,
          el("div", { class: "item-sub", text: (D(g.group_id).hint || "") + (shape ? ` Format: ${shape.shape}` : "") })));
    })));
    const sid = S.local.sid ?? d.sample_id_group.value;
    if (lay === "long") {
      body.push(fieldBox("Column naming the sample on each row", select(candidateIdGroups().map((g) => g.group_id), sid,
        (v) => { S.local.sid = v; renderGuide(); }, { placeholder: "choose…", labels: (g) => G(g).columns[0] })));
      if (d.long_duplicates) body.push(el("ul", { class: "warns" }, el("li", { class: "bad", text: d.long_duplicates.message })));
    }
    return {
      title: sel.length ? (sel.length > 1 ? "These columns together identify each feature. Confirm?" : "This column uniquely identifies each feature. Confirm?") : "Which column identifies each feature?",
      question: sel.length ? el("span", {}, labels.map((l, k) => [k ? " + " : "", code(l)]), ". Duplicates or empty IDs are only reported; nothing is merged or removed.") : "Tick the identifier column(s).",
      body,
      decision: () => {
        if (!sel.length) throw new Error("Choose at least one column.");
        const dec = { feature_identity: { group_ids: sel } };
        if (lay === "long") { if (!sid) throw new Error("Choose the sample column."); dec.sample_id_group = sid; }
        return dec;
      },
    };
  }

  function stepAnnotations() {
    const gids = focusGroups("annotations");
    const body = [];
    if (!gids.length) body.push(el("p", { class: "q", text: "No annotation columns in this file." }));
    const flagged = gids.filter((g) => D(g).marks_rows_as_suspect);
    if (flagged.length) body.push(el("div", { class: "notice", text: `${flagged.length} column(s) mark rows as suspect. Choose which value means ‘flagged’; nothing is removed now.` }));
    body.push(capped(gids, (g) => itemCard(g, { roles: ["feature_annotation", "feature_id", "sample_metadata", "value", "ignore"] })));
    return {
      title: `${gids.length} column(s) describe the features. Are these right?`,
      question: "Each shows the AI's description in plain words. Edit any label, role or flag directly; everything is kept unless you untick it.",
      body,
      decision: () => { checkItems(gids); return { items: itemDecisions(gids) }; },
    };
  }

  function drawHist(canvas, h) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width = canvas.clientWidth * devicePixelRatio, hh = canvas.height = 64 * devicePixelRatio;
    ctx.clearRect(0, 0, w, hh);
    if (!h || !h.bins || !h.bins.length) return;
    const max = Math.max(...h.bins, 1), bw = w / h.bins.length;
    ctx.fillStyle = "#ff8a2a";
    h.bins.forEach((c, k) => { const bh = (c / max) * (hh - 14 * devicePixelRatio); ctx.fillRect(k * bw + 1, hh - 12 * devicePixelRatio - bh, bw - 2, bh); });
    ctx.fillStyle = "#777"; ctx.font = `${10 * devicePixelRatio}px monospace`;
    ctx.fillText(`1e${h.lo.toFixed(1)}`, 2, hh - 2);
    const t = `1e${h.hi.toFixed(1)}`; ctx.fillText(t, w - ctx.measureText(t).width - 2, hh - 2);
    if (h.n_zero) { const z = `${h.n_zero} zeros`; ctx.fillText(z, (w - ctx.measureText(z).width) / 2, hh - 2); }
  }

  function valueCard(gid) {
    const it = D(gid), g = G(gid), p = g.profile || {}, lay = layout();
    const role = cur(gid, "role"), keep = cur(gid, "keep") !== false;
    const n = g.n_columns;
    const voice = lay === "samples_in_rows" ? `${n} columns: ${n} features measured in every sample.` : `${n} columns: one measurement per sample.`;
    const keepCb = el("input", { type: "checkbox", checked: keep && role === "value" });
    keepCb.addEventListener("change", () => { local(gid).keep = keepCb.checked; if (keepCb.checked && cur(gid, "role") !== "value") local(gid).role = "value"; renderGuide(); });
    const canvas = el("canvas", { class: "hist" });
    setTimeout(() => drawHist(canvas, g.histogram), 0);
    const samples = lay === "samples_in_columns" ? (S.draft.blocks_samples?.[gid] || []) : [];
    const computed = el("div", { class: "computed" },
      el("div", { class: "section-label", text: "Computed from the data" }),
      canvas,
      el("div", { class: "stats" }, [["min", p.min], ["p1", p.p1], ["median", p.median], ["p99", p.p99], ["max", p.max],
        ["zeros", pct(p.frac_zero)], ["missing", pct(p.frac_na)], ["whole numbers", p.integer_valued ? "yes" : "no"], ["span", p.log10_span != null ? `${fmtNum(p.log10_span)} decades` : "—"]]
        .map(([k, v]) => el("div", {}, el("span", { text: k }), el("b", { text: typeof v === "number" ? fmtNum(v) : v })))),
      samples.length ? el("div", { class: "item-sub", style: "margin-top:6px" }, `Samples: ${samples.slice(0, 6).join(", ")}${samples.length > 6 ? ` … (${samples.length})` : ""}`) : null);
    const described = el("div", { class: "described" },
      el("div", { class: "section-label", text: "Described by " + (it.source === "ai" ? "the AI" : it.source === "computed" ? "the known format" : "you") }),
      fieldBox("What these values are (your words)", textInput(cur(gid, "label"), (v) => { local(gid).label = v; }, { placeholder: "e.g. LFQ intensity, apparently linear", list: "dl-label" })),
      fieldBox("Assay", textInput(cur(gid, "assay_label") || assayLabels()[0], (v) => { local(gid).assay_label = v; }, { list: "dl-assay" })),
      fieldBox("Role", roleSelect(gid)));
    return el("div", { class: `block-card ${keep && role === "value" ? "kept" : "dropped"} ${role === "unresolved" ? "unres" : ""}`, onmouseenter: () => hoverGroup(gid, true), onmouseleave: () => hoverGroup(gid, false) },
      el("div", { class: "item-head" }, el("h4", {}, cur(gid, "label") || (g.pattern ? g.pattern.text.trim() : `${n} numeric columns`),
        el("span", { class: "item-sub", text: `  ${colsText(g)}` })), metaRow(it)),
      el("div", { class: "q", style: "margin:6px 0 8px" }, voice),
      el("label", { class: "check" }, keepCb, "keep and export this block (its own matrix file)"),
      el("div", { class: "side-by-side" }, described, computed),
      evidence(it.evidence), warnList(it.validation, it.claimed),
      el("div", { class: "item-foot" }, reconsiderOne(gid), structureTools(gid)));
  }

  function stepValues() {
    const all = focusGroups("values");
    const blocks = all.filter((g) => D(g).role === "value" || (D(g).role === "unresolved" && G(g).kind === "numeric_block"));
    const others = all.filter((g) => !blocks.includes(g));
    const body = [el("div", {}, blocks.map(valueCard))];
    if (others.length) {
      body.push(el("div", { class: "section-label", text: "Other numeric columns" }));
      body.push(el("p", { class: "q", text: "Covariates and technical values (age, CD4 count, iron, scale factors…) usually describe samples. Change any of them if needed." }));
      body.push(capped(others, (g) => itemCard(g, { roles: ["sample_metadata", "value", "feature_annotation", "ignore"] })));
    }
    return {
      title: blocks.length > 1 ? `${blocks.length} measurement blocks. What is each one?` : "Is this the measurement block?",
      question: "Nothing is ranked: every block you keep is exported as its own matrix. Check each description against the computed profile. Which block to analyse is decided later, after the research-focus step.",
      body,
      decision: () => {
        checkItems(all);
        if (!all.some((g) => cur(g, "role") === "value" && cur(g, "keep") !== false)) throw new Error("Keep at least one value block.");
        return { items: itemDecisions(all) };
      },
    };
  }

  function applyRule(name, r) {
    let n = name;
    if (r.strip_prefix && n.startsWith(r.strip_prefix)) n = n.slice(r.strip_prefix.length);
    if (r.strip_suffix && n.endsWith(r.strip_suffix)) n = n.slice(0, n.length - r.strip_suffix.length);
    return n;
  }
  function idPreview(ids) {
    const counts = {}; ids.forEach((x) => { counts[x] = (counts[x] || 0) + 1; });
    const dup = Object.keys(counts).filter((k) => counts[k] > 1);
    const shown = ids.length > 13 ? [...ids.slice(0, 10), "…", ...ids.slice(-3)] : ids;
    return el("div", {}, el("div", { class: "ids" }, shown.map((x, k) => [k ? ", " : "", el("span", { class: counts[x] > 1 ? "dup" : "", text: x })])),
      el("div", { class: "item-sub", text: `${ids.length} sample IDs${ids.length > 13 ? ` (… ${ids.length - 13} more)` : ""}${dup.length ? ` · ${dup.length} duplicated: ${dup.slice(0, 5).join(", ")}` : " · all unique"}` }));
  }
  function sampleRow(sid) {
    const st = S.draft.samples[sid];
    S.local.samples = S.local.samples || {};
    const l = S.local.samples[sid] || {};
    const v = (f) => (f in l ? l[f] : st[f]);
    const set = (f, x) => { S.local.samples[sid] = { ...(S.local.samples[sid] || {}), [f]: x }; };
    const cb = el("input", { type: "checkbox", checked: v("is_study_sample") });
    cb.addEventListener("change", () => { set("is_study_sample", cb.checked); renderGuide(); });
    return el("div", { class: `sample-row ${v("is_study_sample") ? "" : "nonstudy"}`, title: st.evidence || "" },
      el("span", { class: "sid", text: sid }),
      textInput(v("label"), (x) => set("label", x), { list: "dl-sample-label", placeholder: "label" }),
      el("label", { class: "check" }, cb, "study sample"),
      provBadge(st.provenance, st.source));
  }
  function stepSamples() {
    const d = S.draft, lay = layout(), body = [];
    if (lay === "samples_in_columns") {
      const prim = Object.keys(d.groups).filter((g) => D(g).role === "value" && D(g).keep !== false);
      S.local.rules = S.local.rules || {};
      for (const gid of prim) {
        const base = d.sample_rules[gid] || { strip_prefix: "", strip_suffix: "" };
        const r = S.local.rules[gid] || { strip_prefix: base.strip_prefix, strip_suffix: base.strip_suffix };
        const ids = G(gid).indices.map((i) => applyRule(header(i), r));
        const prev = el("div", {}, idPreview(ids));
        const mk = (k, ph) => textInput(r[k], (x) => { r[k] = x; S.local.rules[gid] = { ...r }; prev.replaceChildren(idPreview(G(gid).indices.map((i) => applyRule(header(i), r)))); }, { placeholder: ph });
        body.push(el("div", { class: "block-card" }, el("h4", { text: D(gid).label || "Primary block" }),
          el("div", { class: "row2", style: "margin-top:8px" }, fieldBox("Strip prefix", mk("strip_prefix", "none")), fieldBox("Strip suffix", mk("strip_suffix", "none"))),
          prev));
      }
    } else {
      const sid = S.local.sid ?? d.sample_id_group.value;
      body.push(fieldBox("Column identifying each sample", select(candidateIdGroups().map((g) => g.group_id), sid,
        (v) => { S.local.sid = v; renderGuide(); }, { placeholder: "choose…", labels: (g) => G(g).columns[0], bad: !sid })));
      if (sid === d.sample_id_group.value) body.push(idPreview(d.sample_list.ids));
      else body.push(el("div", { class: "item-sub", text: "The ID list updates after you confirm." }));
    }
    const ids = Object.keys(d.samples);
    const isStudy = (s) => (S.local.samples?.[s] && "is_study_sample" in S.local.samples[s] ? S.local.samples[s].is_study_sample : d.samples[s].is_study_sample);
    const non = ids.filter((s) => !isStudy(s));
    const byLabel = {}; ids.forEach((s) => { const lab = S.local.samples?.[s]?.label ?? d.samples[s].label; byLabel[lab] = (byLabel[lab] || 0) + 1; });
    body.push(el("div", { class: "section-label", text: "What each sample is" }),
      el("div", { class: "item-sub", style: "margin-bottom:8px", text: `${ids.length - non.length} study sample(s), ${non.length} other (QC, blank, pool…). Labels: ${Object.entries(byLabel).map(([k, v]) => `${v} ${k || "(no label)"}`).join(" · ")}. Nothing is dropped: non-study samples are only labelled.` }));
    // list non-study samples as proposed (plus any you edited), so rows don't jump away while you edit
    const shortList = ids.filter((s) => !d.samples[s].is_study_sample || S.local.samples?.[s]);
    const list = S.local.showAll ? ids.slice(0, 600) : (shortList.length ? shortList.slice(0, 80) : ids.slice(0, 12));
    body.push(el("div", { class: "sample-list" }, list.map(sampleRow)));
    if (ids.length > list.length || S.local.showAll) body.push(el("button", { class: "linkbtn", type: "button", style: "margin-top:8px",
      text: S.local.showAll ? "Show fewer" : `Show all ${ids.length} samples`, onclick: () => { S.local.showAll = !S.local.showAll; renderGuide(); } }));
    return {
      title: `${d.sample_list.n} samples found. Are the sample IDs and labels right?`,
      question: lay === "samples_in_columns" ? "Sample IDs come from the measurement column names. Adjust what is stripped; the list updates live. Untick ‘study sample’ for QC, blanks, pools…" : "Sample IDs come from one column. Untick ‘study sample’ for QC, blanks, pools…",
      body,
      decision: () => {
        const dec = {};
        if (lay === "samples_in_columns") dec.sample_rules = S.local.rules || {};
        else { const sid = S.local.sid ?? d.sample_id_group.value; if (!sid) throw new Error("Choose the sample ID column."); dec.sample_id_group = sid; }
        if (Object.keys(S.local.samples || {}).length) dec.samples = S.local.samples;
        return dec;
      },
    };
  }

  function metadataSection() {
    const d = S.draft, meta = d.metadata, body = [];
    const fileIn = el("input", { type: "file", accept: ".csv,.tsv,.txt", class: "input" });
    fileIn.addEventListener("change", async () => {
      if (!fileIn.files[0]) return;
      const fd = new FormData(); fd.append("session_id", S.session.session_id); fd.append("file", fileIn.files[0]);
      await busy(aiReady() ? "The AI is reading the metadata file" : "Reading the metadata file", async () => {
        try { const r = await api("/api/metadata-upload", fd); S.draft = r.draft; S.local = {}; } catch (e) { S.local.error = e.message; }
      });
      renderAll();
    });
    body.push(fieldBox(meta && meta.columns ? "Replace the sample metadata file" : "Upload a sample metadata file (optional)", fileIn));
    if (!(meta && !meta.skipped && meta.columns)) return body;
    const rep = meta.report;
    S.local.near = S.local.near || {};
    const keyCols = meta.columns.map((c) => c.column);
    const jk = S.local.joinKey ?? meta.id_column;
    const fact = (c) => { const j = meta.columns.find((x) => x.column === c)?.join || {}; return `${c} — ${j.exact_matches ?? "?"}/${d.sample_list.n} exact${j.normalized_matches !== j.exact_matches ? `, ${j.normalized_matches} normalised` : ""}`; };
    body.push(el("div", { class: "section-label", text: `Metadata file: ${meta.filename} (${meta.n_rows} rows × ${meta.n_columns} columns)` }),
      fieldBox("Join key: the column that names the samples", select(keyCols, jk, (v) => { S.local.joinKey = v; renderGuide(); }, { labels: fact })),
      el("div", { class: "item-sub", text: `${meta.join_key_source === "ai" ? "Proposed by the AI" : "Computed suggestion (most exact matches)"}: ${meta.join_key_evidence || ""}` }),
      jk === meta.id_column ? el("div", { class: "notice", text: `Matched ${rep.n_matched} sample(s). Only in data: ${rep.only_in_data.length}${rep.only_in_data.length ? ` (${rep.only_in_data.slice(0, 6).join(", ")})` : ""}. Only in metadata: ${rep.only_in_metadata.length}.` })
        : el("div", { class: "item-sub", text: "The match report updates after you confirm." }),
      el("details", { class: "saw" }, el("summary", { text: "Parse report of the metadata file" }), el("pre", { text: JSON.stringify(meta.parse_report, null, 2) })));
    if (meta.ai?.error) body.push(el("div", { class: "notice accent", text: `The AI could not read this file (${meta.ai.error.slice(0, 120)}): choose the audit kinds yourself.` }));
    if (rep.near_misses.length && jk === meta.id_column) {
      body.push(el("div", { class: "section-label", text: "Near misses (suggestions only — tick to accept)" }));
      body.push(el("div", { class: "items" }, rep.near_misses.map((n) => {
        const key = `${n.data_id}|${n.metadata_id}`;
        const cb = el("input", { type: "checkbox", checked: !!S.local.near[key] });
        cb.addEventListener("change", () => { S.local.near[key] = cb.checked; });
        return el("label", { class: "item pick" }, cb, code(n.data_id), " ↔ ", code(n.metadata_id), el("span", { class: "item-sub", text: " " + n.reason }));
      })));
    }
    S.local.metaCols = S.local.metaCols || {};
    const cols = meta.columns.filter((c) => c.column !== jk);
    body.push(el("div", { class: "section-label", text: `Metadata columns (${cols.length})` }));
    body.push(capped(cols.map((c) => c.column), (name) => {
      const c = meta.columns.find((x) => x.column === name);
      const l = (S.local.metaCols[c.column] = S.local.metaCols[c.column] || {});
      const v = (f) => (f in l ? l[f] : c[f]);
      const cb = el("input", { type: "checkbox", checked: v("keep") !== false });
      cb.addEventListener("change", () => { l.keep = cb.checked; renderGuide(); });
      const needKind = v("keep") !== false && !S.vocab.audit_kind.includes(v("audit_kind"));
      return el("div", { class: `item ${needKind ? "unres" : ""}` },
        el("div", { class: "item-head" }, el("span", { class: "item-name", text: c.column }),
          el("div", { class: "meta-row" }, provBadge(c.provenance || (c.source === "ai" ? "ai_proposed_confirmed" : "user_set"), c.source), confBar(c.source === "ai" ? c.confidence : null))),
        el("div", { class: "item-controls" },
          fieldBox("Audit kind", select(S.vocab.audit_kind, v("audit_kind"), (x) => { l.audit_kind = x; renderGuide(); }, { placeholder: "what is it for the audit?", bad: needKind })),
          fieldBox("What it is (your words)", textInput(v("label"), (x) => { l.label = x; }, { list: "dl-label" })),
          fieldBox("Detail (optional)", textInput(v("detail"), (x) => { l.detail = x; })),
          el("div", { class: "full" }, el("label", { class: "check" }, cb, "keep in the outputs"))),
        el("div", { class: "item-sub", text: `${c.hint} · join: ${c.join?.exact_matches ?? 0} exact` }), evidence(c.evidence));
    }));
    return body;
  }
  function metadataDecision() {
    const meta = S.draft.metadata;
    const jk = S.local.joinKey ?? meta.id_column;
    const cols = meta.columns.filter((c) => c.column !== jk).map((c) => {
      const l = S.local.metaCols?.[c.column] || {};
      const x = { column: c.column, audit_kind: l.audit_kind ?? c.audit_kind, label: l.label ?? c.label, detail: l.detail ?? c.detail, keep: l.keep ?? c.keep };
      if (x.keep !== false && !S.vocab.audit_kind.includes(x.audit_kind)) throw new Error(`Choose an audit kind for '${c.column}'.`);
      return x;
    });
    const acc = Object.entries(S.local.near || {}).filter(([, v]) => v).map(([k]) => k.split("|"));
    return { ...(jk !== meta.id_column ? { join_key: jk } : {}), metadata: { columns: cols, accept_near_misses: acc } };
  }
  function stepSampleInfo() {
    const d = S.draft, lay = layout(), body = [];
    const hasMeta = d.metadata && !d.metadata.skipped && d.metadata.columns;
    if (lay !== "samples_in_columns") {
      const gids = focusGroups("sample_info");
      if (!gids.length) body.push(el("p", { class: "q", text: "No sample information columns besides the sample ID." }));
      body.push(capped(gids, (g) => itemCard(g, { roles: ["sample_metadata", "sample_id", "feature_annotation", "value", "ignore"] })));
      body.push(el("div", { class: "section-label", text: "A separate sample metadata file (optional)" }), ...metadataSection());
      return { title: `${gids.length} column(s) describe the samples. Are these right?`,
        question: "The audit kind says what the later steps use it for (subject, time point, batch…). Group-like variables are candidates only: PRISM never decides which variable is the research outcome.",
        body,
        decision: () => { checkItems(gids); return { items: itemDecisions(gids), ...(hasMeta ? metadataDecision() : {}) }; } };
    }
    body.push(...metadataSection());
    return {
      title: hasMeta ? "The sample metadata file: are the join key and the columns right?" : "Do you have a sample metadata file?",
      question: hasMeta ? "The AI read each column (from statistics, never raw rows) and proposed the join key from how many sample names each column contains. Edit anything; nothing is merged without you."
        : "Your table has samples in columns, so information about samples (subject, time point, batch, group…) usually lives in a separate sheet. You can also skip this; that is recorded.",
      body,
      extraActions: hasMeta ? [] : [el("button", { class: "btn btn-sm", type: "button", text: "Skip (record as missing)", onclick: () => submit({ metadata: { skip: true } }) })],
      decision: () => {
        if (!hasMeta) throw new Error("Upload a file, or use 'Skip'.");
        return metadataDecision();
      },
    };
  }

  // ------------------------------------------------------------ study design (v2.4 §7)
  const SIDE_LABEL = { subject: "Subject (the individual a sample came from)", time: "Time point" };
  function designSourceOptions(rep) {
    return ["none", "derived_from_sample_names", ...(rep.candidates || []).map((c) => `${c.file}:${c.column}`)];
  }
  function designSourceLabel(rep, v) {
    if (v === "none") return "none";
    if (v === "derived_from_sample_names") return "parsed from the sample names";
    const c = (rep.candidates || []).find((x) => `${x.file}:${x.column}` === v);
    return c ? `${c.column} (${c.file}${c.audit_kind ? ", " + pretty(c.audit_kind) : ""})` : v;
  }
  function srcKey(x) { return x.source === "metadata_column" ? `${x.file}:${x.column}` : x.source; }
  function barChart(perSubject) {
    const entries = Object.entries(perSubject || {});
    if (!entries.length) return null;
    const max = Math.max(...entries.map(([, n]) => n));
    return el("div", { class: "bars" }, entries.slice(0, 60).map(([s, n]) => el("div", { class: "bar-row" },
      el("span", { class: "bar-k", text: s }), el("span", { class: "bar", style: `width:${Math.round((n / max) * 100)}%` }), el("span", { class: "bar-n", text: String(n) }))));
  }
  function timeline(series, values) {
    if (!series) return null;
    const idx = Object.fromEntries((values || []).map((v, k) => [String(v), k]));
    const nums = (values || []).map(Number), numeric = nums.every((x) => !Number.isNaN(x));
    const lo = numeric ? Math.min(...nums) : 0, hi = numeric ? Math.max(...nums) : Math.max(1, (values || []).length - 1);
    const pos = (v) => { const x = numeric ? Number(v) : idx[String(v)]; return hi > lo ? ((x - lo) / (hi - lo)) * 100 : 50; };
    return el("div", { class: "timeline" }, Object.entries(series).slice(0, 60).map(([s, pts]) => el("div", { class: "tl-row" },
      el("span", { class: "bar-k", text: s }),
      el("span", { class: "tl-track" }, pts.filter((p) => p[1] != null).map(([sid, tv]) => el("i", { class: "tl-dot", style: `left:${pos(tv)}%`, title: `${sid}: ${tv}` }))))),
      el("div", { class: "item-sub", text: `time axis: ${values?.[0]} … ${values?.[values.length - 1]}${numeric ? "" : " (ordered labels)"}` }));
  }
  async function previewRule(rule, slot) {
    try {
      const r = await api("/api/design/preview", { session_id: S.session.session_id, rule });
      slot.replaceChildren(el("div", { class: "item-sub", text: `Parsed ${r.n_parsed} of ${r.n_total} sample names${r.n_failures ? ` — not parsed: ${r.failures.slice(0, 6).join(", ")}${r.n_failures > 6 ? " …" : ""}` : ""}. ${r.summary.label || ""}` }),
        el("table", { class: "ledger" }, el("thead", {}, el("tr", {}, ["sample", "subject", "time"].map((h) => el("th", { text: h })))),
          el("tbody", {}, r.rows.slice(0, 10).map((x) => el("tr", { class: x.subject == null && x.time == null ? "bad" : "" }, el("td", { text: x.sample }), el("td", { text: x.subject ?? "—" }), el("td", { text: x.time ?? "—" }))))));
    } catch (e) { slot.replaceChildren(el("div", { class: "alert alert-error", text: e.message })); }
  }
  function stepDesign() {
    const d = S.draft, des = d.design, rep = d.design_report || {}, body = [];
    S.local.des = S.local.des || {};
    const L = S.local.des;
    const cur = (side) => L[side] ?? srcKey(des[side]);
    const rule = L.rule || des.derivation || { delimiter: "_", occurrence: "last", left: "subject", right: "time" };
    for (const side of ["subject", "time"]) {
      body.push(el("div", { class: "item" },
        el("div", { class: "item-head" }, el("span", { class: "item-name", text: SIDE_LABEL[side] }),
          el("div", { class: "meta-row" }, provBadge(des[side].provenance, des[side].source_of_proposal === "ai" ? "ai" : "none"))),
        fieldBox("Comes from", select(designSourceOptions(rep), cur(side), (v) => { L[side] = v; renderGuide(); }, { labels: (v) => designSourceLabel(rep, v) }))));
    }
    if ([cur("subject"), cur("time")].includes("derived_from_sample_names")) {
      const slot = el("div", {});
      const upd = (k, v) => { L.rule = { ...rule, [k]: v || null }; previewRule(L.rule, slot); };
      body.push(el("div", { class: "block-card" }, el("h4", { text: "Rule over the sample names (no regex)" }),
        el("div", { class: "item-controls" },
          fieldBox("Split at", textInput(rule.delimiter, (v) => upd("delimiter", v), { placeholder: "_" })),
          fieldBox("Occurrence", select(["first", "last"], rule.occurrence, (v) => upd("occurrence", v))),
          fieldBox("Left part is", select(["subject", "time", ""], rule.left || "", (v) => upd("left", v), { labels: (x) => x || "nothing" })),
          fieldBox("Right part is", select(["subject", "time", ""], rule.right || "", (v) => upd("right", v), { labels: (x) => x || "nothing" }))),
        slot));
      setTimeout(() => previewRule(rule, slot), 0);
    }
    const rm = rep.repeated_measures;
    body.push(el("div", { class: "section-label", text: "Computed from your choices" }),
      el("div", { class: "notice" }, el("b", { text: rep.label || "—" }),
        rm ? ` · ${rep.n_subjects} subjects, ${rm.subjects_with_single_sample} with a single sample · repeated measures ${rm.detected ? "detected" : "not detected"} (computed)` : "",
        rep.time_kind ? ` · time: ${rep.time_kind}, ${rep.n_distinct_time} distinct values` : ""));
    if (rm) body.push(el("div", { class: "section-label", text: "Samples per subject" }), barChart(rm.per_subject));
    if (rep.series) body.push(el("div", { class: "section-label", text: "Timeline (one dot per sample)" }), timeline(rep.series, rep.time_values));
    if ((rep.cross_checks || []).length) body.push(el("div", { class: "section-label", text: "Cross-checks between sources" }),
      el("ul", { class: "np" }, rep.cross_checks.map((c) => el("li", { class: c.disagree ? "warn-text" : "" }, `${c.check.replace(/_vs_/, " vs ").replace(/_/g, " ")}: ${c.agree} agree, ${c.disagree} disagree`))));
    if ((rep.varies_within_subject || []).length) body.push(el("div", { class: "section-label", text: "Does it change within a subject?" }),
      el("table", { class: "ledger" }, el("tbody", {}, rep.varies_within_subject.slice(0, 80).map((x) => el("tr", {},
        el("td", { text: x.column }), el("td", { text: x.file }), el("td", { text: pretty(x.audit_kind || "") }),
        el("td", { text: x.varies_within_subject === true ? "varies" : x.varies_within_subject === false ? "constant per subject" : "n/a" }))))));
    const toSrc = (v) => v === "none" ? { source: "none" } : v === "derived_from_sample_names" ? { source: v } : { source: "metadata_column", file: v.split(":")[0], column: v.slice(v.indexOf(":") + 1) };
    return {
      title: rm?.detected ? "Repeated measures: is this the study design?" : "Study design: subject and time",
      question: "Where do the subject and the time point come from? The AI proposes, code computes the facts below, and you confirm. The time unit is always asked, never guessed.",
      body,
      decision: () => {
        const dec = {};
        for (const side of ["subject", "time"]) if (L[side] && L[side] !== srcKey(des[side])) dec[side] = toSrc(L[side]);
        if (L.rule) dec.derivation = L.rule;
        return Object.keys(dec).length ? { design: dec } : {};
      },
    };
  }

  function stepHistory() {
    const d = S.draft;
    S.local.hist = S.local.hist || JSON.parse(JSON.stringify(d.processing_history));
    if (S.local.software == null) S.local.software = d.software_and_version || "";
    if (S.local.notes == null) S.local.notes = d.history_notes || "";
    const kept = Object.keys(d.groups).filter((g) => D(g).role === "value" && D(g).keep !== false);
    const body = [];
    const described = kept.map((g) => D(g).label).filter(Boolean);
    if (described.length) body.push(el("div", { class: "notice", text: `Value block(s) described as: ${described.map((x) => `“${x}”`).join(", ")}. That is only a description; please answer below.` }));
    for (const q of S.historyQ) {
      const a = S.local.hist[q.id];
      body.push(el("div", { class: "hq" }, el("p", { text: q.question }),
        el("div", { class: "radios" }, S.vocab.yes_no_unsure.map((v) => el("label", { class: a.answer === v ? "on" : "" },
          el("input", { type: "radio", name: q.id, checked: a.answer === v, onchange: () => { a.answer = v; renderGuide(); } }), pretty(v)))),
        textInput(a.note, (x) => { a.note = x; }, { placeholder: "note (optional)", cls: "wide" })));
    }
    body.push(fieldBox("Software and version", textInput(S.local.software, (x) => { S.local.software = x; }, { placeholder: "e.g. MaxQuant 2.4.9, normalized in Perseus", cls: "wide" })),
      fieldBox("Notes", textInput(S.local.notes, (x) => { S.local.notes = x; }, { placeholder: "anything else about how the data was produced", cls: "wide" })));
    return {
      title: "What happened to the values before you uploaded them?",
      question: "PRISM asks this and never guesses it. No answer is pre-selected; “not sure” is fine.",
      body,
      decision: () => {
        const missing = S.historyQ.filter((q) => !S.local.hist[q.id].answer);
        if (missing.length) throw new Error("Answer every question (yes, no or not sure).");
        return { processing_history: { ...S.local.hist, software_and_version: S.local.software, notes: S.local.notes } };
      },
    };
  }

  function stepReview() {
    const d = S.draft, body = [];
    const groups = Object.entries(d.groups);
    const li = (k, v) => el("li", {}, el("span", { text: k }), el("span", {}, v));
    const flags = groups.filter(([, x]) => x.role === "feature_annotation" && x.marks_rows_as_suspect);
    const nonStudy = Object.values(d.samples).filter((x) => !x.is_study_sample).length;
    body.push(el("ul", { class: "review-list" },
      li("Layout", [pretty(d.layout.value), " ", provBadge(d.layout.provenance, d.layout.source)]),
      ...d.assays.map((a) => li(`Assay · ${a.assay_label}`, [`${a.omics_type} · ${a.source_software}`, " ",
        ["no", "unsure"].includes(a.in_supported_scope) ? el("span", { class: "badge v-warning", text: "outside current scope" }) : null, " ", provBadge(a.provenance, a.source)])),
      li("Feature ID", layout() === "samples_in_rows" ? "column headers" : d.feature_identity.group_ids.map((g) => G(g).columns[0]).join(" + ") || "—"),
      ...groups.filter(([, x]) => x.role === "value").map(([g, x]) => li(`Values${x.keep === false ? " (excluded)" : ""}`, [`${x.label || G(g).columns[0]} · ${G(g).n_columns} col · ${x.assay_label}`, " ",
        provBadge(x.provenance, x.source)])),
      li("Annotations", `${groups.filter(([, x]) => x.role === "feature_annotation").length}${flags.length ? ` (${flags.map(([g, x]) => `${G(g).columns[0]}: ${x.n_flagged ?? "?"} flagged`).join(", ")})` : ""}`),
      li("Sample info", groups.filter(([, x]) => x.role === "sample_metadata").map(([g, x]) => `${G(g).columns[0]} (${pretty(x.audit_kind || "?")})`).join(", ")
        + (d.metadata?.columns ? ` + ${d.metadata.columns.length - 1} from the metadata file` : "") || "—"),
      li("Samples", `${d.sample_list.n}${nonStudy ? ` (${nonStudy} non-study)` : ""}${d.sample_list.duplicates.length ? ` (${d.sample_list.duplicates.length} duplicated)` : ""}`),
      li("Design", `${d.design_report?.label || "—"}${d.design?.time?.unit?.value ? ` · time unit: ${d.design.time.unit.value}` : ""}`)));
    body.push(el("div", { class: "section-label", text: "Every column of every file, accounted for" }), ledgerTable());
    const excl = d.excluded_columns || [];
    if (excl.length) body.push(el("details", { class: "saw" }, el("summary", { text: `${excl.length} column(s) left out of the outputs (nothing is deleted)` }),
      el("ul", { class: "np" }, excl.slice(0, 300).map((x) => el("li", {}, code(x.column), el("span", { class: "item-sub", text: ` ${x.file} · ${x.reason} · by ${pretty(x.by)}` }))))));
    const unres = (d.unresolved || []);
    if (unres.length) {
      body.push(el("div", { class: "section-label", text: "Still open — finishing is blocked" }),
        el("ul", { class: "warns" }, unres.map((u) => el("li", { class: "bad" }, u.what, " ", el("button", { class: "linkbtn", type: "button", text: "go there", onclick: () => go(u.step) })))));
    }
    if (S.finalized) {
      const f = S.finalized;
      if (f.integrity_flags.length) body.push(el("div", { class: "section-label", text: "Integrity flags" }),
        el("ul", { class: "warns" }, f.integrity_flags.map((x) => el("li", { text: `${pretty(x.flag)}: ${x.detail}` }))));
      body.push(el("div", { class: "section-label", text: "Downloads" }), el("div", { class: "downloads" }, f.artifacts.map((a) =>
        el("a", { href: `${API}/api/export/${S.session.session_id}/${a}`, download: a }, a, el("span", { text: "↓" })))),
        el("a", { class: "linkbtn", href: `${API}/api/sessions/${S.session.session_id}/log`, target: "_blank", rel: "noopener", text: "View the full session log" }));
    }
    return {
      title: S.finalized ? "Structure confirmed. Your files are ready." : "Review the recognized structure",
      question: S.finalized ? "Next, PRISM's audit step will start from this confirmed schema." : "Everything below was confirmed by you step by step. Click any step above to change it. Finishing writes schema.json and the canonical tables.",
      body,
      confirmLabel: S.finalized ? "Add another dataset" : "Confirm & finish",
      confirmDisabled: !S.finalized && unres.length > 0,
      custom: async () => {
        if (S.finalized) { reset(); return; }
        try {
          S.finalized = await api("/api/finalize", { session_id: S.session.session_id });
          S.draft.steps.review = "confirmed";
        } catch (e) { S.local.error = e.message; }
        renderAll();
      },
    };
  }

  const BUILDERS = { layout: stepLayout, feature_id: stepFeatureId, annotations: stepAnnotations, values: stepValues,
                     samples: stepSamples, sample_info: stepSampleInfo, design: stepDesign, history: stepHistory, review: stepReview };

  let current = null;
  function renderGuide() {
    if (!S.draft) return;
    const idx = STEPS.findIndex(([id]) => id === S.step);
    const st = stepStatus(S.step);
    const spec = BUILDERS[S.step]();
    current = spec;
    const actions = el("div", { class: "guide-actions" },
      el("button", { class: "btn btn-light btn-sm", type: "button", disabled: spec.confirmDisabled, onclick: () => (spec.custom ? spec.custom() : submit()) },
        spec.confirmLabel || (st === "confirmed" ? "Confirm again" : "Confirm"), el("span", { class: "kbd", text: "⏎" })),
      ...(spec.extraActions || []),
      el("span", { class: "spacer" }),
      idx > 0 ? el("button", { class: "linkbtn", type: "button", text: "← back", onclick: () => go(STEPS[idx - 1][0]) }) : null);
    const err = S.local.error ? el("div", { class: "alert alert-error guide-err", text: S.local.error }) : null;
    const note = S.local.note ? el("div", { class: "notice", text: S.local.note }) : null;
    const status = st === "confirmed" ? " · confirmed" : st === "not_applicable" ? " · not needed for this layout" : "";
    $("guide").replaceChildren(...[
      datalists(),
      el("div", { class: "guide-step" }, el("span", { text: `Step ${idx + 1} of ${STEPS.length} · ${STEPS[idx][1]}${status}` }),
        S.draft.ai.error && S.draft.ai.enabled ? el("span", { class: "badge v-warning", title: S.draft.ai.error, text: "AI unavailable" }) : null),
      el("h3", { text: spec.title }),
      spec.question ? el("p", { class: "q" }, spec.question) : null,
      questionTray(),
      stepQuestions(S.step),
      groupingTrouble(),
      changesBox(),
      note,
      el("div", { class: "guide-body" }, spec.body),
      err, actions].filter(Boolean));
    setTimeout(drawCallout, 30);
  }

  async function submit(decisionOverride) {
    if (S.busy) return;
    S.local.error = null;
    let decision;
    try { decision = decisionOverride || current.decision(); } catch (e) { S.local.error = e.message; renderGuide(); return; }
    const step = S.step;
    let res;
    await busy(step === "layout" ? "Confirming the layout" : "Saving", async (pid) => {
      try {
        res = await api("/api/confirm-step", { session_id: S.session.session_id, step_id: step, decision, progress_id: pid });
      } catch (e) { S.local.error = e.message; }
    });
    if (!res) { renderGuide(); return; }
    S.draft = res.draft;
    if (res.session) S.session = res.session;
    if (res.digests) S.digests = res.digests;
    S.local = {};
    S.step = res.reproposed ? "feature_id" : nextStep(step);
    renderAll();
  }

  document.addEventListener("keydown", (e) => {
    if (!S.draft || $("workspace").classList.contains("hidden") || S.busy) return;
    const t = e.target.tagName;
    if (e.key === "Escape") {
      if (["INPUT", "SELECT", "TEXTAREA"].includes(t)) e.target.blur();
      else if (S.chatOpen) closeChat();
    } else if (e.key === "/" && !["INPUT", "SELECT", "TEXTAREA"].includes(t)) {
      e.preventDefault(); openChat();
    } else if (e.key === "Enter" && !["INPUT", "SELECT", "TEXTAREA", "BUTTON", "A"].includes(t) && !e.metaKey && !e.ctrlKey) {
      e.preventDefault();
      if (current?.custom) { if (!current.confirmDisabled) current.custom(); } else submit();
    }
  });
})();
