// PRISM Step 0 — guided confirmation wizard.
// The backend is the source of truth (the "draft"); every step sends a
// decision to /api/confirm-step and re-renders from the returned draft.

(() => {
  "use strict";

  const API = "";
  const ACCEPTED = [".csv", ".tsv", ".txt", ".tab"];
  const STEPS = [
    ["layout", "Layout"], ["feature_id", "Feature ID"], ["annotations", "Annotations"], ["values", "Values"],
    ["samples", "Samples"], ["sample_info", "Sample info"], ["history", "History"], ["review", "Review"],
  ];
  const PREVIEW_COLS = 30;
  const S = { vocab: null, defs: {}, historyQ: [], session: null, draft: null, digests: null, step: "layout",
              editing: false, local: {}, aiOn: true, finalized: null, busy: false };

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
  const groupsById = () => Object.fromEntries((S.session?.groups || []).map((g) => [g.group_id, g]));
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
    S.vocab = v.vocabulary; S.defs = v.definitions; S.historyQ = v.history_questions; S.unsupported = v.unsupported_notice;
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
    Object.assign(S, { session: null, draft: null, digests: null, step: "layout", editing: false, local: {}, finalized: null });
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
      $("ws-file").replaceChildren(el("b", { text: body.filename }), ` · ${body.n_rows.toLocaleString()} rows × ${body.n_columns.toLocaleString()} columns · ${body.groups.length} column groups`);
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
        S.step = "layout"; S.editing = false; S.local = {};
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
    if (it.role === "value") return `value_${it.block_role || "auxiliary"}`;
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
    const k = it.kind ? ` · ${pretty(it.kind)}` : "";
    return {
      feature_id: "feature ID", feature_annotation: "annotation" + k, sample_id: "sample ID", sample_metadata: "sample info" + k,
      ignore: "ignored", unresolved: "unresolved",
      value: `value · ${it.block_role || "auxiliary"}${it.label ? " · " + it.label : ""}`,
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
      case "samples": return lay === "samples_in_columns" ? all.filter((g) => D(g).role === "value" && D(g).block_role === "primary") : byRole("sample_id");
      case "history": return all.filter((g) => D(g).role === "value" && D(g).block_role === "primary");
      case "review": return [];
      case "values": {
        const blocks = all.filter((g) => D(g).role === "value").sort((a, b) => (D(a).block_role === "primary" ? -1 : 0) - (D(b).block_role === "primary" ? -1 : 0));
        return [...blocks, ...all.filter((g) => !blocks.includes(g) && (stepFor(g) === "values" || (lay === "samples_in_rows"
          && G(g)?.type === "numeric" && G(g).n_columns === 1 && ["sample_metadata", "ignore"].includes(D(g).role))))];
      }
      default: return all.filter((g) => stepFor(g) === step);
    }
  }

  // ------------------------------------------------------------ render
  function renderAll(scroll) {
    renderStepper();
    renderLegend();
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
    const sig = S.draft.signature;
    $("ws-sig").textContent = sig ? `Recognized as ${sig.platform} output` : (S.draft.ai.enabled && S.draft.ai.available ? "No known format: AI-assisted" : "No known format: manual mode");
  }

  function go(step) {
    S.step = step; S.editing = false; S.local = {};
    renderStepper(); renderPreview(); renderGuide();
  }

  function nextStep(from) {
    const idx = STEPS.findIndex(([id]) => id === from);
    for (let k = idx + 1; k < STEPS.length; k++) if (stepStatus(STEPS[k][0]) === "pending" || STEPS[k][0] === "review") return STEPS[k][0];
    return "review";
  }

  function renderLegend() {
    const items = [["feature ID", "var(--c-fid)"], ["annotation", "var(--c-ann)"], ["value (primary)", "var(--c-val)"],
      ["value (auxiliary)", "transparent;border:1px solid var(--c-val)"], ["sample ID", "var(--c-sid)"], ["sample info", "var(--c-smd)"],
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
    $("saw-pre").textContent = head + (S.digests && S.digests.length ? JSON.stringify(S.digests.length === 1 ? S.digests[0] : S.digests, null, 2) : "");
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
      chips.append(el("th", { colspan: j - k + 1, class: focus.has(gid) ? "focus" : "", "data-gid": gid },
        el("span", { class: `chip ${cls}`, title: `${roleLabel(it)} — ${G(gid).n_columns} column(s)` },
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
  function provBadge(p, source) {
    const map = { signature: ["Signature", "p-signature"], computed: ["Computed", "p-computed"],
                  ai_proposed_confirmed: ["AI suggestion", "p-ai"], ai_proposed_corrected: ["You (corrected AI)", "p-you"],
                  user_set: ["You", "p-you"] };
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
    if (claimed) msgs.unshift(el("li", { class: "bad", text: `The AI proposed ${pretty(claimed.role)}${claimed.measurement_type ? " / " + pretty(claimed.measurement_type) : ""}${claimed.kind ? " / " + pretty(claimed.kind) : ""}, but the data contradicts it. Please choose.` }));
    return msgs.length ? el("ul", { class: "warns" }, msgs) : null;
  }
  function metaRow(it, fields) {
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
  function fieldBox(label, node) { return el("label", { class: "field" }, label, node); }
  // local edits of group items for the current step
  function local(gid) { S.local.items = S.local.items || {}; return (S.local.items[gid] = S.local.items[gid] || {}); }
  function cur(gid, f) { const l = S.local.items?.[gid]; return l && f in l ? l[f] : D(gid)[f]; }
  function colsText(g) { return g.n_columns <= 3 ? g.columns.join(", ") : `${g.columns.slice(0, 2).join(", ")} … ${g.columns[g.columns.length - 1]} (${g.n_columns} columns)`; }

  function roleSelect(gid, allowed) {
    const roles = allowed || S.vocab.column_role.filter((r) => r !== "unresolved");
    return select(roles, cur(gid, "role") === "unresolved" ? null : cur(gid, "role"), (v) => { local(gid).role = v; if (v !== D(gid).role) local(gid).kind = null; renderGuide(); },
      { placeholder: "choose a role…", bad: cur(gid, "role") === "unresolved" });
  }
  function kindSelect(gid) {
    const role = cur(gid, "role");
    const opts = role === "feature_annotation" ? S.vocab.feature_annotation_kind : role === "sample_metadata" ? S.vocab.sample_metadata_kind : null;
    if (!opts) return null;
    const v = cur(gid, "kind");
    return select(opts, opts.includes(v) ? v : null, (x) => { local(gid).kind = x; renderGuide(); }, { placeholder: "choose a kind…", bad: !opts.includes(v) });
  }
  function reconsiderBox(gid) {
    if (!(S.draft.ai.enabled && S.draft.ai.available)) return null;
    const open = S.local.reconsider === gid;
    if (!open) return el("button", { class: "linkbtn", type: "button", text: "Ask AI to reconsider", onclick: () => { S.local.reconsider = gid; renderGuide(); } });
    const inp = el("input", { class: "input", type: "text", placeholder: "Optional hint, e.g. 'these are ion adducts'" });
    setTimeout(() => inp.focus(), 0);
    return el("div", { class: "reconsider" }, inp, el("button", { class: "btn btn-sm", type: "button", text: "Ask", onclick: async () => {
      await busy("Asking the AI to reconsider", async (pid) => {
        try {
          const r = await api("/api/reconsider", { session_id: S.session.session_id, group_id: gid, user_hint: inp.value, progress_id: pid });
          S.draft = r.draft;
          if (r.digest) S.digests = [r.digest];
          S.local = {};
        } catch (err) { S.local.error = err.message; }
      });
      renderAll();
    } }));
  }

  // generic item editor (annotations, sample info, other numeric)
  function itemCard(gid, { roles, extra } = {}) {
    const it = D(gid), g = G(gid);
    const unres = cur(gid, "role") === "unresolved" || (["feature_annotation", "sample_metadata"].includes(cur(gid, "role")) && !cur(gid, "kind"));
    const draftUnres = it.role === "unresolved" || (["feature_annotation", "sample_metadata"].includes(it.role) && !it.kind);
    const touched = S.local.items?.[gid] && Object.keys(S.local.items[gid]).length > 0;
    const editing = S.editing || unres || draftUnres || touched || it.validation?.status === "contradicted";
    const kids = [el("div", { class: "item-head" }, el("span", { class: "item-name", text: colsText(g) }), metaRow(it))];
    if (!editing) {
      kids.push(el("div", { class: "item-sub", text: roleLabel({ ...it, role: cur(gid, "role"), kind: cur(gid, "kind") }) + (cur(gid, "keep") === false ? " · dropped" : "") }));
    } else {
      const ctr = el("div", { class: "item-controls" }, roleSelect(gid, roles), kindSelect(gid) || el("span"));
      if (cur(gid, "kind") === "timepoint") ctr.append(el("div", { class: "full" }, select(S.vocab.timepoint_detail, cur(gid, "detail"), (v) => { local(gid).detail = v; }, { placeholder: "time point detail…" })));
      if (["feature_annotation", "sample_metadata"].includes(cur(gid, "role"))) {
        const cb = el("input", { type: "checkbox", checked: cur(gid, "keep") !== false });
        cb.addEventListener("change", () => { local(gid).keep = cb.checked; });
        ctr.append(el("label", { class: "keep full" }, cb, "keep in the outputs"));
      }
      kids.push(ctr);
    }
    if (extra) kids.push(extra);
    kids.push(el("div", { class: "item-sub", text: it.hint || "" }), evidence(it.evidence), warnList(it.validation, it.claimed));
    if (it.source === "ai" || it.role === "unresolved") kids.push(el("div", { style: "margin-top:6px" }, reconsiderBox(gid)));
    return el("div", { class: `item ${unres ? "unres" : ""}`, onmouseenter: () => hoverGroup(gid, true), onmouseleave: () => hoverGroup(gid, false) }, kids);
  }
  function hoverGroup(gid, on) {
    $("pv").querySelectorAll(`th[data-gid="${gid}"] .chip`).forEach((c) => c.style.outline = on ? "1px solid #fff" : "");
  }
  function itemDecisions(gids) {
    return gids.map((gid) => {
      const l = S.local.items?.[gid] || {};
      const out = { group_id: gid };
      for (const f of ["role", "kind", "detail", "keep", "block_role", "measurement_type", "scale", "omics_type", "label"]) if (f in l) out[f] = l[f];
      return out;
    }).filter((d) => Object.keys(d).length > 1);
  }

  // ------------------------------------------------------------ steps
  const LAYOUT_TEXT = {
    samples_in_columns: ["Samples in columns", "Each row looks like one feature, and each sample has its own column. Is that right?"],
    samples_in_rows: ["Samples in rows", "Each row looks like one sample, and each feature has its own column. Is that right?"],
    long: ["Long table", "Each row looks like one measurement: a feature, a sample and a single value. Is that right?"],
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

  function stepLayout() {
    const d = S.draft, lay = S.local.layout ?? d.layout.value, om = S.local.omics ?? d.omics_type.value,
          sw = S.local.software ?? d.source_software.value;
    const known = LAYOUT_TEXT[lay];
    const body = [];
    if (d.signature) body.push(el("div", { class: "notice", text: `Recognized as ${d.signature.platform} output (exact column signature, no AI needed for this).` }));
    const editing = S.editing || !LAYOUT_TEXT[d.layout.value] || S.local.layout != null || !S.vocab.omics_type.includes(om) || om === "unknown";
    if (editing) {
      body.push(el("div", { class: "opt-cards" }, Object.entries(LAYOUT_TEXT).map(([k, [t, q]]) => el("button", {
        type: "button", class: `opt-card ${lay === k ? "sel" : ""}`, onclick: () => { S.local.layout = k; renderGuide(); } },
        layoutIcon(k), el("strong", { text: t }), q.replace(" Is that right?", "")))));
      if (d.layout.hint) body.push(el("div", { class: "notice", text: "Hint from the data: " + d.layout.hint }));
      body.push(el("div", { class: "row2" },
        fieldBox("Omics type", select(S.vocab.omics_type, om, (v) => { S.local.omics = v; renderGuide(); })),
        fieldBox("Source software", (() => { const i = el("input", { class: "input", value: sw === "unknown" ? "" : sw, placeholder: "unknown" }); i.addEventListener("input", () => { S.local.software = i.value; }); return i; })())));
    } else {
      body.push(el("div", { class: "answer" }, el("div", { class: "opt-card sel", style: "cursor:default" }, layoutIcon(lay), el("strong", { text: known[0] })),
        el("div", { class: "meta-row" }, `Omics type: `, el("b", { text: pretty(om) }), " · Software: ", el("b", { text: sw || "unknown" })),
        metaRow(d.layout), evidence(d.layout.evidence), warnList(d.layout.validation)));
    }
    if (om && !S.vocab.supported_downstream.includes(om)) body.push(el("div", { class: "notice accent", text: S.unsupported }));
    return {
      title: known ? known[1] : "How is this table organised?",
      question: known ? null : "Pick the layout that matches your file. Features usually outnumber samples.",
      body,
      decision: () => {
        if (!LAYOUT_TEXT[lay]) throw new Error("Choose one of the three layouts.");
        if (!om || !S.vocab.omics_type.includes(om)) throw new Error("Choose an omics type.");
        return { layout: lay, omics_type: om, source_software: sw || "unknown" };
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
    const body = [];
    const editingFid = S.editing || S.local.fid != null || !d.feature_identity.group_ids.length;
    if (!editingFid) {
      body.push(el("div", { class: "answer" }, el("div", { class: "val" }, labels.map((l, k) => [k ? " + " : "", code(l)])),
        metaRow(fi), evidence(fi.evidence), warnList(fi.validation)));
    } else {
      body.push(el("div", { class: "section-label", text: "Tick one column, or several for a composite key (e.g. m/z + RT)" }));
      body.push(el("div", { class: "items" }, candidateIdGroups().map((g) => {
        const cb = el("input", { type: "checkbox", checked: sel.includes(g.group_id) });
        cb.addEventListener("change", () => { const s = new Set(S.local.fid ?? d.feature_identity.group_ids); cb.checked ? s.add(g.group_id) : s.delete(g.group_id); S.local.fid = [...s]; renderGuide(); });
        return el("label", { class: "item", style: "display:flex;gap:10px;align-items:center;cursor:pointer" }, cb,
          el("span", {}, el("span", { class: "item-name", text: g.columns[0] }), el("div", { class: "item-sub", text: D(g.group_id).hint || "" })));
      })));
      body.push(warnList(fi.validation));
    }
    let sid = S.local.sid ?? d.sample_id_group.value;
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
    const flags = gids.filter((g) => (cur(g, "kind") || "").startsWith("flag_"));
    const body = [];
    if (!gids.length) body.push(el("p", { class: "q", text: "No annotation columns in this file." }));
    for (const gid of flags) {
      const n = S.draft.flags?.[gid];
      if (n != null) body.push(el("div", { class: "notice", text: `${G(gid).columns[0]}: ${n} row(s) are flagged as ${pretty(cur(gid, "kind")).replace("flag ", "")}. Nothing is removed now; flagged rows are recorded for the audit step.` }));
    }
    body.push(el("div", { class: "items" }, gids.map((g) => itemCard(g))));
    return {
      title: `${gids.length} column(s) describe the features. Are these labels right?`,
      question: "Each gets a kind (gene symbol, m/z, flag…). Everything is kept unless you untick it.",
      body,
      decision: () => ({ items: itemDecisions(gids) }),
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
    const br = cur(gid, "block_role") || "auxiliary", mt = cur(gid, "measurement_type"), sc = cur(gid, "scale"), om = cur(gid, "omics_type");
    const n = g.n_columns;
    const voice = lay === "samples_in_rows"
      ? `These ${n} columns look like ${n} features measured in every sample${it.label ? ` (${it.label})` : ""}.`
      : `These ${n} columns look like one measurement per sample${it.label ? ` (${it.label})` : ""}.`;
    const look = `Median ${fmtNum(p.median)}, range ${fmtNum(p.min)}–${fmtNum(p.max)}, ${pct(p.frac_zero)} zeros, ${pct(p.frac_na)} missing — this looks like ${sc === "linear" ? "raw, linear-scale" : pretty(sc || "unknown") + "-scale"} ${pretty(mt || "values")}.`;
    const seg = el("div", { class: "seg orange" }, S.vocab.block_role.map((r) => el("button", { type: "button", class: br === r ? "on" : "", text: r,
      title: S.defs[r] || "", onclick: () => { local(gid).block_role = r; renderGuide(); } })));
    const canvas = el("canvas", { class: "hist" });
    setTimeout(() => drawHist(canvas, g.histogram), 0);
    const touched = S.local.items?.[gid] && ["measurement_type", "scale", "omics_type", "role"].some((f) => f in S.local.items[gid]);
    const editing = S.editing || it.role === "unresolved" || touched || it.validation?.status === "contradicted";
    const controls = editing ? el("div", { class: "item-controls" },
      fieldBox("Measurement", select(S.vocab.measurement_type, mt, (v) => { local(gid).measurement_type = v; renderGuide(); }, { placeholder: "choose…" })),
      fieldBox("Scale", select(S.vocab.scale, sc, (v) => { local(gid).scale = v; renderGuide(); }, { placeholder: "choose…" })),
      fieldBox("Omics type", select(S.vocab.omics_type, om, (v) => { local(gid).omics_type = v; renderGuide(); })),
      fieldBox("Role", roleSelect(gid))) : null;
    const samples = lay === "samples_in_columns" ? (S.draft.blocks_samples?.[gid] || []) : [];
    return el("div", { class: `block-card ${br === "primary" ? "primary" : ""}`, onmouseenter: () => hoverGroup(gid, true), onmouseleave: () => hoverGroup(gid, false) },
      el("div", { class: "item-head" }, el("h4", {}, it.label && it.label !== "values" ? it.label : (g.pattern ? g.pattern.text.trim() : `${n} numeric columns`),
        el("span", { class: "item-sub", text: `  ${pretty(om || "unknown")} · ${n} columns` })), metaRow(it)),
      el("div", { class: "q", style: "margin:6px 0 0" }, voice, " ", look, " Correct?"),
      seg, controls,
      canvas,
      el("div", { class: "stats" }, [["min", p.min], ["p1", p.p1], ["median", p.median], ["p99", p.p99], ["max", p.max],
        ["zeros", pct(p.frac_zero)], ["missing", pct(p.frac_na)], ["whole numbers", p.integer_valued ? "yes" : "no"]]
        .map(([k, v]) => el("div", {}, el("span", { text: k }), el("b", { text: typeof v === "number" ? fmtNum(v) : v })))),
      samples.length ? el("div", { class: "item-sub", style: "margin-top:6px" }, `Samples: ${samples.slice(0, 6).join(", ")}${samples.length > 6 ? ` … (${samples.length})` : ""}`) : null,
      evidence(it.evidence), warnList(it.validation, it.claimed),
      it.source === "ai" || it.role === "unresolved" ? el("div", { style: "margin-top:6px" }, reconsiderBox(gid)) : null);
  }

  function stepValues() {
    const all = focusGroups("values");
    const blocks = all.filter((g) => D(g).role === "value" || (D(g).role === "unresolved" && G(g).kind === "numeric_block"));
    const others = all.filter((g) => !blocks.includes(g));
    const body = [el("div", {}, blocks.map(valueCard))];
    if (others.length) {
      body.push(el("div", { class: "section-label", text: "Numeric columns that don't look like features" }));
      body.push(el("p", { class: "q", text: "Covariates and technical values (age, CD4 count, iron, scale factors…) go to Sample info. Change any of them if needed." }));
      body.push(el("div", { class: "items" }, others.map((g) => itemCard(g, { roles: ["sample_metadata", "value", "ignore", "feature_annotation"] }))));
    }
    const prim = blocks.filter((g) => (cur(g, "block_role")) === "primary");
    return {
      title: blocks.length > 1 ? `${blocks.length} candidate measurement blocks. Which is the main one?` : "Is this the measurement block?",
      question: "Mark one primary block per omics type; the others can stay as auxiliary (kept alongside) or be excluded.",
      body,
      decision: () => {
        if (!prim.length) throw new Error("Mark at least one block as primary.");
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
  function stepSamples() {
    const d = S.draft, lay = layout(), body = [];
    if (lay === "samples_in_columns") {
      const prim = Object.keys(d.groups).filter((g) => D(g).role === "value" && D(g).block_role === "primary");
      S.local.rules = S.local.rules || {};
      for (const gid of prim) {
        const base = d.sample_rules[gid] || { strip_prefix: "", strip_suffix: "" };
        const r = S.local.rules[gid] || { strip_prefix: base.strip_prefix, strip_suffix: base.strip_suffix };
        const ids = G(gid).indices.map((i) => applyRule(header(i), r));
        const mk = (k, ph) => { const i = el("input", { class: "input", value: r[k], placeholder: ph }); i.addEventListener("input", () => { S.local.rules[gid] = { ...r, [k]: i.value }; renderGuideKeepFocus(i); }); return i; };
        body.push(el("div", { class: "block-card" }, el("h4", { text: D(gid).label || "Primary block" }),
          el("div", { class: "row2", style: "margin-top:8px" }, fieldBox("Strip prefix", mk("strip_prefix", "none")), fieldBox("Strip suffix", mk("strip_suffix", "none"))),
          idPreview(ids)));
      }
    } else {
      const sid = S.local.sid ?? d.sample_id_group.value;
      body.push(fieldBox("Column identifying each sample", select(candidateIdGroups().map((g) => g.group_id), sid,
        (v) => { S.local.sid = v; renderGuide(); }, { placeholder: "choose…", labels: (g) => G(g).columns[0], bad: !sid })));
      if (sid === d.sample_id_group.value) body.push(idPreview(d.samples.ids));
      else body.push(el("div", { class: "item-sub", text: "The ID list updates after you confirm." }));
    }
    // sample types
    const types = d.sample_types, ids = Object.keys(types);
    S.local.types = S.local.types || {};
    const tval = (s) => S.local.types[s] ?? types[s].type;
    const counts = {}; ids.forEach((s) => { counts[tval(s)] = (counts[tval(s)] || 0) + 1; });
    body.push(el("div", { class: "section-label", text: "Sample types" }),
      el("div", { class: "item-sub", style: "margin-bottom:8px", text: Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(" · ") + ". Nothing is dropped: QC, blanks and pools are only labelled." }));
    const special = ids.filter((s) => tval(s) !== "study");
    const list = S.local.showAllTypes ? ids.slice(0, 600) : special.slice(0, 60);
    body.push(el("div", { class: "types" }, list.map((s) => el("span", { class: `type-chip t-${tval(s)}`, title: types[s].evidence || "" }, s,
      select(S.vocab.sample_type, tval(s), (v) => { S.local.types[s] = v; renderGuide(); })))));
    body.push(el("button", { class: "linkbtn", type: "button", style: "margin-top:8px",
      text: S.local.showAllTypes ? "Show only non-study samples" : `Show all ${ids.length} samples`, onclick: () => { S.local.showAllTypes = !S.local.showAllTypes; renderGuide(); } }));
    return {
      title: `${d.samples.n} samples found. Are the sample IDs right?`,
      question: lay === "samples_in_columns" ? "Sample IDs come from the measurement column names. Adjust what is stripped; the preview updates live." : "Sample IDs come from one column.",
      body,
      decision: () => {
        const dec = {};
        if (lay === "samples_in_columns") dec.sample_rules = S.local.rules || {};
        else { const sid = S.local.sid ?? d.sample_id_group.value; if (!sid) throw new Error("Choose the sample ID column."); dec.sample_id_group = sid; }
        if (Object.keys(S.local.types || {}).length) dec.sample_types = S.local.types;
        return dec;
      },
    };
  }

  function stepSampleInfo() {
    const d = S.draft, lay = layout(), body = [];
    if (lay !== "samples_in_columns") {
      const gids = focusGroups("sample_info");
      if (!gids.length) body.push(el("p", { class: "q", text: "No sample information columns besides the sample ID." }));
      body.push(el("div", { class: "items" }, gids.map((g) => itemCard(g, { roles: ["sample_metadata", "sample_id", "ignore", "feature_annotation", "value"] }))));
      return { title: `${gids.length} column(s) describe the samples. Are these right?`,
        question: "PRISM only labels candidates: which variable is the research outcome is decided in a later step.",
        body, decision: () => ({ items: itemDecisions(gids) }) };
    }
    const meta = d.metadata;
    const fileIn = el("input", { type: "file", accept: ".csv,.tsv,.txt", class: "input" });
    fileIn.addEventListener("change", async () => {
      if (!fileIn.files[0]) return;
      const fd = new FormData(); fd.append("session_id", S.session.session_id); fd.append("file", fileIn.files[0]);
      try { const r = await api("/api/metadata-upload", fd); S.draft = r.draft; S.local = {}; } catch (e) { S.local.error = e.message; }
      renderGuide();
    });
    body.push(fieldBox("Upload a sample metadata file (optional)", fileIn));
    if (meta && !meta.skipped && meta.report) {
      const rep = meta.report;
      S.local.near = S.local.near || {};
      body.push(el("div", { class: "notice", text: `Matched ${rep.n_matched} sample(s) using column '${meta.id_column}'. Only in data: ${rep.only_in_data.length}. Only in metadata: ${rep.only_in_metadata.length}.` }));
      if (rep.only_in_data.length) body.push(el("div", { class: "item-sub", text: "Only in data: " + rep.only_in_data.slice(0, 10).join(", ") }));
      if (rep.only_in_metadata.length) body.push(el("div", { class: "item-sub", text: "Only in metadata: " + rep.only_in_metadata.slice(0, 10).join(", ") }));
      if (rep.near_misses.length) {
        body.push(el("div", { class: "section-label", text: "Near misses (suggestions — tick to accept)" }));
        body.push(el("div", { class: "items" }, rep.near_misses.map((n) => {
          const key = `${n.data_id}|${n.metadata_id}`;
          const cb = el("input", { type: "checkbox", checked: !!S.local.near[key] });
          cb.addEventListener("change", () => { S.local.near[key] = cb.checked; });
          return el("label", { class: "item", style: "display:flex;gap:10px;align-items:center" }, cb, code(n.data_id), " ↔ ", code(n.metadata_id), el("span", { class: "item-sub", text: " " + n.reason }));
        })));
      }
      S.local.metaCols = S.local.metaCols || {};
      body.push(el("div", { class: "section-label", text: "Metadata columns" }));
      body.push(el("div", { class: "items" }, meta.columns.filter((c) => c.role !== "sample_id").map((c) => {
        const v = S.local.metaCols[c.column] ?? c.kind;
        return el("div", { class: `item ${v ? "" : "unres"}` }, el("div", { class: "item-head" }, el("span", { class: "item-name", text: c.column })),
          el("div", { class: "item-controls" }, select(S.vocab.sample_metadata_kind, v, (x) => { S.local.metaCols[c.column] = x; renderGuide(); }, { placeholder: "choose a kind…", bad: !v })),
          el("div", { class: "item-sub", text: c.hint }));
      })));
    }
    return {
      title: "Do you have a sample metadata file?",
      question: "Your table has samples in columns, so information about samples (group, time point, batch…) usually lives in a separate sheet. You can also skip this; that is recorded.",
      body,
      extraActions: [el("button", { class: "btn btn-sm", type: "button", text: "Skip (record as missing)", onclick: () => submit({ metadata: { skip: true } }) })],
      decision: () => {
        if (!meta || meta.skipped || !meta.report) throw new Error("Upload a file, or use 'Skip'.");
        const cols = meta.columns.filter((c) => c.role !== "sample_id").map((c) => ({ column: c.column, kind: S.local.metaCols?.[c.column] ?? c.kind, keep: true }));
        const acc = Object.entries(S.local.near || {}).filter(([, v]) => v).map(([k]) => k.split("|"));
        return { metadata: { columns: cols, accept_near_misses: acc } };
      },
    };
  }

  function stepHistory() {
    const d = S.draft;
    S.local.hist = S.local.hist || JSON.parse(JSON.stringify(d.processing_history));
    if (S.local.software == null) S.local.software = d.software_and_version || "";
    if (S.local.notes == null) S.local.notes = d.history_notes || "";
    const logBlocks = Object.keys(d.groups).filter((g) => D(g).role === "value" && D(g).block_role === "primary" && ["log2", "log10", "ln"].includes(D(g).scale));
    const body = [];
    if (logBlocks.length) body.push(el("div", { class: "notice", text: `Hint from the data: the values of ${D(logBlocks[0]).label || "the primary block"} look ${D(logBlocks[0]).scale}-scale. This is only a hint; please answer below.` }));
    for (const q of S.historyQ) {
      const a = S.local.hist[q.id];
      body.push(el("div", { class: "hq" }, el("p", { text: q.question }),
        el("div", { class: "radios" }, S.vocab.yes_no_unsure.map((v) => el("label", { class: a.answer === v ? "on" : "" },
          el("input", { type: "radio", name: q.id, checked: a.answer === v, onchange: () => { a.answer = v; renderGuide(); } }), pretty(v)))),
        (() => { const i = el("input", { class: "input", value: a.note || "", placeholder: "note (optional)", style: "width:100%" }); i.addEventListener("input", () => { a.note = i.value; }); return i; })()));
    }
    const sw = el("input", { class: "input", value: S.local.software, placeholder: "e.g. MaxQuant 2.4.9, normalized in Perseus", style: "width:100%" });
    sw.addEventListener("input", () => { S.local.software = sw.value; });
    const nt = el("input", { class: "input", value: S.local.notes, placeholder: "anything else about how the data was produced", style: "width:100%" });
    nt.addEventListener("input", () => { S.local.notes = nt.value; });
    body.push(fieldBox("Software and version", sw), fieldBox("Notes", nt));
    return {
      title: "What happened to the values before you uploaded them?",
      question: "PRISM asks this and never guesses it. No answer is pre-selected; “not sure” is fine.",
      body, noEdit: true,
      decision: () => {
        const missing = S.historyQ.filter((q) => !S.local.hist[q.id].answer);
        if (missing.length) throw new Error("Answer every question (yes, no or not sure).");
        return { processing_history: { ...S.local.hist, software_and_version: S.local.software, notes: S.local.notes } };
      },
    };
  }

  function stepReview() {
    const d = S.draft, body = [];
    const values = Object.keys(d.groups).filter((g) => D(g).role === "value");
    const li = (k, v) => el("li", {}, el("span", { text: k }), el("span", {}, v));
    body.push(el("ul", { class: "review-list" },
      li("Layout", [pretty(d.layout.value), " ", provBadge(d.layout.provenance, d.layout.source)]),
      li("Omics type", [pretty(d.omics_type.value), " ", provBadge(d.omics_type.provenance, d.omics_type.source)]),
      li("Software", d.source_software.value),
      li("Feature ID", layout() === "samples_in_rows" ? "column headers" : d.feature_identity.group_ids.map((g) => G(g).columns[0]).join(" + ") || "—"),
      ...values.map((g) => li(`Values (${D(g).block_role})`, `${D(g).label || G(g).columns[0]} · ${G(g).n_columns} col · ${pretty(D(g).omics_type)} · ${pretty(D(g).measurement_type)} · ${D(g).scale}`)),
      li("Annotations", String(Object.values(d.groups).filter((x) => x.role === "feature_annotation").length)),
      li("Sample info", String(Object.values(d.groups).filter((x) => x.role === "sample_metadata").length + (d.metadata?.columns ? d.metadata.columns.length - 1 : 0))),
      li("Samples", `${d.samples.n}${d.samples.duplicates.length ? ` (${d.samples.duplicates.length} duplicated)` : ""}`)));
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
      question: S.finalized ? "Next, PRISM's audit step will start from this confirmed schema." : "Everything below was confirmed by you step by step. Finishing writes schema.json and the canonical tables.",
      body, noEdit: true, noReconsider: true,
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
                     samples: stepSamples, sample_info: stepSampleInfo, history: stepHistory, review: stepReview };

  function renderGuideKeepFocus(input) {
    const pos = input.selectionStart, ph = input.placeholder, val = input.value;
    renderGuide();
    const again = [...$("guide").querySelectorAll("input")].find((i) => i.placeholder === ph && i.value === val);
    if (again) { again.focus(); try { again.setSelectionRange(pos, pos); } catch (_) {} }
  }

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
      spec.noEdit ? null : el("button", { class: "btn btn-sm", type: "button", onclick: () => { S.editing = !S.editing; renderGuide(); } },
        S.editing ? "Done changing" : "Change…", el("span", { class: "kbd", text: "Esc" })),
      ...(spec.extraActions || []),
      el("span", { class: "spacer" }),
      idx > 0 ? el("button", { class: "linkbtn", type: "button", text: "← back", onclick: () => go(STEPS[idx - 1][0]) }) : null);
    const err = S.local.error ? el("div", { class: "alert alert-error guide-err", text: S.local.error }) : null;
    const status = st === "confirmed" ? " · confirmed" : st === "not_applicable" ? " · not needed for this layout" : "";
    $("guide").replaceChildren(...[
      el("div", { class: "guide-step" }, el("span", { text: `Step ${idx + 1} of 8 · ${STEPS[idx][1]}${status}` }),
        S.draft.ai.error && S.draft.ai.enabled ? el("span", { class: "badge v-warning", title: S.draft.ai.error, text: "AI unavailable" }) : null),
      el("h3", { text: spec.title }),
      spec.question ? el("p", { class: "q" }, spec.question) : null,
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
    S.editing = false; S.local = {};
    S.step = res.reproposed ? "feature_id" : nextStep(step);
    renderAll();
  }

  document.addEventListener("keydown", (e) => {
    if (!S.draft || $("workspace").classList.contains("hidden") || S.busy) return;
    const t = e.target.tagName;
    if (e.key === "Escape") {
      if (["INPUT", "SELECT", "TEXTAREA"].includes(t)) { e.target.blur(); return; }
      if (current && !current.noEdit) { S.editing = !S.editing; renderGuide(); }
    } else if (e.key === "Enter" && !["INPUT", "SELECT", "TEXTAREA", "BUTTON", "A"].includes(t) && !e.metaKey && !e.ctrlKey) {
      e.preventDefault();
      if (current?.custom) { if (!current.confirmDisabled) current.custom(); } else submit();
    }
  });
})();
