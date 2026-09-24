// PRISM Step 0 — upload → preview → structure → confirmed.
// All roles shown here come from the backend; the user must explicitly
// confirm them (and resolve every "unresolved" column) before anything is used.

(() => {
  "use strict";

  const API = "";                 // same origin as the FastAPI app
  const ACCEPTED = [".csv", ".tsv", ".txt"];
  const UNRESOLVED = "unresolved";
  const LOW_CONFIDENCE = 0.6;
  const ROLE_HELP = {
    sample_id: "Sample identifier",
    subject_id: "Subject / patient / animal",
    timepoint: "Time point / visit",
    batch: "Batch / plate / run group",
    group_or_outcome: "Group, condition or outcome",
    feature_value: "Measured value",
    feature_annotation: "Feature annotation / ID",
    ignore: "Ignore (exclude)",
  };

  const $ = (id) => document.getElementById(id);
  const state = { data: null, roles: [], vocab: Object.keys(ROLE_HELP), editable: false };

  // tiny DOM helper; text is always set via textContent (never innerHTML)
  function el(tag, attrs = {}, ...children) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) if (c != null) n.append(c.nodeType ? c : document.createTextNode(String(c)));
    return n;
  }
  const chip = (role) => el("span", { class: `role-chip r-${role}`, text: role });
  const fmt = (n) => (n == null ? "—" : Number(n).toLocaleString());

  function setStep(n) {
    document.querySelectorAll("#steps .step").forEach((s) => {
      const k = Number(s.dataset.step);
      s.classList.toggle("current", k === n);
      s.classList.toggle("done", k < n);
    });
  }
  const show = (id, on = true) => $(id).classList.toggle("hidden", !on);

  // ------------------------------------------------------------ health
  fetch(`${API}/api/health`).then((r) => r.json()).then((h) => {
    state.vocab = h.roles;
    $("ai-status").textContent = h.ai_available ? `AI fallback: ${h.ai_model}` : "AI fallback off · manual mapping";
    $("ai-status").title = h.ai_available
      ? "Used only when no known format signature matches."
      : "Set ANTHROPIC_API_KEY on the server to enable AI-assisted proposals.";
  }).catch(() => { $("ai-status").textContent = "Backend unreachable"; });

  // ------------------------------------------------------------ upload
  const input = $("file-input"), dz = $("dropzone");
  input.addEventListener("change", () => input.files[0] && upload(input.files[0]));
  ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));
  $("restart").addEventListener("click", reset);

  function uploadError(msg) {
    $("upload-error").textContent = msg;
    show("upload-error");
  }

  async function upload(file) {
    show("upload-error", false);
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!ACCEPTED.includes(ext)) {
      uploadError(`"${file.name}" is not a CSV/TSV file. PRISM accepts quantified tables only — ` +
        "export a protein, peptide or feature table from your analysis software (MaxQuant, DIA-NN, " +
        "Spectronaut, FragPipe, XCMS, MZmine…) as .csv or .tsv and upload that. Raw spectra are not processed.");
      input.value = "";
      return;
    }
    $("loading-text").textContent = `Reading ${file.name} and checking known format signatures… (the AI fallback runs only if none match)`;
    show("upload-loading");
    dz.style.pointerEvents = "none";
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${API}/api/upload`, { method: "POST", body: fd });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(body.detail || `Upload failed (HTTP ${res.status}).`);
      onUploaded(body);
    } catch (err) {
      uploadError(err.message || String(err));
    } finally {
      show("upload-loading", false);
      dz.style.pointerEvents = "";
      input.value = "";
    }
  }

  function reset() {
    state.data = null; state.roles = [];
    ["panel-preview", "panel-structure", "panel-summary", "restart", "upload-error", "confirm-error-wrap"].forEach((id) => show(id, false));
    show("panel-upload");
    setStep(1);
    window.scrollTo({ top: 0 });
  }

  function onUploaded(data) {
    state.data = data;
    state.roles = data.columns.map((c) => c.role);
    state.editable = data.method !== "signature";
    show("panel-upload", false);
    show("restart");
    renderPreview();
    renderStructure();
    show("panel-preview"); show("panel-structure");
    setStep(3);
    $("panel-preview").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ------------------------------------------------------------ preview
  function renderPreview() {
    const d = state.data, t = $("preview-table");
    const delim = { ",": "comma", ";": "semicolon", "\t": "tab" }[d.delimiter] || d.delimiter;
    $("preview-meta").textContent =
      `${d.filename} · ${fmt(d.n_rows)} rows × ${fmt(d.n_columns)} columns · ${delim}-separated · showing first ${d.preview_rows.length}`;
    t.replaceChildren(
      el("thead", {}, el("tr", {},
        el("th", { class: "rownum", text: "#" }),
        d.header.map((h, i) => el("th", { title: h }, h || `(column ${i + 1})`, el("span", { class: "role-chip", "data-col": i }))))),
      el("tbody", {}, d.preview_rows.map((r, ri) =>
        el("tr", {}, el("td", { class: "rownum", text: ri + 1 }),
          d.header.map((_, i) => el("td", { title: r[i] ?? "", text: r[i] ?? "" })))))
    );
    updatePreviewChips();
    const w = d.warnings;
    $("file-warnings").replaceChildren(...(w.length ? [el("b", { text: "Notes" }), el("ul", {}, w.map((x) => el("li", { text: x })))] : []));
    show("file-warnings", w.length > 0);
  }

  function updatePreviewChips() {
    document.querySelectorAll("#preview-table th .role-chip").forEach((c) => {
      const r = state.roles[Number(c.dataset.col)];
      c.textContent = r;
      c.className = `role-chip r-${r}`;
    });
  }

  // ------------------------------------------------------------ structure
  function roleSelect(i) {
    const current = state.roles[i];
    const sel = el("select", { class: "select", "aria-label": `Role for column ${state.data.header[i]}` },
      current === UNRESOLVED ? el("option", { value: UNRESOLVED, disabled: true, selected: true, text: "— choose a role —" }) : null,
      state.vocab.map((r) => el("option", { value: r, selected: r === current, title: ROLE_HELP[r], text: r })));
    sel.addEventListener("change", () => { state.roles[i] = sel.value; refreshRow(i); updateConfirmState(); updatePreviewChips(); });
    return sel;
  }

  function confidenceCell(c) {
    if (c.confidence == null) return el("span", { class: "conf-num", text: "—" });
    const pct = Math.round(c.confidence * 100);
    return el("div", { class: `conf${c.confidence < LOW_CONFIDENCE ? " low" : ""}`, title: `${pct}% confidence` },
      el("span", { class: "conf-bar" }, el("i", { style: `width:${pct}%` })),
      el("span", { class: "conf-num", text: `${pct}%` }));
  }

  function sampleValues(i) {
    return state.data.preview_rows.slice(0, 3).map((r) => r[i] ?? "").map((v) => (v.length > 24 ? v.slice(0, 23) + "…" : v)).join(" · ");
  }

  function buildRolesTable(opts) {
    const d = state.data;
    const tbody = el("tbody", { id: "roles-body" });
    d.columns.forEach((c, i) => {
      const proposed = c.role;
      tbody.append(el("tr", { "data-i": i },
        el("td", { class: "conf-num", text: i + 1 }),
        el("td", { class: "colname" }, d.header[i] || `(column ${i + 1})`),
        el("td", { class: "samples", text: sampleValues(i) }),
        el("td", {}, chip(proposed), opts.showConfidence ? el("div", { style: "margin-top:6px" }, confidenceCell(c)) : null,
        ),
        el("td", { class: "evidence", text: opts.showEvidence === false ? "" : (c.evidence || "") }),
        el("td", { class: "role-cell" }, opts.editable ? roleSelect(i) : chip(state.roles[i]))));
    });
    return el("div", { class: "table-scroll", style: "max-height:560px" },
      el("table", { class: "roles-table" },
        el("thead", {}, el("tr", {},
          el("th", { text: "#" }), el("th", { text: "Column" }), el("th", { text: "Sample values" }),
          el("th", { text: opts.proposedLabel }), el("th", { text: "Evidence" }), el("th", { text: "Confirmed role" }))),
        tbody));
  }

  function refreshRow(i) {
    const tr = document.querySelector(`#roles-body tr[data-i="${i}"]`);
    if (!tr) return;
    const r = state.roles[i], proposed = state.data.columns[i].role;
    tr.classList.toggle("unresolved", r === UNRESOLVED);
    tr.classList.toggle("changed", r !== proposed && r !== UNRESOLVED);
    tr.classList.toggle("ignored", r === "ignore");
    const edit = tr.querySelector(".tag-edit");
    if (r !== proposed && proposed !== UNRESOLVED && !edit) tr.querySelector(".role-cell").append(el("div", {}, el("span", { class: "tag-edit", text: "edited" })));
    if (r === proposed && edit) edit.parentElement.remove();
  }

  function toolbar() {
    const search = el("input", { class: "input", type: "search", placeholder: "Filter columns by name…", "aria-label": "Filter columns" });
    const view = el("select", { class: "select", "aria-label": "Show" },
      el("option", { value: "all", text: "All columns" }),
      el("option", { value: "unresolved", text: "Unresolved only" }),
      el("option", { value: "low", text: `Confidence < ${LOW_CONFIDENCE * 100}%` }),
      el("option", { value: "changed", text: "Edited by me" }));
    const bulkRole = el("select", { class: "select", "aria-label": "Role to apply" },
      state.vocab.map((r) => el("option", { value: r, text: r })));
    const count = el("span", { class: "count-note" });
    const visibleRows = () => [...document.querySelectorAll("#roles-body tr")].filter((tr) => !tr.classList.contains("hidden"));

    function apply() {
      const q = search.value.trim().toLowerCase(), mode = view.value;
      let n = 0;
      document.querySelectorAll("#roles-body tr").forEach((tr) => {
        const i = Number(tr.dataset.i), c = state.data.columns[i];
        let ok = !q || state.data.header[i].toLowerCase().includes(q);
        if (mode === "unresolved") ok = ok && state.roles[i] === UNRESOLVED;
        if (mode === "low") ok = ok && (c.confidence ?? 1) < LOW_CONFIDENCE;
        if (mode === "changed") ok = ok && state.roles[i] !== c.role;
        tr.classList.toggle("hidden", !ok);
        n += ok;
      });
      count.textContent = `${n} of ${state.data.header.length} shown`;
    }
    search.addEventListener("input", apply);
    view.addEventListener("change", apply);

    const bulkBtn = el("button", { class: "btn btn-sm", type: "button", text: "Apply to shown" });
    bulkBtn.addEventListener("click", () => {
      const rows = visibleRows();
      if (!rows.length) return;
      if (rows.length > 1 && !confirm(`Set ${rows.length} shown column(s) to "${bulkRole.value}"?`)) return;
      rows.forEach((tr) => {
        const i = Number(tr.dataset.i);
        state.roles[i] = bulkRole.value;
        tr.querySelector(".role-cell").replaceChildren(roleSelect(i));
        refreshRow(i);
      });
      updateConfirmState(); updatePreviewChips(); apply();
    });
    setTimeout(apply);
    return el("div", { class: "toolbar" }, search, view, el("span", { class: "spacer" }), count,
      el("span", { class: "count-note", text: "Set shown to" }), bulkRole, bulkBtn);
  }

  function roleSummary() {
    const counts = {};
    state.roles.forEach((r) => { counts[r] = (counts[r] || 0) + 1; });
    return el("div", { class: "role-summary", "data-live": "roles" },
      Object.entries(counts).map(([r, n]) => el("span", { class: `role-chip r-${r}`, text: `${r} · ${n}` })));
  }

  function renderStructure() {
    const d = state.data, body = $("structure-body");
    body.replaceChildren();

    if (d.method === "signature") {
      const det = d.detection;
      $("structure-meta").textContent = `deterministic match · ${det.signature}`;
      body.append(
        el("div", { class: "recognized" },
          el("span", { class: "badge" }, checkIcon()),
          el("div", {},
            el("h3", { text: `Recognized as ${det.platform} output` }),
            el("p", { text: "Matched an exact column signature, so no AI was used. Roles below were resolved by fixed rules." }))),
        el("div", { class: "kv" },
          kv("Platform", det.platform), kv("Omics type", det.omics_type),
          kv("Value columns", fmt(det.n_value_columns)), kv("Feature ID", det.feature_id_note),
          kv("Layout", det.layout === "long" ? "Long (one row per feature per run)" : "One row per feature")),
        el("div", { class: "eyebrow", style: "margin-bottom:10px", text: "Resolved roles" }),
        roleSummary());

      const holder = el("div", { style: "margin-top:20px" });
      const toggle = el("button", { class: "btn btn-sm", type: "button", text: "Review / adjust column roles" });
      toggle.addEventListener("click", () => {
        state.editable = true;
        toggle.remove();
        holder.replaceChildren(
          el("p", { class: "count-note", style: "margin:0 0 12px", text: "Any change you make is logged against the deterministic proposal." }),
          toolbar(), buildRolesTable({ editable: true, showConfidence: false, proposedLabel: "Resolved role" }));
        state.roles.forEach((_, i) => refreshRow(i));
      });
      holder.append(toggle);
      body.append(holder);
    } else {
      const ai = d.ai || {};
      const isAI = d.method === "ai";
      $("structure-meta").textContent = isAI ? `AI-assisted · ${ai.model}` : "manual assignment";
      body.append(el("div", { class: "recognized" },
        el("span", { class: "badge", style: isAI ? "" : "border-color:var(--accent)" }, isAI ? sparkIcon() : "!"),
        el("div", {},
          el("h3", { text: isAI ? "No known signature matched — AI-proposed roles" : "No known signature matched — assign roles manually" }),
          el("p", { text: isAI
            ? `Each role below is a proposal from ${ai.model}, shown with its confidence and evidence. Nothing is used until you confirm. Highlighted columns are unresolved and need your choice.`
            : (ai.error || "AI fallback unavailable.") + " Choose a role for every column." }))));
      if (isAI) body.append(el("p", { class: "count-note", style: "margin:14px 0 0", text: "Only the header and the first rows (cell text truncated) were sent to the model." }));
      body.append(el("div", { style: "margin:20px 0 14px" }, roleSummary()), toolbar(),
        buildRolesTable({ editable: true, showConfidence: isAI, showEvidence: isAI, proposedLabel: isAI ? "AI proposal" : "Proposal" }));
      state.roles.forEach((_, i) => refreshRow(i));
    }
    updateConfirmState();
  }

  function kv(k, v) { return el("div", {}, el("span", { text: k }), el("strong", { text: v })); }
  function checkIcon() {
    const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    s.setAttribute("width", "18"); s.setAttribute("height", "18"); s.setAttribute("viewBox", "0 0 24 24");
    s.innerHTML = '<path d="M5 12.5l4.5 4.5L19 7.5" fill="none" stroke="#fff" stroke-width="1.5"/>';
    return s;
  }
  function sparkIcon() {
    const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    s.setAttribute("width", "18"); s.setAttribute("height", "18"); s.setAttribute("viewBox", "0 0 24 24");
    s.innerHTML = '<path d="M12 3l1.8 6.2L20 11l-6.2 1.8L12 19l-1.8-6.2L4 11l6.2-1.8z" fill="none" stroke="#fff" stroke-width="1.2"/>';
    return s;
  }

  function updateConfirmState() {
    document.querySelectorAll('[data-live="roles"]').forEach((n) => n.replaceWith(roleSummary()));
    const unresolved = state.roles.filter((r) => r === UNRESOLVED).length;
    const changed = state.roles.filter((r, i) => r !== state.data.columns[i].role && state.data.columns[i].role !== UNRESOLVED).length;
    const note = $("confirm-note");
    note.replaceChildren();
    if (unresolved) {
      note.append(el("b", { text: `${unresolved} column${unresolved > 1 ? "s" : ""} still need${unresolved > 1 ? "" : "s"} a role.` }), " Pick one for each highlighted row to continue.");
    } else {
      note.append(`All ${state.roles.length} columns have a role`, changed ? ` · ${changed} edited by you` : "",
        ". Confirming records these roles; it does not change any values.");
    }
    $("confirm-btn").disabled = unresolved > 0;
  }

  // ------------------------------------------------------------ confirm
  $("confirm-btn").addEventListener("click", async () => {
    const d = state.data;
    if (state.roles.some((r) => r === UNRESOLVED)) return;
    const btn = $("confirm-btn");
    btn.disabled = true; btn.textContent = "Confirming…";
    show("confirm-error-wrap", false);
    try {
      const res = await fetch(`${API}/api/sessions/${d.session_id}/confirm`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ columns: d.header.map((h, i) => ({ index: i, column: h, role: state.roles[i] })) }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        const msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body);
        throw new Error(msg || `HTTP ${res.status}`);
      }
      renderSummary(body);
    } catch (err) {
      $("confirm-error").textContent = `Could not confirm: ${err.message}`;
      show("confirm-error-wrap");
      btn.disabled = false;
    } finally {
      btn.textContent = "Confirm structure";
    }
  });

  function renderSummary(s) {
    show("panel-structure", false);
    show("panel-preview", false);
    show("panel-summary");
    setStep(4);
    document.querySelectorAll("#steps .step").forEach((x) => x.classList.add("done"));
    $("summary-meta").textContent = `session ${s.session_id} · ${new Date(s.confirmed_at).toLocaleString()}`;

    const byRole = {};
    s.columns.forEach((c) => { (byRole[c.role] = byRole[c.role] || []).push(c.column || `(column ${c.index + 1})`); });

    const methodLabel = { signature: "Deterministic signature", ai: "AI-proposed, user-confirmed", manual: "Manually assigned" }[s.method];
    const body = $("summary-body");
    body.replaceChildren(...[
      el("div", { class: "done-hero" },
        el("div", { class: "ring" }, checkIcon()),
        el("h2", { text: "Structure confirmed" }),
        el("p", { text: `${s.filename} is recognized as ${s.platform} (${s.omics_type}). This confirmed structure is the input to the next PRISM step.` })),
      el("div", { class: "big-stats" },
        bigStat(fmt(s.sample_count), "Samples"), bigStat(fmt(s.feature_count), "Features"),
        bigStat(fmt(s.n_columns), "Columns labelled"), bigStat(fmt(s.changes_from_proposal), "Roles set or changed by you")),
      el("div", { class: "kv" },
        kv("Detected platform", s.platform), kv("Omics type", s.omics_type), kv("Method", methodLabel),
        kv("Layout", s.layout_description), kv("Feature ID", s.feature_id), kv("Data rows", fmt(s.n_rows))),
      s.warnings.length ? el("div", { class: "alert alert-warn", style: "margin:0 0 20px" }, el("b", { text: "Notes" }), el("ul", {}, s.warnings.map((w) => el("li", { text: w })))) : null,
      s.design_factors.length ? el("div", { style: "margin-bottom:22px" },
        el("div", { class: "eyebrow", style: "margin-bottom:10px", text: "Design columns" }),
        el("div", { class: "role-summary" }, s.design_factors.map((f) => el("span", { class: "role-chip", text: `${f.column} · ${f.role} · ${f.n_levels} levels` })))) : null,
      el("div", { class: "eyebrow", style: "margin-bottom:10px", text: "Final roles" }),
      el("div", { class: "table-scroll", style: "max-height:420px;margin-bottom:22px" },
        el("table", { class: "roles-table" },
          el("thead", {}, el("tr", {}, el("th", { text: "Role" }), el("th", { text: "Count" }), el("th", { text: "Columns" }))),
          el("tbody", {}, state.vocab.filter((r) => byRole[r]).map((r) =>
            el("tr", {}, el("td", {}, chip(r)), el("td", { class: "conf-num", text: byRole[r].length }),
              el("td", { class: "colname", style: "max-width:none", text: listCols(byRole[r]) })))))),
      el("div", { class: "next-step" },
        el("span", {}, el("b", { style: "color:var(--text-dim);font-weight:500", text: "Next · Step 1 — Statistical audit. " }), "Not built yet; it will start from this confirmed structure."),
        el("span", { style: "display:flex;gap:10px;flex-wrap:wrap" },
          el("button", { class: "btn btn-sm", type: "button", onclick: () => download(s), text: "Download summary (JSON)" }),
          el("a", { class: "btn btn-sm", href: `${API}/api/sessions/${s.session_id}/log`, target: "_blank", rel: "noopener", text: "View session log" }),
          el("button", { class: "btn btn-sm btn-light", type: "button", onclick: reset, text: "Analyze another file" })))]
      .filter(Boolean));
    $("panel-summary").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function listCols(cols) {
    return cols.length <= 12 ? cols.join(", ") : `${cols.slice(0, 10).join(", ")} … and ${cols.length - 10} more`;
  }
  function bigStat(v, label) { return el("div", { class: "stat" }, el("strong", { text: v }), el("span", { text: label })); }
  function download(s) {
    const blob = new Blob([JSON.stringify(s, null, 2)], { type: "application/json" });
    const a = el("a", { href: URL.createObjectURL(blob), download: `prism_structure_${s.session_id}.json` });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
})();
