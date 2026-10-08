// PRISM: the final session report (step 4). A live preview rendered by audit_report.js, plus the
// same report as one self-contained HTML file and a print / PDF version.
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  function el(tag, attrs = {}, ...children) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") n.className = v; else if (k === "text") n.textContent = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v); else n.setAttribute(k, v);
    }
    for (const c of children.flat(Infinity)) if (c != null && c !== false) n.append(c.nodeType ? c : document.createTextNode(String(c)));
    return n;
  }
  async function open(st) {
    const sid = st.session_id;
    const box = $("panel-final");
    for (const id of ["panel-upload", "panel-import", "panel-audit", "workspace"]) { const n = $(id); if (n) n.classList.add("hidden"); }
    box.classList.remove("hidden");
    box.replaceChildren(el("div", { class: "panel-body" }, el("p", { class: "q", text: "Preparing the report…" })));
    let fb;
    try {
      const r = await fetch(`/api/sessions/${sid}/final-report`);
      fb = await r.json();
      if (!r.ok) throw new Error(fb.detail || "The report could not be built.");
    } catch (e) { box.replaceChildren(el("div", { class: "panel-body" }, el("div", { class: "alert alert-error", text: e.message }))); return; }
    const url = `/api/sessions/${sid}/final-report.html`;
    const printIt = () => {
      const w = window.open(`${url}?download=0`, "_blank");
      if (w) w.addEventListener("load", () => setTimeout(() => w.print(), 400));
    };
    const actions = el("div", { class: "ar-actions" },
      el("a", { class: "ar-btn primary", href: url, download: `prism_report_${sid}.html`, id: "final-download" }, "Download HTML"),
      el("button", { class: "ar-btn", type: "button", onclick: printIt }, "Print / PDF"),
      el("button", { class: "ar-btn", type: "button", onclick: () => { box.classList.add("hidden"); window.PRISM_SESSION && window.PRISM_SESSION.showStart(); } }, "Close"));
    const root = el("div", { id: "final-report" });
    box.replaceChildren(el("div", { class: "panel-body final-body" }, root));
    window.PRISM_REPORT.renderFinal(root, fb, { live: true, actions });
    box.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  window.PRISM_FINAL = { open };
})();
