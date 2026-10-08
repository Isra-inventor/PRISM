// PRISM report renderer: the audit report and the final session report, live and exported.
// Summary first (key numbers, highlights, what needs your input, declared vs observed), then the
// details: one collapsible card per audit with plots and tables. Plain JS + SVG; no libraries,
// no network. Every number and sentence comes from the findings.

(() => {
  "use strict";
  const NAMES = { A1: "Integrity", A2: "Data type and scale", A3: "Distribution", A4: "Missing and floor values",
    A5: "Sample size", A6: "Noise and QC", A7: "Batch structure", A8: "Repeated measures",
    A9: "Sample source", A10: "Multi-omics overlap", A11: "Outliers" };
  const ORDER = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A11", "A10"];
  // which sentences surface first in the summary (the rest stay in the cards)
  const PRIORITY = ["declared_not_imputed_floor_ties", "declared_not_normalized_signature", "declared_not_logged_log_like",
    "dimension_mismatch", "samples_x_vs_m", "duplicate_samples", "near_duplicate_samples", "duplicate_sample_ids",
    "batch_confounded", "batch_nested", "permanova_association", "pc_association", "missingness_associated",
    "outlier_sample", "outlier_max_cell", "floor_ties_present", "sentinel_value", "abundance_dependent_missingness",
    "median_scaling_signature", "run_order_drift", "qc_rsd", "source_association", "effective_sample_size", "high_icc",
    "repeated_measures", "irregular_time_grid", "out_of_scope_type", "missing_values_present", "sd_grows_with_mean"];
  const HOWTO = {
    "batch variable": "Add a sample-metadata file with a plate or batch column, or mark a column as a batch variable in Overrides.",
    "sample roles": "Mark QC, blank and pool samples in Overrides (sample role).",
    "subject": "Set where the subject comes from in the wizard's Design step.",
    "time": "Set where the time comes from in the wizard's Design step.",
    "sample source": "Mark a sample-type column, or add a source variable in Overrides.",
    "run order": "Add a run-order column to the sample metadata.",
  };
  const SVGNS = "http://www.w3.org/2000/svg";

  // ------------------------------------------------------------------ small helpers
  function el(tag, attrs = {}, ...children) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k === "html") n.innerHTML = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v);
    }
    for (const c of children.flat(Infinity)) if (c != null && c !== false) n.append(c.nodeType ? c : document.createTextNode(String(c)));
    return n;
  }
  function svg(tag, attrs = {}, ...children) {
    const n = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v != null) n.setAttribute(k, v);
    for (const c of children.flat()) if (c != null) n.append(c.nodeType ? c : document.createTextNode(String(c)));
    return n;
  }
  function fmt(v) {
    if (v == null) return "—";
    if (typeof v === "boolean") return v ? "yes" : "no";
    if (typeof v === "number") {
      if (!isFinite(v)) return "—";
      if (Number.isInteger(v)) return Math.abs(v) >= 10000 ? v.toLocaleString("en-US") : String(v);
      const a = Math.abs(v);
      if (a >= 1000) return v.toLocaleString("en-US", { maximumFractionDigits: 1 });
      if (a >= 0.001) return String(+v.toPrecision(3));
      return v.toExponential(2);
    }
    if (Array.isArray(v)) return v.length <= 6 ? v.map(fmt).join(", ") : `${v.slice(0, 6).map(fmt).join(", ")} … (${v.length})`;
    if (typeof v === "object") return Object.entries(v).slice(0, 12).map(([k, x]) => `${k}: ${typeof x === "object" && x ? JSON.stringify(x) : fmt(x)}`).join(", ");
    return String(v);
  }
  const pct = (v) => v == null ? "—" : `${+(100 * v).toPrecision(3)}%`;
  const get = (o, path) => path.split(".").reduce((x, k) => (x == null ? x : x[k]), o);
  const icon = (d) => { const s = svg("svg", { viewBox: "0 0 24 24", "aria-hidden": "true" }); s.innerHTML = d; return s; };
  const I = {
    print: '<path d="M7 9V3h10v6M7 17H4v-7h16v7h-3M7 14h10v7H7z"/>',
    download: '<path d="M12 3v12M7 10l5 5 5-5M4 20h16"/>',
    chev: '<path d="M6 9l6 6 6-6"/>',
    sun: '<circle cx="12" cy="12" r="4.2"/><path d="M12 2.5v2.2M12 19.3v2.2M2.5 12h2.2M19.3 12h2.2M5.3 5.3l1.6 1.6M17.1 17.1l1.6 1.6M5.3 18.7l1.6-1.6M17.1 6.9l1.6-1.6"/>',
    moon: '<path d="M20 14.6A8.3 8.3 0 0 1 9.4 4a8.3 8.3 0 1 0 10.6 10.6Z"/>',
  };

  // one shared hover tooltip: any element with data-tip
  let TIP = null;
  function tooltips(root) {
    if (!TIP) { TIP = el("div", { class: "ar-tip", role: "tooltip" }); document.body.append(TIP); }
    root.addEventListener("pointermove", (e) => {
      const t = e.target.closest && e.target.closest("[data-tip]");
      if (!t) { TIP.classList.remove("on"); return; }
      TIP.textContent = t.getAttribute("data-tip");
      const x = Math.min(e.clientX + 14, innerWidth - TIP.offsetWidth - 8), y = Math.min(e.clientY + 14, innerHeight - TIP.offsetHeight - 8);
      TIP.style.left = `${x}px`; TIP.style.top = `${y}px`;
      TIP.classList.add("on");
    });
    root.addEventListener("pointerleave", () => TIP.classList.remove("on"));
  }

  // ------------------------------------------------------------------ plots (SVG)
  const W = 440, H = 230, PAD = { l: 46, r: 12, t: 10, b: 34 };
  function short(v) {
    const a = Math.abs(v);
    if (a >= 1e4 || (a > 0 && a < 1e-2)) return v.toExponential(0);
    return String(+v.toPrecision(3));
  }
  function axes(g, x0, x1, y0, y1, sx, sy, o = {}) {
    for (let i = 0; i <= 4; i++) {
      const t = y0 + (y1 - y0) * i / 4;
      g.append(svg("line", { class: "ar-grid", x1: PAD.l, x2: W - PAD.r, y1: sy(t), y2: sy(t) }));
      g.append(svg("text", { x: PAD.l - 6, y: sy(t) + 4, "text-anchor": "end" }, o.ylab ? o.ylab(t) : short(t)));
    }
    for (let i = 0; i <= 3; i++) {
      const t = x0 + (x1 - x0) * i / 3;
      g.append(svg("text", { x: sx(t), y: H - PAD.b + 15, "text-anchor": i === 0 ? "start" : i === 3 ? "end" : "middle" }, o.xlab ? o.xlab(t) : short(t)));
    }
    g.append(svg("line", { class: "ar-axis", x1: PAD.l, x2: W - PAD.r, y1: H - PAD.b, y2: H - PAD.b }));
    if (o.xtitle) g.append(svg("text", { x: (PAD.l + W - PAD.r) / 2, y: H - 3, "text-anchor": "middle" }, o.xtitle));
  }
  function box(title, node, extra, sub) {
    return el("figure", { class: "ar-plot", style: "margin:0" },
      el("figcaption", { class: "ar-plot-title" }, el("b", { text: title }), sub ? el("span", { text: sub }) : null, extra || null), node);
  }
  function histogram(h, title, xtitle) {
    if (!h || !h.counts || !h.counts.length) return null;
    const e = h.edges, c = h.counts, ymax = Math.max(...c, 1);
    const sx = (v) => PAD.l + (v - e[0]) / ((e[e.length - 1] - e[0]) || 1) * (W - PAD.l - PAD.r);
    const sy = (v) => H - PAD.b - v / ymax * (H - PAD.t - PAD.b);
    const g = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title });
    axes(g, e[0], e[e.length - 1], 0, ymax, sx, sy, { xtitle: h.log10 ? `${xtitle || "value"} (log10)` : xtitle, ylab: (t) => String(Math.round(t)) });
    c.forEach((n, i) => {
      const x = sx(e[i]), w = Math.max(1, sx(e[i + 1]) - x - 2);       // 2px surface gap between bars
      const tip = `${short(e[i])} to ${short(e[i + 1])}${h.log10 ? " (log10)" : ""}\n${n} feature(s)`;
      if (n > 0) g.append(svg("rect", { class: "ar-bar", x, y: sy(n), width: w, height: Math.max(0, H - PAD.b - sy(n)), rx: 2, "data-tip": tip }));
      g.append(svg("rect", { class: "ar-hit", x, y: PAD.t, width: w + 2, height: H - PAD.b - PAD.t, "data-tip": tip }));
    });
    return box(title, g, null, `${c.reduce((a, b) => a + b, 0).toLocaleString("en-US")} values`);
  }
  function scatterSvg(P, o = {}) {
    if (!P.length) return null;
    const xs = P.map((p) => p.x), ys = P.map((p) => p.y);
    let [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    if (x0 === x1) { x0 -= 1; x1 += 1; } if (y0 === y1) { y0 -= 1; y1 += 1; }
    const sx = (v) => PAD.l + (v - x0) / (x1 - x0) * (W - PAD.l - PAD.r), sy = (v) => H - PAD.b - (v - y0) / (y1 - y0) * (H - PAD.t - PAD.b);
    const g = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.label || "scatter" });
    axes(g, x0, x1, y0, y1, sx, sy, { xtitle: o.xtitle, xlab: o.logx ? (t) => `1e${short(t)}` : null, ylab: o.logy ? (t) => `1e${short(t)}` : null });
    if (o.line) {
      const L = o.line.filter((p) => isFinite(p[0]) && isFinite(p[1]));
      g.append(svg("polyline", { class: "ar-line", points: L.map((p) => `${sx(p[0])},${sy(p[1])}`).join(" ") }));
    }
    for (const p of P) {
      const c = svg("circle", { class: `ar-dot${p.cls ? " " + p.cls : ""}`, cx: sx(p.x), cy: sy(p.y), r: o.r || 4, style: p.color ? `fill:${p.color}` : null, "data-tip": p.tip || null });
      g.append(c);
      if (p.tip && (o.r || 4) < 6) g.append(svg("circle", { class: "ar-hit", cx: sx(p.x), cy: sy(p.y), r: 8, "data-tip": p.tip }));
    }
    return g;
  }
  function scatter(points, title, o = {}) {
    const tx = o.logx ? Math.log10 : (v) => v, ty = o.logy ? Math.log10 : (v) => v;
    const P = (points || []).map((p) => ({ x: tx(p[0]), y: ty(p[1]), tip: p[2] || `${fmt(p[0])}, ${fmt(p[1])}` })).filter((p) => isFinite(p.x) && isFinite(p.y));
    const g = scatterSvg(P, { ...o, label: title, r: o.r || 2.6 });
    return g ? box(title, g, null, o.sub) : null;
  }
  // PCA: <= 3 levels get categorical slots 1-3; more levels stay neutral and a legend entry
  // highlights one level (scatter plots validate only three hues against each other).
  const VIEWS = {};
  function pcaPlot(pca, key) {
    if (!pca || !pca.samples || !pca.samples.length) return null;
    const vars = Object.keys(pca.samples[0].values || {});
    const k = pca.variance_explained.length;
    const st = { color: vars.find((v) => v !== "subject") || vars[0] || "", x: 0, y: Math.min(1, k - 1), focus: null };
    const wrap = el("figure", { class: "ar-plot", style: "margin:0" });
    const sel = (opts, cur, on) => el("select", { onchange: (e) => on(e.target.value) }, opts.map(([v, l]) => el("option", { value: v, selected: String(v) === String(cur) ? "selected" : null, text: l })));
    const draw = () => {
      const lv = [...new Set(pca.samples.map((s) => (s.values || {})[st.color] || ""))].sort((a, b) => (isFinite(a) && isFinite(b) ? a - b : String(a).localeCompare(String(b))));
      const few = lv.length <= 3;
      const nums = lv.filter((x) => x !== "").map(Number);
      const numeric = !few && nums.length === lv.filter((x) => x !== "").length && nums.every(isFinite);
      const lo = Math.min(...nums), hi = Math.max(...nums);
      const ramp = (v) => `color-mix(in oklab, var(--s1) ${Math.round(18 + 82 * ((+v - lo) / ((hi - lo) || 1)))}%, var(--r-muted-mark))`;
      const P = pca.samples.map((s) => {
        const v = (s.values || {})[st.color] || "";
        let color = null, cls = "";
        if (st.color && few) color = `var(--s${lv.indexOf(v) + 1})`;
        else if (st.color && numeric && st.focus == null) color = v === "" ? null : ramp(v);
        else if (st.color) { if (st.focus != null && v === st.focus) color = "var(--s1)"; else cls = st.focus != null ? "muted dim" : "muted"; }
        return { x: s.scores[st.x], y: s.scores[st.y], color, cls, tip: `${s.sample}${st.color ? `\n${st.color}: ${v || "—"}` : ""}\nPC${st.x + 1} ${fmt(s.scores[st.x])} · PC${st.y + 1} ${fmt(s.scores[st.y])}` };
      });
      const pcs = Array.from({ length: k }, (_, i) => [i, `PC${i + 1} · ${pct(pca.variance_explained[i])}`]);
      const legend = el("div", { class: "ar-legend" }, st.color ? lv.slice(0, 24).map((l, i) => few
        ? el("span", {}, el("i", { style: `background:var(--s${i + 1})` }), ` ${l || "(empty)"}`)
        : el("button", { type: "button", class: st.focus === l ? "on" : null, onclick: () => { st.focus = st.focus === l ? null : l; draw(); } },
          el("i", { style: `background:${st.focus === l ? "var(--s1)" : numeric && st.focus == null && l !== "" ? ramp(l) : "var(--r-muted-mark)"}` }), l || "(empty)")) : null,
        st.color && !few ? el("span", { class: "ar-faint", text: "· click a level to highlight it" }) : null);
      wrap.replaceChildren(el("figcaption", { class: "ar-plot-title" }, el("b", { text: "Samples on the main axes (PCA)" }),
        sel(pcs, st.x, (v) => { st.x = +v; draw(); }), "vs", sel(pcs, st.y, (v) => { st.y = +v; draw(); }),
        vars.length ? ["colour", sel(vars.map((v) => [v, v]), st.color, (v) => { st.color = v; st.focus = null; draw(); })] : null),
        scatterSvg(P, { r: 4.5, xtitle: `PC${st.x + 1} · PC${st.y + 1}`, label: "PCA" }), legend);
    };
    draw();
    if (key) VIEWS[key] = (v) => { if (v.color_by != null) { st.color = v.color_by; st.focus = null; } if (v.pcs) { st.x = v.pcs[0] - 1; st.y = v.pcs[1] - 1; } draw(); return wrap; };
    return wrap;
  }
  function timeline(rows, unit) {
    if (!rows || !rows.length || !rows.every((r) => r.time !== "" && r.time != null && isFinite(+r.time))) return null;
    const subs = [...new Set(rows.map((r) => r.subject || "(none)"))].sort();
    const ts = rows.map((r) => +r.time), t0 = Math.min(...ts), t1 = Math.max(...ts) || 1;
    const h = Math.max(H, 22 * subs.length + PAD.t + PAD.b);
    const L = 74;
    const sx = (v) => L + (v - t0) / ((t1 - t0) || 1) * (W - L - PAD.r);
    const sy = (s) => PAD.t + 10 + subs.indexOf(s) * ((h - PAD.t - PAD.b - 10) / Math.max(1, subs.length - 1));
    const g = svg("svg", { viewBox: `0 0 ${W} ${h}`, role: "img", "aria-label": "subjects over time" });
    for (const s of subs) {
      const own = rows.filter((r) => (r.subject || "(none)") === s).map((r) => +r.time).sort((a, b) => a - b);
      g.append(svg("line", { class: "ar-grid", x1: L, x2: W - PAD.r, y1: sy(s), y2: sy(s) }));
      if (own.length > 1) g.append(svg("line", { class: "ar-axis", x1: sx(own[0]), x2: sx(own[own.length - 1]), y1: sy(s), y2: sy(s), "stroke-width": 2 }));
      g.append(svg("text", { x: L - 8, y: sy(s) + 4, "text-anchor": "end" }, s));
    }
    for (const r of rows) {
      g.append(svg("circle", { class: "ar-dot", cx: sx(+r.time), cy: sy(r.subject || "(none)"), r: 4.5, "data-tip": `${r.sample}\nsubject ${r.subject}, time ${r.time}${unit ? " " + unit : ""}` }));
    }
    for (let i = 0; i <= 3; i++) { const t = t0 + (t1 - t0) * i / 3; g.append(svg("text", { x: sx(t), y: h - PAD.b + 15, "text-anchor": "middle" }, short(t))); }
    g.append(svg("text", { x: (L + W) / 2, y: h - 3, "text-anchor": "middle" }, `time${unit ? ` (${unit})` : ""}`));
    return box("Each subject over time", g, null, `${subs.length} subjects`);
  }
  function table(head, rows, o = {}) {
    if (!rows || !rows.length) return el("div", { class: "ar-empty", text: o.empty || "None." });
    return el("div", { class: "ar-table-wrap" }, el("table", {}, el("thead", {}, el("tr", {}, head.map((h) => el("th", { text: h })))),
      el("tbody", {}, rows.map((r) => el("tr", { class: r._flag ? "flag" : null }, r.filter((_, i) => i < head.length).map((c) => el("td", { class: typeof c === "number" ? "num" : null, text: fmt(c) })))))));
  }
  function crosstab(c) {
    return el("div", { class: "ar-plot" }, el("div", { class: "ar-plot-title" }, el("b", { text: `${c.batch} × ${c.design}` })),
      table([`${c.batch} \\ ${c.design}`, ...c.design_levels], c.table.map((row, i) => [c.batch_levels[i], ...row])));
  }

  // ------------------------------------------------------------------ per-audit content
  function keyMeasures(f) {
    const m = f.measures || {}, out = [];
    const add = (label, v, as) => { if (v !== undefined && v !== null) out.push([label, as === "pct" ? pct(v) : fmt(v)]); };
    switch (f.audit_id) {
      case "A1": add("cells", get(m, "cells.n")); add("missing", get(m, "cells.missing")); add("infinite", get(m, "cells.infinite"));
        add("constant features", get(m, "constant_features.n")); add("duplicate samples", (m.duplicate_samples || []).length); break;
      case "A2": add("looks like", (m.classification || "").replace(/_/g, " ")); add("skew before / after log2", [get(m, "skewness.median_x"), get(m, "skewness.median_y")]);
        add("spread of feature medians (MAD log2)", get(m, "median_scaling.mad_log2_feature_medians")); add("max / p99", get(m, "range.max_over_p99")); break;
      case "A3": add("SD vs mean slope", get(m, "mean_sd_x.slope")); add("95% CI", get(m, "mean_sd_x.ci95")); add("mean–SD rho after log", get(m, "mean_sd_y.spearman_rho"));
        add("spread of sample medians", get(m, "rle.mad_of_sample_medians")); break;
      case "A4": add("missing cells", get(m, "missing.rate"), "pct"); add("floor-tied features", `${fmt(get(m, "floor_ties.n_features"))} of ${fmt(get(m, "floor_ties.of"))}`);
        add("floor-tied share", get(m, "floor_ties.share"), "pct"); add("floor vs abundance rho", get(m, "abundance_dependence.floor.rho")); break;
      case "A5": add("samples / features", `${fmt(m.n)} / ${fmt(m.p)}`); add("subjects", m.n_subjects); add("effective n", get(m, "design_effect.n_eff"));
        add("smallest cell", m.smallest_cell ? `${m.smallest_cell.n} (${m.smallest_cell.variable} ${m.smallest_cell.level})` : null); add("sparsity", get(m, "sparsity.share_missing_zero_or_floor"), "pct"); break;
      case "A6": add("roles", get(m, "roles.counts")); add("QC median RSD %", get(m, "qc_rsd.summary.median")); add("run-order rho", get(m, "run_order.spearman_median_y")); break;
      case "A7": add("PCs", get(m, "pca.k")); add("variance PC1 / PC2", (get(m, "pca.variance_explained") || []).slice(0, 2).map(pct).join(" / ") || null);
        add("batch candidates", (get(m, "variables.B") || []).map((v) => v.name).join(", ") || "none"); add("tests", get(m, "associations.n_tests")); break;
      case "A8": add("subjects", get(m, "design.n_subjects")); add("samples per subject", get(m, "design.subjects_per_cluster_size")); add("median ICC", get(m, "icc.raw.median"));
        add("gaps min / median / max", m.time_grid ? [m.time_grid.gap_min, m.time_grid.gap_median, m.time_grid.gap_max] : null); break;
      case "A9": add("source variables", (m.variables || []).map((v) => v.variable).join(", ") || null); break;
      case "A10": add("unified samples", get(m, "overlap.n_unified")); add("in every dataset", get(m, "overlap.n_in_all")); add("open ID suggestions", get(m, "overlap.open_id_suggestions")); break;
      case "A11": add("flagged samples", (get(m, "samples.flagged") || []).join(", ") || "none"); add("flagged cells", get(m, "cells.n_flagged")); add("cut-off |z|", get(m, "cells.cutoff")); break;
    }
    return out;
  }
  function assocTable(rows) {
    return table(["on", "variable", "test", "statistic", "p", "q", "p min", "n", "permutations"],
      (rows || []).map((x) => [x.pc ? `PC${x.pc}` : (x.response || x.test), x.variable, x.test, x.status === "computed" ? x.value : x.reason, x.p, x.q, x.p_min_attainable, x.n_used, x.scheme]),
      { empty: "Nothing was tested." });
  }
  function details(f, file) {
    const m = f.measures || {}, p = f.plot_data || {}, plots = [], tables = [];
    const T = (title, node) => tables.push(el("h3", { text: title }), node);
    switch (f.audit_id) {
      case "A1": T("Most similar sample pairs", table(["a", "b", "r"], (m.top_sample_correlations || []).map((x) => [x.a, x.b, x.r])));
        T("Sample metadata completeness", table(["column", "kind", "missing", "levels"], (m.metadata || []).map((x) => [x.column, x.audit_kind, pct(x.share_missing), x.n_levels]))); break;
      case "A2": plots.push(histogram(p.value_histogram_log10, "All values", "value"), histogram(p.feature_skew_x, "Skewness per feature", "skewness")); break;
      case "A3": plots.push(scatter(p.mean_sd_x, "Mean vs SD per feature", { logx: true, logy: true, xtitle: "mean (log scale)" }));
        if (p.rle) T("Relative log expression per sample", table(["sample", "median", "IQR"], p.rle.map((x) => [x.sample, x.median, x.iqr]))); break;
      case "A4": plots.push(histogram(p.feature_floor_rate, "Share of samples at the floor, per feature", "share"), histogram(p.feature_missing_rate, "Missing share per feature", "share"));
        T(`By ${get(m, "by_stratum.source") === "feature_class" ? "feature class" : "block"}`, table(["stratum", "features", "floor-tied", "share", "missing"], (get(m, "by_stratum.strata") || []).map((s) => [s.stratum, s.n_features, s.n_floor_tied, pct(s.floor_tied_share), pct(s.missing_rate)])));
        T("Most frequent values", table(["value", "count", "share"], (m.most_frequent_values || []).map((x) => [x.value, x.count, pct(x.share)])));
        T("Per-sample rates against batch and design", assocTable(get(m, "associations.rows"))); break;
      case "A5": plots.push(histogram(p.detection_rate, "Detection rate per feature", "rate"));
        T("Group sizes", table(["variable", "counts", "smallest"], (m.cells || []).map((c) => [c.variable, Object.entries(c.counts).map(([k, v]) => `${k}: ${v}`).join(", "), c.smallest.n]))); break;
      case "A6": plots.push(histogram(p.qc_rsd, "QC RSD per feature (%)", "RSD %"));
        if (p.run_order_points && get(m, "run_order.lowess")) plots.push(scatter(p.run_order_points, `Sample median by ${m.run_order.variable}`, { line: m.run_order.lowess, xtitle: m.run_order.variable, r: 4 }));
        T("Per-sample quality", table(["sample", "median (log2)", "detected", "r with median profile"], (m.per_sample || []).map((x) => [x.sample, x.median_y, x.n_detected, x.r_median_profile]))); break;
      case "A7": plots.push(pcaPlot(p.pca, file)); (p.crosstabs || []).forEach((c) => plots.push(crosstab(c)));
        T("Structure", table(["batch", "design", "code", "Cramér's V"], (m.structure || []).map((s) => [s.batch, s.design, s.code || s.reason, s.cramers_v]), { empty: "No batch variable to cross with the design." }));
        T("Associations", assocTable(get(m, "associations.rows"))); break;
      case "A8": plots.push(timeline(p.timeline, get(m, "time_grid.unit")), histogram(p.icc_histogram, "How much each feature is driven by the subject (ICC)", "ICC"));
        if (m.time_grid) T("Samples per time", table(["time", "samples", "subjects"], Object.keys(m.time_grid.samples_per_time).map((t) => [t, m.time_grid.samples_per_time[t], m.time_grid.subjects_per_time[t]]))); break;
      case "A9": T("Levels", table(["variable", "levels"], (m.variables || []).map((v) => [v.variable, Object.entries(v.levels).map(([k, n]) => `${k}: ${n}`).join(", ")]))); T("Associations", assocTable(get(m, "associations.rows"))); break;
      case "A10": T("Layers", table(["dataset", "unit", "features", "samples", "total variance (log2)"], (m.layers || []).map((l) => [l.dataset, l.label, l.n_features, l.n_samples, l.total_variance_y])));
        T("Agreement between layers (RV)", table(["a", "b", "shared", "RV", "p"], (m.rv || []).map((r) => [r.a, r.b, r.n_shared, r.rv ?? r.reason, r.p]))); break;
      case "A11": T("Samples", table(["sample", "z distance", "z PC1", "z PC2", "flags"], (get(m, "samples.per_sample") || []).map((x) => Object.assign([x.sample, x.z_distance, x.z_pc1, x.z_pc2, (x.flags || []).join(", ")], { _flag: (x.flags || []).length > 0 }))));
        T("Largest value per stratum", table(["stratum", "max", "max / p99", "feature", "sample", "z"], (get(m, "by_stratum.strata") || []).map((s) => Object.assign([s.stratum, s.max, s.max_over_p99, s.feature, s.sample, s.z], { _flag: s.flagged }))));
        T("Most extreme cells", table(["feature", "sample", "value", "z"], (get(m, "cells.top") || []).slice(0, 20).map((c) => [c.feature, c.sample, c.value, c.z]))); break;
    }
    return { plots: plots.filter((x) => x && x.nodeType), tables };
  }
  function flatten(o, path = "", out = []) {
    if (out.length > 400) return out;
    if (o && typeof o === "object" && !Array.isArray(o)) { for (const [k, v] of Object.entries(o)) flatten(v, path ? `${path}.${k}` : k, out); }
    else if (Array.isArray(o) && o.length && typeof o[0] === "object") out.push([path, `${o.length} row(s)`]);
    else out.push([path, o]);
    return out;
  }
  const headline = (f) => f.indicators && f.indicators.length ? f.indicators[0].text
    : f.status_reason || (f.status === "computed" ? "Nothing stood out." : f.status.replace(/_/g, " "));
  function card(f, file) {
    const c = el("details", { class: "ar-card", id: `card-${f.audit_id}-${f.dataset}-${f.assay}`, "data-audit": f.audit_id,
      "data-search": `${f.audit_id} ${NAMES[f.audit_id] || ""} ${(f.indicators || []).map((i) => i.text).join(" ")} ${f.status_reason || ""}`.toLowerCase() });
    let built = false;
    const body = el("div", { class: "body" });
    c.addEventListener("toggle", () => {
      if (!c.open || built) return;
      built = true;                                    // build plots on first open: keeps the page light
      const { plots, tables } = details(f, file);
      const kv = keyMeasures(f);
      body.append(...[
        f.indicators && f.indicators.length > 1 ? el("ul", { class: "ar-ind" }, f.indicators.map((i) => el("li", {}, el("span", { class: "ar-tag", text: "•", "data-tip": i.code }), el("span", { text: i.text })))) : null,
        f.status_reason && f.indicators && f.indicators.length ? el("p", { class: "ar-note", text: f.status_reason }) : null,
        f.needs && f.needs.length ? el("p", { class: "ar-note" }, el("b", { text: "Needs: " }), f.needs.map((n) => `${n}${HOWTO[n] ? ` — ${HOWTO[n]}` : ""}`).join(" · ")) : null,
        kv.length ? el("div", { class: "ar-kv" }, kv.map(([k, v]) => el("div", {}, el("span", { text: k }), el("b", { text: v })))) : null,
        plots.length ? el("div", { class: "ar-plots" }, plots) : null,
        tables.length ? el("details", { class: "more" }, el("summary", { text: "Tables" }), tables) : null,
        el("details", { class: "more" }, el("summary", { text: "All numbers and method" }),
          table(["measure", "value"], flatten({ measures: f.measures, method: f.method, feeds: f.feeds })))].filter(Boolean));
    });
    const n = (f.indicators || []).length;
    c.append(el("summary", {},
      el("span", { class: "id", text: f.audit_id }),
      el("span", { class: "t" }, el("b", { text: NAMES[f.audit_id] || f.audit }), el("p", { text: headline(f) })),
      el("span", { class: "r" }, n > 1 ? el("span", { class: "ar-faint", text: `${n} notes` }) : null,
        el("span", { class: `ar-pill ${f.status}`, text: f.status === "computed" ? "done" : f.status.replace(/_/g, " ") }), icon(I.chev))), body);
    c.querySelector("summary svg").setAttribute("class", "chev");
    return c;
  }

  // ------------------------------------------------------------------ summary blocks
  function unitKey(f) { return f.dataset === "session" ? "session" : `${f.dataset}/${f.assay}`; }
  function dsName(b, did) { const d = (b.datasets || []).find((x) => x.dataset_id === did); return d ? d.name : did; }
  function kpis(b) {
    const u = b.manifest.scope.units || [];
    const nInd = b.findings.reduce((a, f) => a + (f.indicators || []).length, 0);
    const needs = collectNeeds(b).length;
    const mism = (b.manifest.declared_vs_observed || []).filter((r) => r.status === "mismatch").length;
    const tiles = [
      [b.datasets.length, b.datasets.length === 1 ? "dataset" : "datasets"],
      [Math.max(0, ...u.map((x) => x.n_samples)), "samples (largest dataset)"],
      [u.reduce((a, x) => a + x.n_features, 0).toLocaleString("en-US"), "features in total"],
      [b.findings.filter((f) => f.status === "computed").length + " / " + b.findings.length, "checks completed"],
      [nInd, "observations"],
      [needs + mism, "need your attention", needs + mism > 0],
    ];
    return el("div", { class: "ar-kpis" }, tiles.map(([v, l, attn], i) => el("div", { class: `ar-kpi${attn ? " attn" : ""}`, style: `animation-delay:${i * 50}ms` }, el("b", { text: String(v) }), el("span", { text: l }))));
  }
  function highlights(b, onJump) {
    const by = {};
    for (const f of b.findings) for (const i of f.indicators || []) (by[unitKey(f)] = by[unitKey(f)] || []).push({ f, i });
    const cols = Object.keys(by).sort((a, c) => (a === "session") - (c === "session") || a.localeCompare(c)).map((key) => {
      const rank = (x) => { const r = PRIORITY.indexOf(x.i.code); return r < 0 ? 999 : r; };
      const items = by[key].sort((a, c) => rank(a) - rank(c) || ORDER.indexOf(a.f.audit_id) - ORDER.indexOf(c.f.audit_id));
      const seen = {}, rows = [];
      for (const x of items) {                       // one line per kind of sentence, with a count
        if (seen[x.i.code]) { seen[x.i.code].n++; continue; }
        seen[x.i.code] = { ...x, n: 1 };
        rows.push(seen[x.i.code]);
      }
      const f0 = by[key][0].f;
      const title = key === "session" ? "All datasets" : `${f0.dataset} · ${dsName(b, f0.dataset)}`;
      const col = el("article", { class: "ar-hl-col" },
        el("header", {}, el("b", { text: title }), el("span", { class: "ar-faint", text: `${by[key].length} observation(s)` })),
        el("ul", { class: "ar-ind" }, rows.map((x, idx) => el("li", { class: idx >= 5 ? "more" : null },
          el("button", { type: "button", class: "ar-tag", text: x.f.audit_id, "data-tip": `${NAMES[x.f.audit_id]}: open the details`, onclick: () => onJump(x.f) }),
          el("span", {}, x.i.text, x.n > 1 ? el("span", { class: "ar-faint", text: ` (+${x.n - 1} similar)` }) : null)))));
      if (rows.length > 5) col.append(el("button", { type: "button", class: "ar-link", text: `Show ${rows.length - 5} more`, onclick: (e) => { col.classList.toggle("open"); e.target.textContent = col.classList.contains("open") ? "Show fewer" : `Show ${rows.length - 5} more`; } }));
      return col;
    });
    return cols.length ? el("div", { class: "ar-hl" }, cols) : el("p", { class: "ar-empty", text: "Nothing stood out in this run." });
  }
  function collectNeeds(b) {
    const out = [];
    for (const f of b.findings) {
      const ns = [...new Set(f.needs || [])];
      if (!ns.length && !["insufficient_metadata", "insufficient_data"].includes(f.status)) continue;
      out.push({ f, needs: ns, reason: f.status_reason });
    }
    return out;
  }
  function needsBlock(b, onJump) {
    const ns = collectNeeds(b);
    if (!ns.length) return null;
    return el("div", { class: "ar-needs" }, ns.map(({ f, needs, reason }) => el("div", { class: "ar-need" }, el("span", { class: "dot" }),
      el("div", {}, el("b", { text: `${f.dataset === "session" ? "Session" : f.dataset} · ${NAMES[f.audit_id]}` }),
        needs.length ? ` needs ${needs.join(", ")}.` : ` — ${reason}`,
        el("small", { text: needs.map((n) => HOWTO[n]).filter(Boolean).join(" ") || reason || "" }),
        el("button", { type: "button", class: "ar-link no-print", text: "Open", onclick: () => onJump(f) })))));
  }
  function dvoBlock(b) {
    const rows = b.manifest.declared_vs_observed || [];
    if (!rows.length) return null;
    const items = ["imputed", "normalized", "log_transformed"];
    const by = {};
    for (const r of rows) (by[`${r.dataset}/${r.assay}`] = by[`${r.dataset}/${r.assay}`] || {})[r.item] = r;
    const obs = (r) => Object.entries(r.observed || {}).map(([k, v]) => `${k.replace(/_/g, " ")}: ${fmt(v)}`).join("\n");
    const says = { mismatch: "does not match", consistent: "matches", not_declared: "not declared" };
    return el("div", { class: "ar-dvo" },
      el("div", { class: "ar-dvo-row head" }, el("div", { text: "dataset" }), items.map((i) => el("div", { text: i.replace(/_/g, " ") }))),
      Object.entries(by).map(([key, r]) => el("div", { class: "ar-dvo-row" },
        el("div", {}, el("b", { text: key.split("/")[0] }), ` ${dsName(b, key.split("/")[0])}`),
        items.map((i) => r[i] ? el("div", { class: `cell ${r[i].status}`, "data-tip": `Observed:\n${obs(r[i])}` },
          el("span", { class: `ar-pill ${r[i].status}`, text: says[r[i].status] || r[i].status }),
          el("small", { text: `declared: ${r[i].declared || "not answered"}` })) : el("div", { class: "cell" }, "—")))));
  }
  function reproBlock(b) {
    const m = b.manifest, s = m.scope, ov = s.overrides_applied || [];
    return el("details", { class: "more" }, el("summary", { text: "Reproducibility: run, parameters, versions, overrides" }),
      el("div", { class: "ar-kv" }, [["run", m.run_id], ["created", m.created_at], ["parameters hash", m.params_sha256.slice(0, 16)], ["seed", m.seed],
        ["PRISM", `${m.prism_version}${m.git_hash ? " · " + m.git_hash.slice(0, 8) : ""}`], ["Python / numpy", `${m.python} / ${m.numpy}`], ["time", `${fmt(m.timings_s.total)} s`]]
        .map(([k, v]) => el("div", {}, el("span", { text: k }), el("b", { text: String(v) })))),
      el("h3", { text: "What was used" }),
      table(["dataset", "unit", "samples", "features", "excluded", "non-study", "scale"], s.units.map((u) => [u.dataset, u.label, u.n_samples, u.n_features, (u.excluded_by_override || []).length, (u.non_study_samples || []).length, u.scale])),
      el("h3", { text: `Overrides applied (${ov.length})` }),
      table(["id", "kind", "target", "value", "reason", "by"], ov.map((o) => [o.override_id, o.kind, o.sample || o.column || o.key || "", o.role || (o.value != null ? o.value : ""), o.reason, o.by]), { empty: "None." }),
      Object.keys(s.params_changed || {}).length ? [el("h3", { text: "Parameters changed" }), table(["parameter", "default", "value", "source"], Object.entries(s.params_changed).map(([k, v]) => [k, v.default, v.value, v.source]))] : null,
      el("p", { class: "ar-note" }, "Rules of thumb (heuristics): ", (b.params_meta.heuristic || []).map((k) => `${k} = ${fmt(m.params[k])}`).join(" · ")));
  }

  // ------------------------------------------------------------------ the audit report
  function detailsBlock(b, root) {
    const byUnit = {};
    for (const f of b.findings) (byUnit[unitKey(f)] = byUnit[unitKey(f)] || []).push(f);
    const keys = Object.keys(byUnit).sort((a, c) => (a === "session") - (c === "session") || a.localeCompare(c));
    const fileOf = new Map(b.findings.map((f, i) => [f, (b.manifest.findings[i] || {}).file]));
    const label = (k) => k === "session" ? "All datasets" : `${byUnit[k][0].dataset} · ${dsName(b, byUnit[k][0].dataset)}`;
    let cur = keys[0], observer = null;
    const tabs = el("nav", { class: "ar-tabs no-print", role: "tablist" });
    const side = el("nav", { class: "ar-side", "aria-label": "Audits" });
    const cards = el("div", { class: "ar-cards" });
    const search = el("input", { class: "ar-search", type: "search", placeholder: "Search the findings…", "aria-label": "Search the findings" });
    const filter = () => {
      const q = search.value.trim().toLowerCase();
      cards.querySelectorAll(".ar-card").forEach((c) => c.classList.toggle("hidden-by-search", !!q && !c.dataset.search.includes(q)));
    };
    search.addEventListener("input", filter);
    const setAll = (open) => cards.querySelectorAll(".ar-card").forEach((c) => { c.open = open; });
    const show = (k) => {
      cur = k;
      tabs.replaceChildren(...keys.map((x) => el("button", { class: `ar-tab${x === cur ? " sel" : ""}`, type: "button", role: "tab", "aria-selected": String(x === cur), text: label(x), onclick: () => show(x) })));
      const fs = byUnit[k].slice().sort((a, c) => ORDER.indexOf(a.audit_id) - ORDER.indexOf(c.audit_id));
      cards.replaceChildren(...fs.map((f) => card(f, fileOf.get(f))));
      side.replaceChildren(...fs.map((f) => el("button", { type: "button", "data-for": `card-${f.audit_id}-${f.dataset}-${f.assay}`,
        onclick: () => openCard(f) }, el("span", { class: `sdot ${f.status}` }), `${f.audit_id} ${NAMES[f.audit_id]}`,
        (f.indicators || []).length ? el("span", { class: "cnt", text: String(f.indicators.length) }) : null)));
      filter();
      if (observer) observer.disconnect();
      if ("IntersectionObserver" in window) {                // scroll-spy for the side nav
        observer = new IntersectionObserver((es) => es.forEach((e) => {
          if (e.isIntersecting) side.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x.dataset.for === e.target.id));
        }), { rootMargin: "-40% 0px -55% 0px" });
        cards.querySelectorAll(".ar-card").forEach((c) => observer.observe(c));
      }
    };
    const openCard = (f) => {
      if (unitKey(f) !== cur) show(unitKey(f));
      const n = document.getElementById(`card-${f.audit_id}-${f.dataset}-${f.assay}`);
      if (!n) return null;
      n.open = true;
      n.scrollIntoView({ behavior: "smooth", block: "start" });
      n.classList.add("flash"); setTimeout(() => n.classList.remove("flash"), 1200);
      return n;
    };
    const block = el("section", { class: "ar-section", id: "ar-details" },
      el("header", {}, el("h2", { text: "Every check, in detail" }), el("span", { class: "ar-sub", text: "Open a card to see its plots and numbers." })),
      tabs,
      el("div", { class: "ar-toolbar no-print" }, search,
        el("button", { type: "button", class: "ar-link", text: "Open all", onclick: () => setAll(true) }),
        el("button", { type: "button", class: "ar-link", text: "Close all", onclick: () => setAll(false) })),
      el("div", { class: "ar-detail" }, side, cards));
    if (keys.length) show(cur); else cards.append(el("p", { class: "ar-empty", text: "No findings." }));
    const byFile = (file) => b.findings.find((f) => fileOf.get(f) === file);
    return { block, openCard, show, focus: (file) => { const f = byFile(file); return f ? openCard(f) : null; },
      setView: (v) => { const f = byFile(v.finding); if (f) openCard(f); const fn = VIEWS[v.finding]; return fn ? fn(v) : null; },
      openAll: () => setAll(true) };
  }

  function render(root, b, opts = {}) {
    root.classList.add("ar");
    tooltips(root);
    const det = detailsBlock(b, root);
    const jump = (f) => det.openCard(f);
    const needs = needsBlock(b, jump), dvo = dvoBlock(b);
    root.replaceChildren(opts.top || "",
      el("header", { class: "ar-hero" },
        el("div", {}, el("div", { class: "ar-eyebrow", text: `Audit · ${b.session.name}` }), el("h1", { text: "What the checks found" }),
          el("p", { class: "ar-sub", text: "Numbers and plain sentences only. Nothing in your data was changed, and no verdict is given." })),
        opts.actions || standaloneActions(root, det)),
      kpis(b),
      el("section", { class: "ar-section" }, el("header", {}, el("h2", { text: "Highlights" }), el("span", { class: "ar-sub", text: "The most important observations first; click a tag for the details." })), highlights(b, jump)),
      needs ? el("section", { class: "ar-section" }, el("header", {}, el("h2", { text: "Needs your input" })), needs) : null,
      dvo ? el("section", { class: "ar-section" }, el("header", {}, el("h2", { text: "What you declared vs what the data shows" }), el("span", { class: "ar-sub", text: "Hover a cell for the observed numbers." })), dvo) : null,
      det.block,
      el("section", { class: "ar-section" }, reproBlock(b)));
    return { showTab: det.show, focus: det.focus, setView: det.setView };
  }
  function themeButton() {
    const b = el("button", { type: "button", class: "ar-theme no-print", "aria-label": "Switch light / night mode" });
    const paint = () => { const dark = (document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark")) === "dark";
      b.replaceChildren(icon(dark ? I.sun : I.moon)); b.firstChild.setAttribute("width", "16"); b.firstChild.setAttribute("height", "16");
      b.firstChild.setAttribute("fill", "none"); b.firstChild.setAttribute("stroke", "currentColor"); b.firstChild.setAttribute("stroke-width", "1.7"); };
    b.addEventListener("click", () => {
      const r = document.documentElement, cur = r.dataset.theme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
      const go = () => { r.dataset.theme = cur === "dark" ? "light" : "dark"; try { localStorage.setItem("prism.theme", r.dataset.theme); } catch (_) {} paint(); };
      document.startViewTransition ? document.startViewTransition(go) : go();
    });
    paint();
    return b;
  }
  function standaloneActions(root, det) {
    return el("div", { class: "ar-actions no-print" }, themeButton(),
      el("button", { type: "button", class: "ar-btn", onclick: () => { det && det.openAll && det.openAll(); setTimeout(() => print(), 300); } }, icon(I.print), "Print / PDF"));
  }

  // ------------------------------------------------------------------ the final report
  function sentenceSummary(fb) {
    const ds = fb.datasets || [], au = fb.audit, mg = fb.merge;
    const ps = [];
    const desc = ds.map((d) => `${d.name} (${[d.omics_family, d.n_features != null ? `${fmt(d.n_features)} features` : null].filter(Boolean).join(", ")})`);
    ps.push(`This report covers ${ds.length} dataset${ds.length === 1 ? "" : "s"}: ${desc.join(" and ")}.`);
    const subj = ds.map((d) => d.design && d.design.n_subjects).find((x) => x);
    const ns = Math.max(0, ...ds.map((d) => d.n_samples || 0));
    ps.push(`There are ${ns} samples${subj ? ` from ${subj} subjects` : ""}${ds.some((d) => d.design && d.design.time_unit) ? `, measured over time (${ds.find((d) => d.design && d.design.time_unit).design.time_unit})` : ""}.`);
    const ov = get(mg, "cross_dataset.sample_overlap");
    if (ds.length > 1 && ov) ps.push(`${ov.n_in_all} of ${ov.n_unified} samples are present in every dataset${ov.n_in_all === ov.n_unified ? ", so the datasets line up completely" : ""}.`);
    if (au) {
      const nInd = au.findings.reduce((a, f) => a + (f.indicators || []).length, 0);
      const mism = (au.manifest.declared_vs_observed || []).filter((r) => r.status === "mismatch").length;
      const needs = collectNeeds(au).length;
      ps.push(`The audit ran ${au.findings.length} checks and made ${nInd} observations.` +
        (mism ? ` In ${mism} place${mism === 1 ? "" : "s"} what was declared about the processing does not match what the values show.` : "") +
        (needs ? ` ${needs} check${needs === 1 ? "" : "s"} need more information (see "Needs your input").` : ""));
    } else ps.push("No audit has been run yet.");
    return el("div", { class: "fr-summary" }, ps.map((p) => el("p", { text: p })));
  }
  function datasetCards(fb) {
    const H = { yes: "yes", no: "no", not_sure: "not sure" };
    return el("div", { class: "fr-ds" }, (fb.datasets || []).map((d) => el("article", {},
      el("header", {}, el("b", { text: `${d.dataset_id} · ${d.name}` }), " ", el("span", { class: "ar-pill computed", text: d.status })),
      el("dl", {},
        [["source file", d.source_file], ["omics", d.omics_family], ["layout", (d.layout || "").replace(/_/g, " ")],
          ["measurements", (d.assays || []).map((a) => `${a.label}: ${fmt(a.n_features)} × ${fmt(a.n_samples)}`).join("; ")],
          ["feature ID", d.feature_id], ["subject", get(d, "design.subject")], ["time", get(d, "design.time") ? `${d.design.time}${d.design.time_unit ? ` (${d.design.time_unit})` : ""}` : null],
          ["design", get(d, "design.label")], ["sample information", d.n_metadata_columns != null ? `${d.n_metadata_columns} column(s)` : null],
          ["processing declared", Object.entries(d.processing_history || {}).map(([k, v]) => `${k.replace(/_/g, " ")}: ${H[v] || v || "—"}`).join(" · ")],
          ["how it was confirmed", d.origin]]
          .filter(([, v]) => v != null && v !== "").map(([k, v]) => [el("dt", { text: k }), el("dd", { text: String(v) })])))));
  }
  function keyFigures(au) {
    if (!au) return null;
    const out = [];
    const byUnit = {};
    for (const f of au.findings) if (f.dataset !== "session") (byUnit[unitKey(f)] = byUnit[unitKey(f)] || {})[f.audit_id] = f;
    const fileOf = new Map(au.findings.map((f, i) => [f, (au.manifest.findings[i] || {}).file]));
    for (const [key, fs] of Object.entries(byUnit)) {
      const plots = [];
      if (get(fs, "A7.plot_data.pca")) plots.push(pcaPlot(fs.A7.plot_data.pca, "final:" + fileOf.get(fs.A7)));
      if (get(fs, "A4.plot_data.feature_floor_rate")) plots.push(histogram(fs.A4.plot_data.feature_floor_rate, "Share of samples at the floor, per feature", "share"));
      if (get(fs, "A8.plot_data.icc_histogram")) plots.push(histogram(fs.A8.plot_data.icc_histogram, "Subject-driven variation per feature (ICC)", "ICC"));
      const ok = plots.filter(Boolean);
      if (ok.length) out.push(el("h3", { text: `${key.split("/")[0]} · ${dsName(au, key.split("/")[0])}` }), el("div", { class: "ar-plots" }, ok));
    }
    return out.length ? out : null;
  }
  function renderFinal(root, fb, opts = {}) {
    root.classList.add("ar");
    tooltips(root);
    const au = fb.audit;
    const det = au ? detailsBlock(au, root) : null;
    const jump = (f) => det && det.openCard(f);
    const sections = [
      ["summary", "Summary"], ["highlights", "Key findings"], ["input", "Needs your input"], ["declared", "Declared vs observed"],
      ["datasets", "Datasets"], ["figures", "Key figures"], ["details", "All checks"], ["methods", "Methods"]];
    const needs = au ? needsBlock(au, jump) : null, dvo = au ? dvoBlock(au) : null, figs = keyFigures(au);
    const has = { input: !!needs, declared: !!dvo, highlights: !!au, figures: !!figs, details: !!au };
    const S = (id, title, sub, ...body) => el("section", { class: "ar-section fr-reveal", id: `fr-${id}` },
      el("header", {}, el("h2", { text: title }), sub ? el("span", { class: "ar-sub", text: sub }) : null), body);
    const progress = el("div", { class: "fr-progress", "aria-hidden": "true" });
    root.replaceChildren(progress,
      el("header", { class: "fr-cover" },
        el("div", { class: "ar-hero" },
          el("div", {}, el("div", { class: "ar-eyebrow", text: "PRISM · final report" }), el("h1", { text: fb.session.name || "Study session" }),
            el("p", { class: "ar-sub", text: `Generated ${fb.generated_at.slice(0, 16).replace("T", " ")} UTC${au ? ` · audit run ${au.manifest.run_id}` : ""} · PRISM ${fb.prism_version}` })),
          opts.actions || standaloneActions(root, det)),
        au ? kpis(au) : null,
        el("nav", { class: "fr-toc no-print" }, sections.filter(([id]) => has[id] !== false).map(([id, t]) => el("a", { href: `#fr-${id}`, text: t,
          onclick: (e) => { e.preventDefault(); document.getElementById(`fr-${id}`).scrollIntoView({ behavior: "smooth" }); } })))),
      S("summary", "Summary", null, sentenceSummary(fb)),
      au ? S("highlights", "Key findings", "The most important observations per dataset.", highlights(au, jump)) : null,
      needs ? S("input", "Needs your input", null, needs) : null,
      dvo ? S("declared", "Declared vs observed", "What the Step 0 history says next to what the values show.", dvo) : null,
      S("datasets", "Datasets", "How each dataset was read and confirmed.", datasetCards(fb)),
      figs ? S("figures", "Key figures", "Interactive here; static in print.", figs) : null,
      det ? el("div", { class: "fr-page fr-reveal", id: "fr-details" }, det.block) : null,
      el("section", { class: "ar-section fr-reveal", id: "fr-methods" }, el("header", {}, el("h2", { text: "Methods" })),
        el("p", { class: "ar-note", text: "Step 0 recognized each table and you confirmed every step. The Tier 1 audit is deterministic: the same files, parameters and seed give the same numbers. It reads only the confirmed output folders and your overrides; it never changes, filters or imputes values. Sentences come from fixed templates; p-values use permutations that respect the design (whole subjects, within subject, or free) and are BH-adjusted within each check." }),
        au ? reproBlock(au) : null));
    const reveal = () => root.querySelectorAll(".fr-reveal:not(.in)").forEach((n) => { if (n.getBoundingClientRect().top < innerHeight - 40) n.classList.add("in"); });
    const onScroll = () => {
      const h = document.documentElement, total = h.scrollHeight - h.clientHeight;
      progress.style.width = total > 0 ? `${Math.min(100, 100 * h.scrollTop / total)}%` : "0";
      reveal();
    };
    addEventListener("scroll", onScroll, { passive: true });
    addEventListener("beforeprint", () => { root.querySelectorAll(".fr-reveal").forEach((n) => n.classList.add("in")); det && det.openAll(); });
    requestAnimationFrame(onScroll);
    return det ? { showTab: det.show, focus: det.focus, setView: det.setView } : {};
  }

  function diffPanel(cmp) {
    const f = cmp.findings || [];
    return el("div", { class: "ar ar-diff" }, el("b", { text: `Compared with ${cmp.a}: ` }),
      Object.keys(cmp.params_changed).length ? `parameters ${Object.entries(cmp.params_changed).map(([k, v]) => `${k} ${fmt(v.a)} → ${fmt(v.b)}`).join(", ")}; ` : "same parameters; ",
      cmp.overrides_changed ? "overrides changed; " : "same overrides; ",
      cmp.findings_identical ? "identical findings." : `${f.length} finding(s) differ.`,
      f.length ? el("details", { class: "more" }, el("summary", { text: "What changed" }), el("ul", {}, f.slice(0, 30).map((x) => el("li", { text: `${x.finding}: ${x.only_in ? `only in ${x.only_in}` : [x.status ? `status ${x.status.join(" → ")}` : null, x.indicators_added && x.indicators_added.length ? `+ ${x.indicators_added.join(", ")}` : null, x.indicators_removed && x.indicators_removed.length ? `− ${x.indicators_removed.join(", ")}` : null, x.n_measures_changed ? `${x.n_measures_changed} number(s) changed` : "method parameters only"].filter(Boolean).join("; ")}` })))) : null);
  }
  window.PRISM_REPORT = { render, renderFinal, diffPanel, fmt };
})();
