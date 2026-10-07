// PRISM Tier 1 audit report renderer (v3 §7). Shared by the live page (audit.js) and the
// self-contained HTML export (prism/audit/report.py inlines this file and the data).
// Plain JS + SVG; no libraries, no network. Numbers and sentences come from the findings only.

(() => {
  "use strict";
  const NAMES = { A1: "Integrity", A2: "Data type and scale", A3: "Distribution and variance-mean", A4: "Missingness and floor values",
    A5: "Dimensionality and effective n", A6: "Technical noise and QC", A7: "Batch structure", A8: "Repeated measures",
    A9: "Sample source", A10: "Multi-omics overlap", A11: "Outlier flags" };
  const ORDER = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A11"];
  const PALETTE = ["#d9c7a3", "#ff8a2a", "#8fb3c9", "#b5d38a", "#c99ad6", "#e0e0e0", "#d1736b", "#7fc7b5", "#a68d5c", "#6f7fbf"];
  const SVGNS = "http://www.w3.org/2000/svg";

  function el(tag, attrs = {}, ...children) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
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
      if (a >= 0.001) return String(+v.toPrecision(4));
      return v.toExponential(2);
    }
    if (Array.isArray(v)) return v.length <= 6 ? v.map(fmt).join(", ") : `${v.slice(0, 6).map(fmt).join(", ")} … (${v.length})`;
    if (typeof v === "object") return Object.entries(v).slice(0, 12).map(([k, x]) => `${k}: ${typeof x === "object" && x ? JSON.stringify(x) : fmt(x)}`).join(", ");
    return String(v);
  }
  const pct = (v) => v == null ? "—" : `${+(100 * v).toPrecision(3)}%`;
  const get = (o, path) => path.split(".").reduce((x, k) => (x == null ? x : x[k]), o);

  // ------------------------------------------------------------------ plots
  const W = 440, H = 220, PAD = { l: 44, r: 12, t: 10, b: 30 };
  function axes(g, x0, x1, y0, y1, sx, sy, opts = {}) {
    const ticks = (a, b, n = 4) => { const out = []; for (let i = 0; i <= n; i++) out.push(a + (b - a) * i / n); return out; };
    for (const t of ticks(y0, y1)) {
      g.append(svg("line", { class: "ar-grid", x1: PAD.l, x2: W - PAD.r, y1: sy(t), y2: sy(t) }));
      g.append(svg("text", { x: PAD.l - 4, y: sy(t) + 3, "text-anchor": "end" }, opts.ylab ? opts.ylab(t) : short(t)));
    }
    for (const t of ticks(x0, x1, 3)) g.append(svg("text", { x: sx(t), y: H - PAD.b + 12, "text-anchor": "middle" }, opts.xlab ? opts.xlab(t) : short(t)));
    g.append(svg("line", { class: "ar-axis", x1: PAD.l, x2: W - PAD.r, y1: H - PAD.b, y2: H - PAD.b }));
    g.append(svg("line", { class: "ar-axis", x1: PAD.l, x2: PAD.l, y1: PAD.t, y2: H - PAD.b }));
    if (opts.xtitle) g.append(svg("text", { x: (PAD.l + W - PAD.r) / 2, y: H - 2, "text-anchor": "middle" }, opts.xtitle));
  }
  function short(v) {
    const a = Math.abs(v);
    if (a >= 1e4 || (a > 0 && a < 1e-2)) return v.toExponential(0);
    return String(+v.toPrecision(3));
  }
  function plotBox(title, node, extra) {
    return el("div", { class: "ar-plot" }, el("div", { class: "ar-plot-title" }, el("span", { text: title }), extra || null), node);
  }
  function histogram(h, title, xtitle) {
    if (!h || !h.counts || !h.counts.length) return null;
    const e = h.edges, c = h.counts, ymax = Math.max(...c, 1);
    const sx = (v) => PAD.l + (v - e[0]) / ((e[e.length - 1] - e[0]) || 1) * (W - PAD.l - PAD.r);
    const sy = (v) => H - PAD.b - v / ymax * (H - PAD.t - PAD.b);
    const g = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title });
    axes(g, e[0], e[e.length - 1], 0, ymax, sx, sy, { xtitle: h.log10 ? `${xtitle || "value"} (log10)` : xtitle, ylab: (t) => String(Math.round(t)) });
    c.forEach((n, i) => {
      const x = sx(e[i]), w = Math.max(1, sx(e[i + 1]) - x - 1);
      const b = svg("rect", { class: "ar-bar", x, y: sy(n), width: w, height: H - PAD.b - sy(n) });
      b.append(svg("title", {}, `${short(e[i])} to ${short(e[i + 1])}: ${n}`));
      g.append(b);
    });
    return plotBox(title, g);
  }
  function scatter(points, title, opts = {}) {
    if (!points || !points.length) return null;
    const tx = opts.logx ? (v) => Math.log10(v) : (v) => v, ty = opts.logy ? (v) => Math.log10(v) : (v) => v;
    const P = points.filter((p) => isFinite(tx(p[0])) && isFinite(ty(p[1]))).map((p) => [tx(p[0]), ty(p[1]), p[2], p[3]]);
    if (!P.length) return null;
    const xs = P.map((p) => p[0]), ys = P.map((p) => p[1]);
    let [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    if (x0 === x1) { x0 -= 1; x1 += 1; } if (y0 === y1) { y0 -= 1; y1 += 1; }
    const sx = (v) => PAD.l + (v - x0) / (x1 - x0) * (W - PAD.l - PAD.r), sy = (v) => H - PAD.b - (v - y0) / (y1 - y0) * (H - PAD.t - PAD.b);
    const g = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title });
    axes(g, x0, x1, y0, y1, sx, sy, { xtitle: opts.xtitle, xlab: opts.logx ? (t) => `1e${short(t)}` : null, ylab: opts.logy ? (t) => `1e${short(t)}` : null });
    for (const p of P) {
      const d = svg("circle", { class: "ar-dot", cx: sx(p[0]), cy: sy(p[1]), r: opts.r || 2, style: p[3] ? `fill:${p[3]}` : null });
      if (p[2]) d.append(svg("title", {}, p[2]));
      g.append(d);
    }
    if (opts.line) {
      const L = opts.line.filter((p) => isFinite(p[0]) && isFinite(p[1]));
      g.append(svg("polyline", { class: "ar-line", points: L.map((p) => `${sx(p[0])},${sy(p[1])}`).join(" ") }));
    }
    return { node: g, box: plotBox(title, g, opts.extra) };
  }
  const VIEWS = {};
  function pcaPlot(pca, key) {
    if (!pca || !pca.samples || !pca.samples.length) return null;
    const vars = Object.keys(pca.samples[0].values || {});
    const k = pca.variance_explained.length;
    const state = { color: vars[0] || "", x: 0, y: Math.min(1, k - 1) };
    const wrap = el("div", { class: "ar-plot" });
    const legend = el("div", { class: "ar-legend" });
    const sel = (opts, cur, on) => el("select", { onchange: (e) => on(e.target.value) }, opts.map(([v, l]) => el("option", { value: v, selected: String(v) === String(cur) ? "selected" : null, text: l })));
    const draw = () => {
      const levels = [...new Set(pca.samples.map((s) => (s.values || {})[state.color] || ""))].sort();
      const col = (v) => PALETTE[levels.indexOf(v) % PALETTE.length];
      const pts = pca.samples.map((s) => [s.scores[state.x], s.scores[state.y], `${s.sample}${state.color ? ` · ${state.color}: ${(s.values || {})[state.color] || "—"}` : ""}`, state.color ? col((s.values || {})[state.color] || "") : null]);
      const pcs = Array.from({ length: k }, (_, i) => [i, `PC${i + 1} (${pct(pca.variance_explained[i])})`]);
      const res = scatter(pts.map((p) => [p[0], p[1], p[2], p[3]]), "", { r: 3.2, xtitle: `PC${state.x + 1} vs PC${state.y + 1}` });
      wrap.replaceChildren(el("div", { class: "ar-plot-title" }, el("span", { text: "PCA of Y" }),
        "x", sel(pcs, state.x, (v) => { state.x = +v; draw(); }), "y", sel(pcs, state.y, (v) => { state.y = +v; draw(); }),
        vars.length ? ["colour by", sel(vars.map((v) => [v, v]), state.color, (v) => { state.color = v; draw(); })] : null),
        res ? res.node : el("div", { class: "ar-empty", text: "No scores." }), legend);
      legend.replaceChildren(...(state.color ? levels.slice(0, 20).map((l) => el("span", {}, el("i", { style: `background:${col(l)}` }), l || "(empty)")) : []));
    };
    draw();
    if (key) VIEWS[key] = (v) => { if (v.color_by != null) state.color = v.color_by; if (v.pcs) { state.x = v.pcs[0] - 1; state.y = v.pcs[1] - 1; } draw(); return wrap; };
    return wrap;
  }
  function timeline(rows, unit) {
    if (!rows || !rows.length) return null;
    const subs = [...new Set(rows.map((r) => r.subject || "(none)"))].sort();
    const num = rows.every((r) => r.time !== "" && r.time != null && isFinite(+r.time));
    if (!num) return null;
    const ts = rows.map((r) => +r.time), t0 = Math.min(...ts), t1 = Math.max(...ts) || 1;
    const h = Math.max(H, 18 * subs.length + PAD.t + PAD.b);
    const sx = (v) => 70 + (v - t0) / ((t1 - t0) || 1) * (W - 70 - PAD.r);
    const sy = (s) => PAD.t + 9 + subs.indexOf(s) * ((h - PAD.t - PAD.b - 9) / Math.max(1, subs.length - 1 || 1));
    const g = svg("svg", { viewBox: `0 0 ${W} ${h}`, role: "img", "aria-label": "subject by time" });
    for (const s of subs) {
      const y = sy(s);
      g.append(svg("line", { class: "ar-grid", x1: 70, x2: W - PAD.r, y1: y, y2: y }));
      g.append(svg("text", { x: 66, y: y + 3, "text-anchor": "end" }, s));
    }
    for (const r of rows) {
      const d = svg("circle", { class: "ar-dot", cx: sx(+r.time), cy: sy(r.subject || "(none)"), r: 3.2 });
      d.append(svg("title", {}, `${r.sample}: ${r.subject} at ${r.time}`));
      g.append(d);
    }
    for (let i = 0; i <= 4; i++) { const t = t0 + (t1 - t0) * i / 4; g.append(svg("text", { x: sx(t), y: h - PAD.b + 12, "text-anchor": "middle" }, short(t))); }
    g.append(svg("text", { x: (70 + W) / 2, y: h - 2, "text-anchor": "middle" }, `time${unit ? ` (${unit})` : ""}`));
    return plotBox("Subjects over time", g);
  }
  function table(head, rows, opts = {}) {
    if (!rows || !rows.length) return el("div", { class: "ar-empty", text: opts.empty || "None." });
    return el("div", { class: "ar-table-wrap" }, el("table", {}, el("thead", {}, el("tr", {}, head.map((h) => el("th", { text: h })))),
      el("tbody", {}, rows.map((r) => el("tr", { class: r._flag ? "ar-flag" : null }, r.filter((_, i) => i < head.length).map((c) => el("td", { class: typeof c === "number" ? "num" : null, text: fmt(c) })))))));
  }
  function crosstab(c) {
    const head = [`${c.batch} \\ ${c.design}`, ...c.design_levels];
    return el("div", { class: "ar-plot" }, el("div", { class: "ar-plot-title", text: `Cross-tab: ${c.batch} × ${c.design}` }),
      table(head, c.table.map((row, i) => [c.batch_levels[i], ...row])));
  }

  // ------------------------------------------------------------------ per-audit summaries
  function keyMeasures(f) {
    const m = f.measures || {}, out = [];
    const add = (label, v) => { if (v !== undefined) out.push([label, v]); };
    switch (f.audit_id) {
      case "A1": add("cells", get(m, "cells.n")); add("missing", get(m, "cells.missing")); add("infinite", get(m, "cells.infinite"));
        add("negative", get(m, "cells.negative")); add("zero", get(m, "cells.zero")); add("constant features", get(m, "constant_features.n"));
        add("features < 3 observed", get(m, "features_fewer_than_3_observed.n")); add("duplicate sample groups", (m.duplicate_samples || []).length);
        add("unparsable cells", get(m, "ids.n_unparsable_cells")); break;
      case "A2": add("classification", m.classification); add("median skew X", get(m, "skewness.median_x")); add("median skew Y", get(m, "skewness.median_y"));
        add("MAD log2 feature medians", get(m, "median_scaling.mad_log2_feature_medians")); add("CV of sample sums", m.cv_sample_sums);
        add("log10(p99/p1)", get(m, "range.log10_p99_over_p1")); add("max / p99", get(m, "range.max_over_p99")); break;
      case "A3": add("log-log slope (X)", get(m, "mean_sd_x.slope")); add("95% CI", get(m, "mean_sd_x.ci95")); add("Spearman mean-SD (Y)", get(m, "mean_sd_y.spearman_rho"));
        add("median skew X", get(m, "skewness.x.median")); add("share |skew| > 1 (X)", get(m, "skewness.x.share_abs_gt_1"));
        add("MAD of sample medians (RLE)", get(m, "rle.mad_of_sample_medians")); break;
      case "A4": add("missing cells", get(m, "missing.n")); add("missing rate", get(m, "missing.rate")); add("zeros", get(m, "zeros.n"));
        add("floor-tied features", `${fmt(get(m, "floor_ties.n_features"))} of ${fmt(get(m, "floor_ties.of"))}`); add("floor-tied share", get(m, "floor_ties.share"));
        add("ties / observed (p50)", get(m, "floor_ties.ties_over_observed.p50")); add("rho floor vs abundance", get(m, "abundance_dependence.floor.rho"));
        add("rho missing vs abundance", get(m, "abundance_dependence.missing.rho")); add("tests (BH)", get(m, "associations.n_tests")); break;
      case "A5": add("n", m.n); add("p", m.p); add("n/p", m.n_over_p); add("subjects", m.n_subjects); add("n_eff", get(m, "design_effect.n_eff"));
        add("design effect", get(m, "design_effect.deff")); add("smallest cell", m.smallest_cell ? `${m.smallest_cell.n} (${m.smallest_cell.variable} = ${m.smallest_cell.level})` : null);
        add("sparsity", get(m, "sparsity.share_missing_zero_or_floor")); add("detection rate p10", get(m, "detection_rate.p10")); break;
      case "A6": add("roles", get(m, "roles.counts")); add("QC median RSD %", get(m, "qc_rsd.summary.median")); add("run-order rho", get(m, "run_order.spearman_median_y"));
        add("LOWESS range", get(m, "run_order.lowess_range")); add("suspect features", get(m, "suspect_rows.flagged.n_features")); break;
      case "A7": add("complete features", get(m, "pca.n_features_complete")); add("PCs", get(m, "pca.k")); add("variance explained", get(m, "pca.variance_explained"));
        add("tests (BH)", get(m, "associations.n_tests")); add("batch candidates", (get(m, "variables.B") || []).map((v) => v.name)); break;
      case "A8": add("subjects", get(m, "design.n_subjects")); add("cluster sizes", get(m, "design.subjects_per_cluster_size")); add("balanced", get(m, "design.balanced"));
        add("paired", get(m, "design.paired")); add("median ICC", get(m, "icc.raw.median")); add("ICC IQR", get(m, "icc.iqr")); add("share ICC > cut", get(m, "icc.share_above.share"));
        add("r same subject", get(m, "sample_correlation.same_subject_mean_r")); add("r different subjects", get(m, "sample_correlation.different_subject_mean_r"));
        add("gaps (min / median / max)", m.time_grid ? [m.time_grid.gap_min, m.time_grid.gap_median, m.time_grid.gap_max] : null); break;
      case "A9": add("variables", (m.variables || []).map((v) => v.variable)); add("tests (BH)", get(m, "associations.n_tests")); break;
      case "A10": add("unified samples", get(m, "overlap.n_unified")); add("in every dataset", get(m, "overlap.n_in_all")); add("open ID suggestions", get(m, "overlap.open_id_suggestions")); break;
      case "A11": add("flagged samples", get(m, "samples.flagged")); add("flagged cells", get(m, "cells.n_flagged")); add("features with flagged cells", get(m, "cells.n_features_with_any"));
        add("cut-off", get(m, "cells.cutoff")); break;
    }
    return out.filter(([, v]) => v !== undefined);
  }
  function associationTable(rows) {
    const r = (rows || []).map((x) => [x.pc ? `PC${x.pc}` : (x.response || x.test), x.variable, x.test, x.status === "computed" ? x.value : x.reason, x.p, x.q, x.p_min_attainable, x.n_used, x.scheme]);
    return table(["on", "variable", "test", "statistic", "p", "q (BH)", "p min", "n", "permutations"], r, { empty: "No association tested." });
  }
  function details(f, file) {
    const m = f.measures || {}, p = f.plot_data || {}, plots = [], extra = [];
    switch (f.audit_id) {
      case "A1": extra.push(el("h3", { text: "Most correlated sample pairs" }), table(["a", "b", "r"], (m.top_sample_correlations || []).map((x) => [x.a, x.b, x.r])));
        extra.push(el("h3", { text: "Sample metadata completeness" }), table(["column", "kind", "missing", "levels", "constant", "all unique"], (m.metadata || []).map((x) => [x.column, x.audit_kind, pct(x.share_missing), x.n_levels, x.constant, x.all_unique])));
        break;
      case "A2": plots.push(histogram(p.value_histogram_log10, "Values (log scale)", "value"), histogram(p.feature_skew_x, "Per-feature skewness (X)", "skewness")); break;
      case "A3": { const s = scatter(p.mean_sd_x, "Mean vs SD per feature (X, log-log)", { logx: true, logy: true, xtitle: "mean" }); if (s) plots.push(s.box);
        if (p.rle) extra.push(el("h3", { text: "Relative log expression per sample" }), table(["sample", "median", "IQR"], p.rle.map((x) => [x.sample, x.median, x.iqr])));
        break; }
      case "A4": plots.push(histogram(p.feature_missing_rate, "Missing rate per feature", "rate"), histogram(p.feature_floor_rate, "Floor rate per feature", "rate"));
        extra.push(el("h3", { text: `By stratum (${get(m, "by_stratum.source") || ""})` }), table(["stratum", "features", "floor-tied", "share", "missing rate"], (get(m, "by_stratum.strata") || []).map((s) => [s.stratum, s.n_features, s.n_floor_tied, pct(s.floor_tied_share), pct(s.missing_rate)])));
        extra.push(el("h3", { text: "Most frequent values" }), table(["value", "count", "share"], (m.most_frequent_values || []).map((x) => [x.value, x.count, pct(x.share)])));
        extra.push(el("h3", { text: "Per-sample rates against B and G" }), associationTable(get(m, "associations.rows")));
        break;
      case "A5": plots.push(histogram(p.detection_rate, "Detection rate per feature", "rate"));
        extra.push(el("h3", { text: "Cells" }), table(["variable", "role", "counts", "smallest"], (m.cells || []).map((c) => [c.variable, c.role, Object.entries(c.counts).map(([k, v]) => `${k}: ${v}`).join(", "), c.smallest.n])));
        break;
      case "A6": plots.push(histogram(p.qc_rsd, "QC RSD per feature (%)", "RSD %"));
        if (p.run_order_points && get(m, "run_order.lowess")) {
          const s = scatter(p.run_order_points, `Per-sample median (Y) by ${m.run_order.variable}, with LOWESS`, { line: m.run_order.lowess, xtitle: m.run_order.variable, r: 3 });
          if (s) plots.push(s.box);
        }
        extra.push(el("h3", { text: "Per-sample quality" }), table(["sample", "median Y", "detected", "r with median profile"], (m.per_sample || []).map((x) => [x.sample, x.median_y, x.n_detected, x.r_median_profile])));
        if (m.suspect_rows) extra.push(el("h3", { text: "Rows marked suspect in Step 0" }), table(["group", "features", "median Y", "missing", "floor"], [["flagged", m.suspect_rows.flagged], ["unflagged", m.suspect_rows.unflagged]].filter(([, g]) => g).map(([k, g]) => [k, g.n_features, g.median_y, pct(g.missing_rate), pct(g.floor_rate)])));
        break;
      case "A7": plots.push(pcaPlot(p.pca, file)); (p.crosstabs || []).forEach((c) => plots.push(crosstab(c)));
        extra.push(el("h3", { text: "Structure (each batch candidate against each design variable)" }), table(["batch", "design", "code", "Cramér's V", "n"], (m.structure || []).map((s) => [s.batch, s.design, s.code || s.reason, s.cramers_v, s.n_used])));
        extra.push(el("h3", { text: "PC and PERMANOVA associations" }), associationTable(get(m, "associations.rows")));
        break;
      case "A8": plots.push(timeline(p.timeline, get(m, "time_grid.unit")), histogram(p.icc_histogram, "Feature ICC(1)", "ICC"));
        if (m.time_grid) extra.push(el("h3", { text: "Samples per time" }), table(["time", "samples", "subjects"], Object.keys(m.time_grid.samples_per_time).map((t) => [t, m.time_grid.samples_per_time[t], m.time_grid.subjects_per_time[t]])));
        extra.push(el("h3", { text: "Nesting" }), table(["inner", "nested within"], (m.nested || []).map((x) => [x.inner, x.outer]), { empty: "No nesting among categorical variables." }));
        break;
      case "A9": extra.push(table(["variable", "levels", "constant within subject"], (m.variables || []).map((v) => [v.variable, Object.entries(v.levels).map(([k, n]) => `${k}: ${n}`).join(", "), v.constant_within_subject])),
        el("h3", { text: "Associations" }), associationTable(get(m, "associations.rows"))); break;
      case "A10": extra.push(el("h3", { text: "Layers" }), table(["dataset", "unit", "features", "samples", "median spread (MAD log2)", "total variance Y"], (m.layers || []).map((l) => [l.dataset, l.label, l.n_features, l.n_samples, l.median_spread_mad_log2, l.total_variance_y])),
        el("h3", { text: "RV coefficient between layers (shared samples)" }), table(["a", "b", "shared", "RV", "p", "p min"], (m.rv || []).map((r) => [r.a, r.b, r.n_shared, r.rv ?? r.reason, r.p, r.p_min_attainable])),
        el("h3", { text: "Overlap" }), table(["pair", "shared", "only first", "only second", "Jaccard"], (get(m, "overlap.pairs") || []).map((x) => [`${x.a} · ${x.b}`, x.n_shared, x.n_only_a, x.n_only_b, x.jaccard])));
        break;
      case "A11": extra.push(el("h3", { text: "Samples" }), table(["sample", "z distance", "z PC1", "z PC2", "r with median", "flags"], (get(m, "samples.per_sample") || []).map((x) => Object.assign([x.sample, x.z_distance, x.z_pc1, x.z_pc2, x.r_median, (x.flags || []).join(", ")], { _flag: (x.flags || []).length > 0 }))),
        el("h3", { text: "Largest value per stratum" }), table(["stratum", "max", "p99", "max/p99", "feature", "sample", "z"], (get(m, "by_stratum.strata") || []).map((s) => Object.assign([s.stratum, s.max, s.p99, s.max_over_p99, s.feature, s.sample, s.z], { _flag: s.flagged }))),
        el("h3", { text: "Most extreme cells" }), table(["feature", "sample", "value", "z", "stratum"], (get(m, "cells.top") || []).slice(0, 25).map((c) => [c.feature, c.sample, c.value, c.z, c.stratum])));
        break;
    }
    return { plots: plots.filter((x) => x && x.nodeType), extra };
  }
  function flatten(o, path = "", out = []) {
    if (out.length > 400) return out;
    if (o && typeof o === "object" && !Array.isArray(o)) { for (const [k, v] of Object.entries(o)) flatten(v, path ? `${path}.${k}` : k, out); }
    else if (Array.isArray(o) && o.length && typeof o[0] === "object") out.push([path, `${o.length} row(s)`]);
    else out.push([path, o]);
    return out;
  }
  function card(f, file) {
    const { plots, extra } = details(f, file);
    return el("section", { class: "ar-card", id: `card-${f.audit_id}-${f.dataset}-${f.assay}` },
      el("div", { class: "ar-card-head" }, el("h2", { text: `${f.audit_id} · ${NAMES[f.audit_id] || f.audit}` }),
        el("span", { class: `ar-pill ${f.status}`, text: f.status.replace(/_/g, " ") }),
        el("span", { class: "ar-sub", text: `n used ${fmt(f.method && f.method.n_used)}${f.method && f.method.transform ? ` · Y: ${f.method.transform.transform}${f.method.transform.c != null ? ` (c = ${fmt(f.method.transform.c)})` : ""}` : ""}` })),
      f.status_reason ? el("p", { class: "ar-note", text: f.status_reason }) : null,
      f.indicators && f.indicators.length ? el("ul", { class: "ar-ind" }, f.indicators.map((i) => el("li", {}, i.text, el("span", { class: "ar-code", text: i.code })))) : el("p", { class: "ar-empty", text: "No indicator sentence." }),
      f.needs && f.needs.length ? el("div", { class: "ar-needs", text: `Needs: ${f.needs.join(", ")}` }) : null,
      keyMeasures(f).length ? el("div", { class: "ar-kv" }, keyMeasures(f).map(([k, v]) => el("div", {}, el("span", { text: k }), el("span", { text: typeof v === "number" && v >= 0 && v <= 1 && /share|rate|sparsity/.test(k) ? pct(v) : fmt(v) })))) : null,
      plots.length ? el("div", { class: "ar-plots" }, plots) : null,
      extra.length ? el("div", {}, extra) : null,
      el("details", {}, el("summary", { text: "All measures and method" }),
        table(["measure", "value"], flatten({ measures: f.measures, method: f.method, feeds: f.feeds }))));
  }

  // ------------------------------------------------------------------ page
  function header(b) {
    const m = b.manifest, s = m.scope;
    const ov = s.overrides_applied || [];
    return el("header", { class: "ar-head" },
      el("h1", { text: "Tier 1 audit" }),
      el("div", { class: "ar-sub", text: "Numbers and templated indicator sentences only: nothing was changed, filtered or removed, and no verdict is given." }),
      el("div", { class: "ar-meta" },
        el("div", {}, el("span", { text: "Session" }), `${b.session.name} `, el("code", { text: b.session.session_id })),
        el("div", {}, el("span", { text: "Run" }), el("code", { text: m.run_id })),
        el("div", {}, el("span", { text: "Created" }), m.created_at),
        el("div", {}, el("span", { text: "Parameters" }), el("code", { text: m.params_sha256.slice(0, 12) }), ` · seed ${m.seed}`),
        el("div", {}, el("span", { text: "PRISM" }), `${m.prism_version}${m.git_hash ? ` · ${m.git_hash.slice(0, 8)}` : ""} · numpy ${m.numpy}`),
        el("div", {}, el("span", { text: "Time" }), `${fmt(m.timings_s.total)} s`)),
      el("h3", { text: "Datasets and what was used" }),
      table(["dataset", "unit", "samples used", "features", "excluded by override", "non-study samples", "scale"],
        s.units.map((u) => [`${u.dataset} · ${(b.datasets.find((d) => d.dataset_id === u.dataset) || {}).name || ""}`, u.label, u.n_samples, u.n_features, u.excluded_by_override, u.non_study_samples, u.scale])),
      el("h3", { text: `Overrides applied (${ov.length})` }),
      table(["id", "kind", "target", "value", "reason", "by", "at"], ov.map((o) => [o.override_id, o.kind, o.sample || o.column || o.key || "", o.role || (o.value != null ? o.value : ""), o.reason, o.by, o.at]), { empty: "None." }),
      Object.keys(s.params_changed || {}).length ? [el("h3", { text: "Parameters changed from the defaults" }),
        table(["parameter", "default", "value", "source"], Object.entries(s.params_changed).map(([k, v]) => [k, v.default, v.value, v.source]))] : null,
      el("p", { class: "ar-note" }, "Heuristic thresholds: ", (b.params_meta.heuristic || []).map((k) => el("span", { class: "ar-pill heuristic", text: `${k} = ${fmt(m.params[k])}` })), " — rules of thumb, labelled as such."));
  }
  function dvoPanel(b) {
    const rows = b.manifest.declared_vs_observed || [];
    const obs = (r) => Object.entries(r.observed || {}).map(([k, v]) => `${k.replace(/_/g, " ")}: ${fmt(v)}`).join("; ");
    return el("section", { class: "ar-panel ar-dvo" }, el("h2", { text: "Declared vs observed" }),
      el("p", { class: "ar-note", text: "What Step 0 recorded as processing history, next to what the values show. A comparison only: nothing is overridden." }),
      rows.length ? el("div", { class: "ar-table-wrap" }, el("table", {}, el("thead", {}, el("tr", {}, ["dataset", "unit", "item", "declared", "observed", "status"].map((h) => el("th", { text: h })))),
        el("tbody", {}, rows.map((r) => el("tr", {}, el("td", { text: r.dataset }), el("td", { text: r.assay }), el("td", { text: r.item.replace(/_/g, " ") }),
          el("td", { text: r.declared || "not answered" }), el("td", { text: obs(r) }), el("td", {}, el("span", { class: `ar-pill ${r.status}`, text: r.status.replace(/_/g, " ") })))))))
        : el("p", { class: "ar-empty", text: "A4 was not run: no comparison." }));
  }
  function render(root, b, opts = {}) {
    root.classList.add("ar");
    const byUnit = {};
    for (const f of b.findings) {
      const key = f.dataset === "session" ? "session" : `${f.dataset}/${f.assay}`;
      (byUnit[key] = byUnit[key] || []).push(f);
    }
    const keys = Object.keys(byUnit).sort((a, b2) => (a === "session") - (b2 === "session") || a.localeCompare(b2));
    const label = (k) => {
      if (k === "session") return "Session (all datasets)";
      const f = byUnit[k][0], d = b.datasets.find((x) => x.dataset_id === f.dataset);
      return `${f.dataset} · ${d ? d.name : ""} · ${f.assay_label || f.assay}`;
    };
    const fileOf = new Map(b.findings.map((f, i) => [f, (b.manifest.findings[i] || {}).file]));
    const keyOf = (f) => (f.dataset === "session" ? "session" : `${f.dataset}/${f.assay}`);
    let cur = keys[0];
    const tabs = el("nav", { class: "ar-tabs", role: "tablist" });
    const body = el("div", { class: "ar-cards" });
    const show = (k) => {
      cur = k;
      tabs.replaceChildren(...keys.map((x) => el("button", { class: `ar-tab${x === cur ? " sel" : ""}`, type: "button", role: "tab", "aria-selected": String(x === cur), text: label(x), onclick: () => show(x) })));
      const fs = byUnit[k].slice().sort((a, c) => ORDER.indexOf(a.audit_id) - ORDER.indexOf(c.audit_id));
      body.replaceChildren(...fs.map((f) => card(f, fileOf.get(f))));
    };
    root.replaceChildren(opts.top || "", header(b), dvoPanel(b), tabs, body);
    if (keys.length) show(cur);
    else body.append(el("p", { class: "ar-empty", text: "No findings." }));
    const byFile = (file) => b.findings.find((f) => fileOf.get(f) === file);
    const focus = (file) => {
      const f = byFile(file);
      if (!f) return null;
      if (keyOf(f) !== cur) show(keyOf(f));
      const n = document.getElementById(`card-${f.audit_id}-${f.dataset}-${f.assay}`);
      if (n) n.scrollIntoView({ behavior: "smooth", block: "start" });
      return n;
    };
    return {
      showTab: show, focus,
      setView(v) { focus(v.finding); const fn = VIEWS[v.finding]; return fn ? fn(v) : null; },
    };
  }
  function diffPanel(cmp) {
    const f = cmp.findings || [];
    return el("div", { class: "ar ar-diff" }, el("b", { text: `Compared with ${cmp.a}` }),
      el("div", { text: Object.keys(cmp.params_changed).length ? `Parameters: ${Object.entries(cmp.params_changed).map(([k, v]) => `${k} ${fmt(v.a)} → ${fmt(v.b)}`).join(", ")}` : "Same parameters." }),
      el("div", { text: cmp.overrides_changed ? "Overrides changed." : "Same overrides." }),
      el("div", { text: cmp.findings_identical ? "Findings identical." : `${f.length} finding(s) differ.` }),
      f.length ? el("ul", {}, f.slice(0, 30).map((x) => el("li", { text: `${x.finding}: ${x.only_in ? `only in ${x.only_in}` : [x.status ? `status ${x.status.join(" → ")}` : null, x.indicators_added && x.indicators_added.length ? `+ ${x.indicators_added.join(", ")}` : null, x.indicators_removed && x.indicators_removed.length ? `− ${x.indicators_removed.join(", ")}` : null, x.n_measures_changed ? `${x.n_measures_changed} measure(s) changed` : "method parameters only"].filter(Boolean).join("; ")}` }))) : null);
  }
  window.PRISM_REPORT = { render, diffPanel, fmt };
})();
