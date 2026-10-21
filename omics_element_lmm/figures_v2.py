"""Figures for the full analysis.  python figures_v2.py --data-dir <dir> --out-dir results_v2"""
import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import full_analysis as fa

warnings.filterwarnings("ignore")
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
COL = {"Naive": "#8a8a85", "Pre-ART": "#eb6834", "Early-ART": "#9cc3ef", "Mid-ART": "#2a78d6", "Late-ART": "#0b3f7e"}
MK = {"Naive": "s", "Pre-ART": "o", "Early-ART": "^", "Mid-ART": "D", "Late-ART": "v"}
GROUPS = list(COL)
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})


def legend_stage(fig, y=0.0):
    hs = [plt.Line2D([], [], marker=MK[g], ls="", color=COL[g], label=g, markersize=7, mec=SURF) for g in GROUPS]
    fig.legend(handles=hs, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, y))


def fig_element_traj(out, xl, meta):
    x, tr, wk = fa.element_trajectories(xl, meta)
    els = fa.MAIN + ["Ni"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    rng = np.random.default_rng(0)
    for ax, el in zip(axes.ravel(), els):
        for an, sub in x[~x.naive].groupby("animal"):
            sub = sub.assign(gi=[GROUPS.index(g) for g in sub.group]).sort_values("week")
            ax.plot(sub.gi, sub[el], color="#bdbcb6", lw=.8, zorder=1)
        for g in GROUPS:
            s = x[x.group == g]; ax.scatter(GROUPS.index(g) + rng.uniform(-.08, .08, len(s)), s[el], c=COL[g], marker=MK[g], s=36, ec=SURF, lw=.8, zorder=3)
            ax.hlines(s[el].mean(), GROUPS.index(g) - .22, GROUPS.index(g) + .22, color=INK, lw=1.6, zorder=4)
        t = tr[tr.element == el].set_index("contrast")
        txt = "\n".join(f"{c.split(' vs ')[0].replace('-ART','')} vs Naive q={t.loc[c,'q']:.3f}" for c in ["Pre-ART vs Naive", "Early-ART vs Naive", "Mid-ART vs Naive", "Late-ART vs Naive"] if t.loc[c, "q"] < .1)
        ax.set_title(el + ("  (negative control)" if el == "Ni" else "")); ax.set_xticks(range(5), ["Naive", "Pre", "Early", "Mid", "Late"]); ax.set_ylabel("concentration (as supplied)")
        if txt: ax.text(.02, .97, txt, transform=ax.transAxes, va="top", fontsize=7, color=MUTED)
    fig.suptitle("Plasma elements by ART stage - all 37 samples (30 infected, 7 naive); grey lines = same animal; bar = group mean", y=.995)
    legend_stage(fig, 0.005); fig.tight_layout(rect=(0, .03, 1, .98)); fig.savefig(out / "fig1_element_trajectories.png", dpi=150); plt.close(fig)


def fig_ni_control(out, xl, summ):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8))
    for nv, nm in [(True, "Naive"), (False, "Pre-ART")]:
        s = xl[xl.naive == nv]
        a.scatter(s.Fe, s.Ni, c=COL["Naive"] if nv else COL["Late-ART"], marker="s" if nv else "o", s=40, ec=SURF, label="Naive" if nv else "Infected")
    r = stats.linregress(xl.Fe, xl.Ni); xs = np.linspace(xl.Fe.min(), xl.Fe.max(), 10)
    a.plot(xs, r.intercept + r.slope * xs, color=INK, lw=1)
    a.set_xlabel("Fe"); a.set_ylabel("Ni"); a.set_title(f"Ni rises with Fe (r = {r.rvalue:.2f}, all 37 samples)\nNi/Fe = {np.mean(xl.Ni/xl.Fe):.4f} on average")
    a.legend(frameon=False)
    pal = {"Ni": "#eb6834", "Ni_resFe": "#1baf7a"}
    from lmm_engine import np as _np
    for el in fa.ELS:
        r_ = pd.read_csv(out / "per_element" / f"proteomics_{el}.csv"); p = np.sort(r_["M1_stage_p_E"].dropna().values); n = len(p)
        e = (np.arange(1, n + 1) - .5) / n; lam = summ.set_index("element").loc[el, "lambda_M1"]
        c = pal.get(el, "#c5c4be"); lw = 2 if el in pal else 1
        b.plot(-np.log10(e), -np.log10(p), color=c, lw=lw, zorder=3 if el in pal else 1)
        if el in pal or el in ("Cu", "Mg", "P"):
            b.text(-np.log10(e[0]) * .98, -np.log10(p[0]), f"{el} λ={lam:.2f}", fontsize=8, color=INK, ha="right", va="center")
    m = 4; b.plot([0, m], [0, m], color=INK, lw=.8, ls="--"); b.set_xlim(0, 4.3); b.set_ylim(0, 5)
    b.set_xlabel("expected -log10 p"); b.set_ylabel("observed -log10 p")
    b.set_title("QQ of protein ~ element p-values (M1), 11,011 proteins\nNi / Fe-corrected Ni = negative controls; grey = other elements")
    fig.tight_layout(); fig.savefig(out / "fig2_Ni_negative_control.png", dpi=150); plt.close(fig)


def fig_volcano(out):
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    for ax, el in zip(axes.ravel(), fa.MAIN):
        r = pd.read_csv(out / "per_element" / f"proteomics_{el}.csv"); y = -np.log10(r.M1_stage_p_E); sug = r.M1_stage_p_E < .001
        ax.scatter(r.M1_stage_beta_E[~sug], y[~sug], s=5, c="#c5c4be"); ax.scatter(r.M1_stage_beta_E[sug], y[sug], s=14, c="#eb6834", ec=SURF, lw=.4)
        for i in r.index[:4]: ax.annotate(str(r.label[i])[:12], (r.M1_stage_beta_E[i], y[i]), fontsize=7, xytext=(3, 2), textcoords="offset points")
        ax.set_title(f"{el}   min FDR q = {r.M1_stage_q_E.min():.2f}"); ax.set_xlabel("beta per 1 SD element (log2 units)"); ax.set_ylabel("-log10 p")
    fig.suptitle("Protein ~ element + stage (+ animal), all samples. Orange = p < 0.001 (nominal; none reach FDR < 0.05)", y=.995)
    fig.tight_layout(); fig.savefig(out / "fig3_volcano_by_element.png", dpi=140); plt.close(fig)


def fig_top_proteins(out, d, Y):
    S = ["Pre-ART", "Early-ART", "Mid-ART", "Late-ART"]
    for el in fa.MAIN:
        r = pd.read_csv(out / "per_element" / f"proteomics_{el}.csv").head(6)
        fig, axes = plt.subplots(2, 6, figsize=(20, 6.6))
        for k, (_, row) in enumerate(r.iterrows()):
            a = axes[0, k]; y = Y[row.feature].values
            for an in d.nhp_id.unique():
                m = (d.nhp_id == an).values; o = np.argsort(d.week.values[m]); a.plot(d[el].values[m][o], y[m][o], color="#bdbcb6", lw=.7, zorder=1)
            for sg in S:
                m = (d.ART_stage == sg).values; a.scatter(d[el].values[m], y[m], c=COL[sg], marker=MK[sg], s=30, ec=SURF, lw=.6, zorder=3)
            a.set_title(f"{row.label}\nβ={row.M1_stage_beta_E:+.2f}  p={row.M1_stage_p_E:.1e}", fontsize=9); a.set_xlabel(el); a.set_ylabel("log2 RFU")
            b = axes[1, k]
            sl = [row[f"M6_slope_beta_{s}"] for s in ["Pre", "Early", "Mid", "Late"]]; se = [row[f"M6_slope_se_{s}"] for s in ["Pre", "Early", "Mid", "Late"]]
            for j, sg in enumerate(S):
                b.errorbar(j, sl[j], yerr=1.96 * se[j], color=COL[sg], marker=MK[sg], ms=6, capsize=3, lw=1.3)
            b.axhline(0, color=INK, lw=.7); b.set_xticks(range(4), ["Pre", "Early", "Mid", "Late"]); b.set_ylabel(f"slope vs {el} (±1.96 SE)")
            b.set_title(f"slope heterogeneity p={row.M6_slope_p_heterogeneity:.2f}", fontsize=8)
        fig.suptitle(f"{el}: six strongest protein associations (top: raw data, grey lines = same animal; bottom: association by ART stage)", y=.995)
        legend_stage(fig, 0.0) if False else None
        fig.tight_layout(); fig.savefig(out / f"fig4_top_proteins_{el}.png", dpi=110); plt.close(fig)


def fig_traj_heatmap(out):
    t = pd.read_csv(out / "protein_trajectory_stage_week.csv").head(60)
    M = t[[f"mean_log2_{s}" for s in ["Pre-ART", "Early-ART", "Mid-ART", "Late-ART"]]].values; Z = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
    from scipy.cluster.hierarchy import linkage, leaves_list
    o = leaves_list(linkage(Z, "average")); fig, ax = plt.subplots(figsize=(6, 13))
    im = ax.imshow(Z[o], cmap="RdBu_r", vmin=-1.6, vmax=1.6, aspect="auto"); ax.set_xticks(range(4), ["Pre", "Early", "Mid", "Late"]); ax.grid(False)
    ax.set_yticks(range(len(o)), t.label.values[o], fontsize=7); ax.set_title("Top 60 proteins by ART-stage effect\n(row-standardised stage means; 48 have FDR<0.05)")
    fig.colorbar(im, ax=ax, shrink=.4, label="z of stage mean"); fig.tight_layout(); fig.savefig(out / "fig5_protein_trajectories.png", dpi=140); plt.close(fig)


def fig_modules(out):
    m = pd.read_csv(out / "tri" / "modules_vs_elements.csv"); info = pd.read_csv(out / "tri" / "modules_vs_stage_week.csv")
    B = m.pivot(index="module", columns="element", values="beta_E_stage_model"); P = m.pivot(index="module", columns="element", values="p_E_stage_model")
    B = B[fa.MAIN + ["Ni"]]; P = P[fa.MAIN + ["Ni"]]
    fig, ax = plt.subplots(figsize=(9, 6.5)); im = ax.imshow(B.values, cmap="RdBu_r", vmin=-.6, vmax=.6, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(B.shape[1]), B.columns); ylab = [f"{mm} (n={int(info.set_index('module').loc[mm,'n_proteins'])})" for mm in B.index]; ax.set_yticks(range(B.shape[0]), ylab)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            if P.values[i, j] < .05: ax.text(j, i, "*" if P.values[i, j] >= .01 else "**", ha="center", va="center", fontsize=11)
    ax.set_title("Protein modules vs elements (module eigengene per 1 SD element, adjusted for stage)\n* p<0.05  ** p<0.01 (nominal; no result passes FDR across the module x element tests)")
    fig.colorbar(im, ax=ax, label="beta (SD of eigengene)"); fig.tight_layout(); fig.savefig(out / "fig6_modules_vs_elements.png", dpi=140); plt.close(fig)


def fig_pairs(out):
    f = out / "tri" / "pair_screen_summary.csv"
    if not f.exists(): return
    s = pd.read_csv(f); fig, ax = plt.subplots(figsize=(8, 4.2)); x = np.arange(len(s))
    ax.bar(x - .18, s.null_max_95, .36, color="#c5c4be", label="95th percentile of null (permuted element)")
    ax.bar(x + .18, s.obs_max_delta, .36, color="#2a78d6", label="observed best pair")
    ax.set_xticks(x, s.element); ax.set_ylabel("gain in R² from the 2nd protein"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Best protein pair per element vs what chance produces (all ~60M pairs screened)"); fig.tight_layout(); fig.savefig(out / "fig7_pair_screen.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default="."); ap.add_argument("--out-dir", default="results_v2"); a = ap.parse_args()
    out = Path(a.out_dir); d, Y, ann, xl, meta = fa.load(a.data_dir); summ = pd.read_csv(out / "summary_by_element.csv")
    fig_element_traj(out, xl, meta); fig_ni_control(out, xl, summ); fig_volcano(out); fig_top_proteins(out, d, Y); fig_traj_heatmap(out); fig_modules(out); fig_pairs(out)
