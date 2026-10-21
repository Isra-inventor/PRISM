"""
Full element x proteome analysis (all samples kept, no exclusions).

Per element E (Na Mg P K Ca Fe Cu Zn  +  Ni as negative-control) and per protein, random-intercept LMM (REML, t-tests):

  M1_stage      y ~ E + stage
  M2_vl         y ~ E + log10 VL                         (no stage)
  M3_week       y ~ E + log2(week+1)
  M4_wb         y ~ E_within + E_between + stage         (within-animal vs between-animal element effect)
  M5_E_x_week   y ~ E * log2(week+1)                     (does the association change along treatment time?)
  M8_rank       y ~ rank-normal(E) + stage  (reduces leverage of extreme samples; nothing is excluded)
  M7_hemo       y ~ E + stage + hemolysis score (red-cell protein proxy)
  M6_E_x_stage  y ~ stage + E:stage (4 stage-specific slopes; F-test of slope heterogeneity)
plus diagnostics: leave-one-animal-out stability of the M1 element term (diagnostic only; main results use all samples).

usage: python full_analysis.py --data-dir <dir with the 3 uploaded files> --out-dir results_v2
"""
import argparse, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from lmm_engine import fit_reml, term_tests, wald_f, bh, bw_df

warnings.filterwarnings("ignore")
MAIN = ["Na", "Mg", "P", "K", "Ca", "Fe", "Cu", "Zn"]
ELS = MAIN + ["Ni", "Ni_resFe"]
STAGES = ["Pre-ART", "Early-ART", "Mid-ART", "Late-ART"]
z = lambda v: (v - v.mean()) / v.std(ddof=1)
def fa_z_rank(v):
    r = stats.rankdata(v); return z(stats.norm.ppf((r - 0.5) / len(v)))


# t-test df, set by permutation calibration that keeps the animal clustering of the element
# (8 animals, 27 samples): terms that carry the element -> 8 ; between-animal element term -> 4 ;
# within-animal / time / stage contrasts -> n-G-(p-1) (=14)
DF_E, DF_BETWEEN = 8, 4


# ----------------------------------------------------------------------------- data
def load(data_dir):
    D = Path(data_dir)
    xl = pd.read_excel(D / "Cleaned_data.xlsx")
    xl = xl.loc[:, ~xl.columns.astype(str).str.startswith("Unnamed")]
    xl.columns = [str(c).strip() for c in xl.columns]

    def sid(n):
        m = re.match(r"^(\w+)\s+\S+\s+wk(\d+)\s*$", str(n).strip())
        return f"{m.group(1)}_{m.group(2)}" if m else None

    xl["ID"] = xl["Sample Name"].map(sid)
    xl["naive"] = xl["Sample Name"].str.contains("Naive", case=False)
    xl.loc[xl.naive, "ID"] = xl.loc[xl.naive, "Sample Name"].str.replace(r"\s*Naive NHP", "", regex=True).str.strip()
    xl["Ni_resFe"] = np.nan                      # filled below (Ni with the Fe-explained part removed, all 37 samples)
    b = np.polyfit(xl["Fe"], xl["Ni"], 1)
    xl["Ni_resFe"] = xl["Ni"] - np.polyval(b, xl["Fe"])

    meta = pd.read_csv(D / "sample_metadata.tsv", sep="\t").set_index("nhp_week_index")
    meta["viral_load"] = meta["viral_load"].fillna(50.0)        # blank = undetectable (<50 copies), as in Run2
    meta["Log10VL"] = np.log10(meta["viral_load"].clip(lower=50))

    s = pd.read_csv(D / "SomaExpr_common.csv", index_col=0)
    samples = [c for c in s.columns if re.match(r"^[A-Za-z0-9]+_\d+$", c)]
    if "Type_meta" in s:
        s = s[s["Type_meta"] == "Protein"]
    ann = s[[c for c in ["Target", "TargetFullName", "EntrezGeneSymbol", "UniProt"] if c in s]].copy()
    ann["label"] = ann["EntrezGeneSymbol"].fillna(pd.Series(s.index, s.index)).astype(str)
    Y = np.log2(s[samples] + 1).T

    inf = xl[~xl.naive].set_index("ID")
    d = inf.loc[samples, ELS].join(meta[["nhp_id", "ART_stage", "Log10VL", "study_group"]])
    d["week"] = [int(i.rsplit("_", 1)[1]) for i in d.index]
    d["tz"] = z(np.log2(d["week"] + 1.0))
    # hemolysis proxy: mean z-score of red-cell proteins on the SomaScan panel
    HEMO = ["HBB", "HBD", "CA1", "CAT", "BLVRB", "FECH", "ALAS2", "SOD1"]
    hidx = [ann.index[ann["label"] == g_][0] for g_ in HEMO if (ann["label"] == g_).any()]
    d["hemolysis_z"] = z(Y.loc[d.index, hidx].apply(z).mean(axis=1))
    ann["is_hemolysis_marker"] = ann.index.isin(hidx)
    d["VLz"] = z(d["Log10VL"])
    d["ART_stage"] = pd.Categorical(d["ART_stage"], STAGES)
    return d, Y.loc[d.index], ann, xl, meta


def stage_dummies(d, drop_first=True):
    return pd.get_dummies(d["ART_stage"], drop_first=drop_first).astype(float)[STAGES[1:] if drop_first else STAGES]


def lincomb(fit, c, df):
    """estimate / se / t / p of c'beta for every feature."""
    c = np.asarray(c, float)
    est = c @ fit["beta"]
    var = np.einsum("i,fij,j->f", c, fit["cov"], c)
    se = np.sqrt(var)
    t = est / se
    return est, se, 2 * stats.t.sf(np.abs(t), df)


# ----------------------------------------------------------------------------- models for one element
def run_element(d, Y, ann, el, loao=True):
    n = len(d); g = d["nhp_id"].values; Yv = Y.values
    E = z(d[el].values); sd_tot = d[el].std(ddof=1)
    S = stage_dummies(d).values
    one = np.ones(n); t = d["tz"].values; vl = d["VLz"].values
    out = pd.DataFrame({"element": el, "feature": Y.columns, "label": ann.loc[Y.columns, "label"].values})
    for c in ["TargetFullName", "UniProt"]:
        if c in ann: out[c] = ann.loc[Y.columns, c].values

    def add(prefix, X, names, terms, fit=None):
        fit = fit or fit_reml(X, Yv, g)
        for term in terms:
            col, lab = term[0], term[1]
            b, se, tt, df, p = term_tests(fit, X, names.index(col), term[2] if len(term) > 2 else "bw")
            out[f"{prefix}_beta_{lab}"] = b; out[f"{prefix}_se_{lab}"] = se
            out[f"{prefix}_p_{lab}"] = p; out[f"{prefix}_q_{lab}"] = bh(p)
        return fit

    sn = ["Early", "Mid", "Late"]
    # M1 stage
    X1 = np.column_stack([one, E, S]); n1 = ["c", "E"] + sn
    f1 = add("M1_stage", X1, n1, [("E", "E", DF_E)] + [(s, f"{s}_vs_Pre") for s in sn])
    Fv, pF = wald_f(f1, [2, 3, 4]); out["M1_stage_p_stageF"] = pF; out["M1_stage_q_stageF"] = bh(pF)
    out["animal_var_ratio"] = f1["lam"]; out["animal_var_zero"] = f1["at_boundary"]
    # M2 VL only
    add("M2_vl", np.column_stack([one, E, vl]), ["c", "E", "VL"], [("E", "E", DF_E), ("VL", "VL")])
    # M3 week only
    add("M3_week", np.column_stack([one, E, t]), ["c", "E", "week"], [("E", "E", DF_E), ("week", "week")])
    # M4 within / between
    Em = pd.Series(d[el].values).groupby(g).transform("mean").values
    Ew = (d[el].values - Em) / sd_tot; Eb = (Em - d[el].mean()) / sd_tot
    add("M4_wb", np.column_stack([one, Ew, Eb, S]), ["c", "Ew", "Eb"] + sn, [("Ew", "within"), ("Eb", "between", DF_BETWEEN)])
    # M5 E x week
    add("M5_Exweek", np.column_stack([one, E, t, E * t]), ["c", "E", "week", "Ex"], [("E", "E_atMeanWeek", DF_E), ("Ex", "E_x_week", DF_E)])
    # M6 stage-specific slopes
    X6 = np.column_stack([one, S, E, E[:, None] * S]); n6 = ["c"] + sn + ["E"] + [f"E_{s}" for s in sn]
    f6 = fit_reml(X6, Yv, g)
    df6 = DF_E
    for k, lab in enumerate(["Pre", "Early", "Mid", "Late"]):
        c = np.zeros(X6.shape[1]); c[4] = 1
        if k > 0: c[5 + k - 1] = 1
        est, se, p = lincomb(f6, c, df6)
        out[f"M6_slope_beta_{lab}"] = est; out[f"M6_slope_se_{lab}"] = se; out[f"M6_slope_p_{lab}"] = p
    Fh, ph = wald_f(f6, [5, 6, 7], DF_E); out["M6_slope_p_heterogeneity"] = ph; out["M6_slope_q_heterogeneity"] = bh(ph)
    # M7 hemolysis-adjusted
    out["is_hemolysis_marker"] = ann.loc[Y.columns, "is_hemolysis_marker"].values
    add("M7_hemo", np.column_stack([one, E, S, d["hemolysis_z"].values]), ["c", "E"] + sn + ["H"], [("E", "E", DF_E), ("H", "hemolysis")])
    # M8 rank-based element (normal scores): keeps every sample but removes the leverage of extreme concentrations
    Er = fa_z_rank(d[el].values)
    add("M8_rank", np.column_stack([one, Er, S]), ["c", "E"] + sn, [("E", "E", DF_E)])
    # diagnostic: leave-one-animal-out stability of M1 element term
    if loao:
        sign = np.sign(out["M1_stage_beta_E"].values); nsig = np.zeros(len(out)); worst = np.zeros(len(out))
        animals = np.unique(g); cnt = 0
        for a in animals:
            keep = g != a
            if keep.sum() < 8 or len(np.unique(g[keep])) < 3: continue
            Ek = z(d[el].values[keep]); Sk = S[keep]; Sk = Sk[:, Sk.sum(0) > 0]
            Xk = np.column_stack([one[keep], Ek, Sk])
            fk = fit_reml(Xk, Yv[keep], g[keep]); b, se, tt, df, p = term_tests(fk, Xk, 1, DF_E)
            nsig += ((p < .05) & (np.sign(b) == sign)); worst = np.maximum(worst, np.where(np.sign(b) == sign, p, 1.0)); cnt += 1
        out["LOAO_n_refits"] = cnt; out["LOAO_n_p05_same_sign"] = nsig; out["LOAO_worst_p"] = worst
    return out.sort_values("M1_stage_p_E").reset_index(drop=True)


def lambda_gc(p):
    p = np.asarray(p); p = p[np.isfinite(p) & (p > 0)]
    return float(np.median(stats.chi2.isf(p, 1)) / stats.chi2.ppf(0.5, 1))


# ----------------------------------------------------------------------------- element trajectories incl. naive
def element_trajectories(xl, meta, d_all=None):
    """log2 element ~ group (Naive / Pre / Early / Mid / Late) + (1|animal); ALL 37 samples."""
    x = xl.copy()
    inf = ~x.naive
    ids = x["ID"]
    x["week"] = [int(i.rsplit("_", 1)[1]) if inf_ else np.nan for i, inf_ in zip(ids, inf)]
    x["animal"] = [i.rsplit("_", 1)[0] if inf_ else i for i, inf_ in zip(ids, inf)]
    x["group"] = ["Naive" if nv else meta.loc[i, "ART_stage"] for i, nv in zip(ids, x.naive)]
    x["tz"] = z(np.log2(x["week"] + 1)) if False else np.nan
    G = ["Naive", "Pre-ART", "Early-ART", "Mid-ART", "Late-ART"]
    dm = pd.get_dummies(pd.Categorical(x["group"], G), drop_first=True).astype(float).values
    Yel = np.log2(x[MAIN + ["Ni"]].values.astype(float))
    X = np.column_stack([np.ones(len(x)), dm])
    f = fit_reml(X, Yel, x["animal"].values); df = bw_df(f["n"], f["G"], f["p"])
    rows = []
    for j, el in enumerate(MAIN + ["Ni"]):
        for k, gname in enumerate(G[1:], start=1):
            b = f["beta"][k, j]; se = f["se"][k, j]; p = 2 * stats.t.sf(abs(b / se), df)
            rows.append(dict(element=el, contrast=f"{gname} vs Naive", log2FC=b, se=se, p=p))
        for a_i, a in enumerate(G[1:], start=1):
            for b_i, b_ in enumerate(G[1:], start=1):
                if b_i > a_i and (a, b_) in [("Pre-ART", "Late-ART"), ("Pre-ART", "Early-ART"), ("Early-ART", "Late-ART")]:
                    c = np.zeros(X.shape[1]); c[b_i] = 1; c[a_i] = -1
                    est = c @ f["beta"][:, j]; se = np.sqrt(c @ f["cov"][j] @ c)
                    rows.append(dict(element=el, contrast=f"{b_} vs {a}", log2FC=est, se=se, p=2 * stats.t.sf(abs(est / se), df)))
    tr = pd.DataFrame(rows); tr["q"] = bh(tr["p"].values)
    # continuous week (infected only, n=30)
    xi = x[inf].copy(); xi["t"] = z(np.log2(xi["week"] + 1))
    Xi = np.column_stack([np.ones(len(xi)), xi["t"].values]); fi = fit_reml(Xi, np.log2(xi[MAIN + ["Ni"]].values.astype(float)), xi["animal"].values)
    dfi = bw_df(fi["n"], fi["G"], fi["p"])
    wk = pd.DataFrame({"element": MAIN + ["Ni"], "log2FC_per_SD_log2week": fi["beta"][1], "se": fi["se"][1],
                       "p": 2 * stats.t.sf(np.abs(fi["beta"][1] / fi["se"][1]), dfi)}); wk["q"] = bh(wk["p"].values)
    return x, tr, wk


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default="."); ap.add_argument("--out-dir", default="results_v2")
    ap.add_argument("--no-loao", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args(); out = Path(a.out_dir); (out / "per_element").mkdir(parents=True, exist_ok=True)
    d, Y, ann, xl, meta = load(a.data_dir)
    print(f"proteomics samples: {len(d)} (all kept) | animals: {d.nhp_id.nunique()} | proteins: {Y.shape[1]} | element table rows: {len(xl)} (naive: {int(xl.naive.sum())})")
    d.to_csv(out / "sample_table_used.csv")
    summ = []
    for el in ELS:
        r = run_element(d, Y, ann, el, loao=not a.no_loao)
        r.to_csv(out / "per_element" / f"proteomics_{el}.csv", index=False, float_format="%.6g")
        row = {"element": el, "lambda_M1": lambda_gc(r["M1_stage_p_E"]), "lambda_M2": lambda_gc(r["M2_vl_p_E"]), "lambda_M3": lambda_gc(r["M3_week_p_E"])}
        for k in ["M1_stage_q_E", "M2_vl_q_E", "M3_week_q_E", "M4_wb_q_within", "M4_wb_q_between", "M5_Exweek_q_E_x_week", "M6_slope_q_heterogeneity", "M7_hemo_q_E", "M8_rank_q_E"]:
            row[k.replace("_q", "_n_q05")] = int((r[k] < .05).sum())
        row["M1_stage_p<0.001"] = int((r["M1_stage_p_E"] < .001).sum())
        summ.append(row); print(el, {k: v for k, v in row.items() if k != "element"})
    pd.DataFrame(summ).round(3).to_csv(out / "summary_by_element.csv", index=False)
    # element-free protein trajectory
    S = stage_dummies(d).values; one = np.ones(len(d)); g = d.nhp_id.values
    f = fit_reml(np.column_stack([one, S]), Y.values, g); Fv, pF = wald_f(f, [1, 2, 3])
    f2 = fit_reml(np.column_stack([one, d.tz.values]), Y.values, g); b, se, tt, df, p2 = term_tests(f2, None, 1, "bw")
    tr = pd.DataFrame({"feature": Y.columns, "label": ann.loc[Y.columns, "label"].values})
    for s_ in STAGES: tr[f"mean_log2_{s_}"] = Y[d.ART_stage.values == s_].mean().values
    tr["p_stage_omnibus"] = pF; tr["q_stage_omnibus"] = bh(pF); tr["beta_per_SD_log2week"] = b; tr["p_week"] = p2; tr["q_week"] = bh(p2)
    tr.sort_values("p_stage_omnibus").to_csv(out / "protein_trajectory_stage_week.csv", index=False, float_format="%.6g")
    print("protein trajectory: stage q<.05:", int((tr.q_stage_omnibus < .05).sum()), " week q<.05:", int((tr.q_week < .05).sum()))
    # element trajectories (all 37)
    x, etr, ewk = element_trajectories(xl, meta)
    etr.to_csv(out / "element_trajectories_vs_naive.csv", index=False, float_format="%.5g"); ewk.to_csv(out / "element_week_trend.csv", index=False, float_format="%.5g")
    x.to_csv(out / "element_table_all37.csv", index=False)
    print(etr[etr.q < .1].round(4).to_string())


if __name__ == "__main__":
    main()
