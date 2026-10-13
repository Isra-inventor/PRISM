"""
Omics (proteomics / metabolomics)  ~  element + ART stage   — loop over ALL elements.

Model per feature (protein or metabolite), for every element E in turn:

    log-abundance ~ E_std [+ VL_std] + ART_stage(ref = Pre-ART) + (1 | animal)

Usage
-----
    python run_all_elements.py --data-dir ../data --out-dir results            # proteomics
    python run_all_elements.py --omics metabolomics \
        --metabolite-matrix metabolomics_raw.tsv --metabolite-annotation "tables(Chemical Annotation).csv"
    python run_all_elements.py --elements Fe Zn --vl none                      # subset / sensitivity
"""
import argparse, re, warnings
from pathlib import Path

import numpy as np
import pandas as pd

from lmm_engine import fit_reml, term_tests, wald_f, bh

warnings.filterwarnings("ignore")

ELEMENTS = ["Na", "Mg", "P", "K", "Ca", "Fe", "Cu", "Zn"]
STAGES = ["Pre-ART", "Early-ART", "Mid-ART", "Late-ART"]


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_common(data_dir):
    """Elements + VL (Run2_common.csv) joined to sample metadata. One row per sample."""
    data_dir = Path(data_dir)
    el = pd.read_csv(data_dir / "Run2_common.csv", index_col="ID")
    meta = pd.read_csv(data_dir / "sample_metadata.tsv", sep="\t")
    meta = meta.set_index("nhp_week_index")[["nhp_id", "ART_stage"]]
    meta["ART_stage"] = meta["ART_stage"].str.strip()
    d = el.join(meta, how="left")
    if d["nhp_id"].isna().any():
        raise ValueError(f"samples without metadata: {d.index[d['nhp_id'].isna()].tolist()}")
    d["Week"] = [int(i.rsplit("_", 1)[1]) for i in d.index]
    d["ART_stage"] = pd.Categorical(d["ART_stage"], STAGES)
    return d


def load_proteomics(data_dir, samples):
    """SomaScan: rows = aptamers, columns = samples (+ annotation). log2(x+1)."""
    s = pd.read_csv(Path(data_dir) / "SomaExpr_common.csv", index_col=0)
    if "Type_meta" in s:                      # drop hybridisation controls / non-human spike-ins
        s = s[s["Type_meta"] == "Protein"]
    ann = s[[c for c in ["Target", "TargetFullName", "EntrezGeneSymbol", "UniProt"] if c in s]].copy()
    ann["label"] = ann.get("EntrezGeneSymbol", ann.get("Target", pd.Series(s.index, s.index))).fillna(
        pd.Series(s.index, s.index)).astype(str)
    Y = np.log2(s[samples] + 1).T            # samples x features
    return Y, ann


def load_metabolomics(matrix, annotation, samples):
    """Matrix rows = samples, cols = CHEM_ID, ALREADY log-transformed (as in the notebook)."""
    m = pd.read_csv(matrix, sep="\t", index_col=0)
    m.index = m.index.astype(str).str.strip()
    m.columns = m.columns.astype(str).str.strip()
    ann = pd.read_csv(annotation, dtype={"CHEM_ID": str}).drop_duplicates("CHEM_ID").set_index("CHEM_ID")
    ann["label"] = ann["PLOT_NAME"].fillna(ann["CHEMICAL_NAME"])
    keep = [s for s in samples if s in m.index]
    return m.loc[keep], ann.reindex(m.columns).assign(label=lambda a: a["label"].fillna(pd.Series(a.index, a.index)))


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
z = lambda v: (v - v.mean()) / v.std(ddof=1)


def vif_table(d, element, use_vl, use_stage=True):
    """Variance-inflation factors of the fixed-effect design (excluding intercept)."""
    X = pd.get_dummies(d["ART_stage"], drop_first=True).astype(float) if use_stage else pd.DataFrame(index=d.index)
    X.insert(0, element, z(d[element].values))
    if use_vl:
        X.insert(1, "VL", z(d["Log10VL"].values))
    out = {}
    for c in X:
        others = np.column_stack([np.ones(len(X)), X.drop(columns=c).values])
        beta, *_ = np.linalg.lstsq(others, X[c].values, rcond=None)
        r2 = 1 - ((X[c].values - others @ beta) ** 2).sum() / ((X[c].values - X[c].mean()) ** 2).sum()
        out[c] = 1 / (1 - r2)
    return pd.Series(out)


def run_element(Y, d, ann, element, use_vl=True, df_method="bw", use_stage=True):
    d = d.loc[Y.index]
    cols = {"const": np.ones(len(d)), element: z(d[element].values)}
    if use_vl:
        cols["VL"] = z(d["Log10VL"].values)
    stage_dm = pd.get_dummies(d["ART_stage"], drop_first=True).astype(float)
    present = [s for s in STAGES[1:] if s in stage_dm and stage_dm[s].sum() > 0] if use_stage else []
    for s in present:
        cols[s] = stage_dm[s].values
    names = list(cols)
    X = np.column_stack(list(cols.values()))
    fit = fit_reml(X, Y.values, d["nhp_id"].values)

    res = pd.DataFrame({"feature": Y.columns, "label": ann.loc[Y.columns, "label"].values})
    for extra in ["TargetFullName", "UniProt", "CHEMICAL_NAME", "SUPER_PATHWAY", "SUB_PATHWAY"]:
        if extra in ann:
            res[extra] = ann.loc[Y.columns, extra].values
    terms = [(element, element)] + ([("VL", "VL")] if use_vl else []) + [(s, f"{s.split('-')[0]}_vs_Pre") for s in present]
    for col, out in terms:
        b, se, t, df, p = term_tests(fit, X, names.index(col), df_method)
        res[f"beta_{out}"] = b; res[f"se_{out}"] = se; res[f"p_{out}"] = p
        res[f"q_{out}"] = bh(p)
    if present:
        Fv, pF = wald_f(fit, [names.index(s) for s in present], df_method)
        res["F_stage"] = Fv; res["p_stage"] = pF; res["q_stage"] = bh(pF)
    res["animal_var_ratio"] = fit["lam"]; res["animal_var_zero"] = fit["at_boundary"]
    res["df"] = df
    res.insert(0, "element", element)
    return res.rename(columns={f"beta_{element}": "beta_E", f"se_{element}": "se_E",
                               f"p_{element}": "p_E", f"q_{element}": "q_E"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--omics", choices=["proteomics", "metabolomics"], default="proteomics")
    ap.add_argument("--elements", nargs="*", default=ELEMENTS)
    ap.add_argument("--scenarios", nargs="*", default=["stage", "vl", "stage+vl"],
                    choices=["stage", "vl", "stage+vl"],
                    help="stage = element + ART stage | vl = element + log10 VL (no stage) | "
                         "stage+vl = element + VL + stage (original notebook model)")
    ap.add_argument("--df", choices=["bw", "resid", "conservative"], default="bw")
    ap.add_argument("--metabolite-matrix"); ap.add_argument("--metabolite-annotation")
    a = ap.parse_args()

    d = load_common(a.data_dir)
    if a.omics == "proteomics":
        Y, ann = load_proteomics(a.data_dir, list(d.index))
    else:
        Y, ann = load_metabolomics(a.metabolite_matrix, a.metabolite_annotation, list(d.index))
    d = d.loc[Y.index]
    print(f"{a.omics}: {Y.shape[1]} features, {len(d)} samples, {d.nhp_id.nunique()} animals")
    print(pd.crosstab(d.nhp_id, d.ART_stage).T.to_string())

    for mode in a.scenarios:
        print(f"\n=============== scenario: {mode} ===============")
        run_mode(a, d, Y, ann, mode)


def run_mode(a, d, Y, ann, mode):
    out = Path(a.out_dir) / a.omics / mode.replace("+", "_")
    out.mkdir(parents=True, exist_ok=True)
    allres, vifs = [], {}
    for el in a.elements:
        vifs[el] = vif_table(d, el, "vl" in mode, "stage" in mode)
        r = run_element(Y, d, ann, el, "vl" in mode, a.df, "stage" in mode)
        r.to_csv(out / f"lmm_{el}.csv", index=False)
        allres.append(r)
        n = {k: int((r[k] < .05).sum()) for k in [c for c in r if c.startswith("q_")]}
        print(f"{el:>3}: " + "  ".join(f"{k[2:]}={v}" for k, v in n.items()))
    big = pd.concat(allres)
    # FDR across ALL elements for the element term too (8 x features tests)
    big["q_E_global"] = bh(big["p_E"].values)
    big.to_csv(out / "lmm_all_elements.csv", index=False)
    pd.DataFrame(vifs).T.round(2).to_csv(out / "vif.csv")
    print("\nVIF (>5 = worrying):\n", pd.DataFrame(vifs).T.round(1).to_string())
    sig = lambda s: (s < .05).sum()
    spec = {"n_features": ("feature", "size"), "E_q05": ("q_E", sig), "E_global_q05": ("q_E_global", sig)}
    if "VL" in mode.upper():
        spec["VL_q05"] = ("q_VL", sig)
    if "stage" in mode:
        spec["stage_q05"] = ("q_stage", sig)
    spec["animal_var_zero"] = ("animal_var_zero", "mean")
    summ = big.groupby("element").agg(**spec).reindex(a.elements)
    summ.to_csv(out / "summary_by_element.csv")
    print("\n", summ.round(3).to_string())


if __name__ == "__main__":
    main()
