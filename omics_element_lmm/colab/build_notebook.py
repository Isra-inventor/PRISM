import nbformat as nbf
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): C.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Elements × plasma proteome in ART-treated macaques — four complementary analyses
**What this notebook does (all 27 proteomics samples are used; nothing is excluded; Ni is ignored)**

| # | Analysis | Question |
|---|---|---|
| 1 | **Metal-binding protein sets** | Do proteins *known* to bind a metal (UniProt) or carry it (ceruloplasmin, transferrin…) move with that element? One set-level test instead of 11,000 single-protein tests. |
| 2 | **Element ratios & the whole ionome** | Do ratios (Cu/Zn, Fe/Cu, Na/K…) or the multi-element profile relate to the proteome better than single elements? Includes a multivariate PLS model tested on animals it has never seen. |
| 3 | **Recovery toward naive** | Using the 7 naive animals as the reference, how far is each infected sample's element profile from "normal", and does the proteome track that distance? |
| 4 | **Lead / lag** | Within an animal, does the element change *before* the protein, or after? |

**Statistical backbone (same for every variable).** Everything is a *within-animal* analysis: each animal gets its own baseline (so animal, cohort and – optionally – microarray-slide differences cannot create associations), a shared smooth time trend (quadratic in log2-weeks) is removed, and p-values come from **permuting the variable within each animal** (keeps the proteome's correlation structure, no distributional assumptions). Set-level tests use the same permutations, so protein–protein correlation is respected.

**Caveats you should keep in mind:** 8 animals / 27 samples (24 usable for within-animal tests: 3 animals have a single sample); slide is nearly nested in animal; naive animals are a different cohort. Treat positive results as leads to validate.

**How to run in Colab:** *Runtime → Run all*. When asked, upload `Cleaned_data.xlsx`, `SomaExpr_common.csv`, `sample_metadata.tsv`. Default `B_PERM=500` takes ≈10–15 min; raise to 2000 for final numbers.
""")

code('''
# ------------------------------------------------------------------ setup
import sys, os, re, io, warnings, time, shutil
import numpy as np, pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
IN_COLAB = "google.colab" in sys.modules

# ---- settings you may change
B_PERM   = 500        # permutations per variable (2000 for final results)
BY_SLIDE = False      # True = baseline per animal x microarray slide (batch-adjusted, fewer usable samples)
B_PLS    = 100        # permutations for the multivariate PLS test
B_LAG    = 2000       # permutations for the lead/lag test
OUT      = "results"
DATA_DIR = os.environ.get("DATA_DIR", "/content")
os.makedirs(OUT, exist_ok=True)

ELEMENTS = ["Na", "Mg", "P", "K", "Ca", "Fe", "Cu", "Zn"]          # Ni deliberately ignored
STAGES   = ["Pre-ART", "Early-ART", "Mid-ART", "Late-ART"]
NAVY, ORANGE, GREY, BLUE, LBLUE = "#0b3f7e", "#eb6834", "#8a8a85", "#2a78d6", "#9cc3ef"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1", "figure.dpi": 110, "font.size": 9})

def zs(v): v = np.asarray(v, float); return (v - v.mean()) / v.std(ddof=1)
def bh(p):
    p = np.asarray(p, float); out = np.full_like(p, np.nan); ok = ~np.isnan(p); pv = p[ok]; o = np.argsort(pv)
    r = pv[o] * len(pv) / (np.arange(len(pv)) + 1); r = np.minimum.accumulate(r[::-1])[::-1]; q = np.empty_like(pv); q[o] = np.minimum(r, 1); out[ok] = q; return out
''')

code('''
# ------------------------------------------------------------------ upload the three files (Colab) or read them from DATA_DIR
need = ["Cleaned_data.xlsx", "SomaExpr_common.csv", "sample_metadata.tsv"]
if IN_COLAB and not all(os.path.exists(f"{DATA_DIR}/{f}") for f in need):
    from google.colab import files
    print("Upload:", need); up = files.upload()
    for k in up: shutil.move(k, f"{DATA_DIR}/{k}") if os.path.dirname(os.path.abspath(k)) != DATA_DIR else None
print({f: os.path.exists(f"{DATA_DIR}/{f}") for f in need})
''')

md("## 0. Load and harmonise the data")
code('''
def load_data(D):
    xl = pd.read_excel(f"{D}/Cleaned_data.xlsx"); xl = xl.loc[:, ~xl.columns.astype(str).str.startswith("Unnamed")]; xl.columns = [str(c).strip() for c in xl.columns]
    xl["naive"] = xl["Sample Name"].str.contains("Naive", case=False)
    def sid(n):
        m = re.match(r"^(\\w+)\\s+\\S+\\s+wk(\\d+)\\s*$", str(n).strip()); return f"{m.group(1)}_{m.group(2)}" if m else None
    xl["ID"] = xl["Sample Name"].map(sid)
    meta = pd.read_csv(f"{D}/sample_metadata.tsv", sep="\\t").set_index("nhp_week_index")
    meta["viral_load"] = meta["viral_load"].fillna(50.0); meta["Log10VL"] = np.log10(meta["viral_load"].clip(lower=50))   # blank = undetectable (<50 copies)
    s = pd.read_csv(f"{D}/SomaExpr_common.csv", index_col=0)
    samples = [c for c in s.columns if re.match(r"^[A-Za-z0-9]+_\\d+$", c)]
    if "Type_meta" in s: s = s[s["Type_meta"] == "Protein"]                      # drop hybridisation controls / spike-ins
    ann = s[[c for c in ["Target", "TargetFullName", "EntrezGeneSymbol", "UniProt"] if c in s]].copy()
    ann["label"] = ann["EntrezGeneSymbol"].fillna(pd.Series(s.index, s.index)).astype(str)
    Y = np.log2(s[samples] + 1).T
    inf = xl[~xl.naive & xl.ID.notna()].set_index("ID")
    d = inf.loc[samples, ELEMENTS].join(meta[["nhp_id", "ART_stage", "Log10VL"]]); d["slide"] = meta.loc[d.index, "SlideId"].astype(str)
    d["week"] = [int(i.rsplit("_", 1)[1]) for i in d.index]; d["ART_stage"] = pd.Categorical(d["ART_stage"], STAGES)
    # all 37 element samples (30 infected + 7 naive) for the recovery analysis
    el_all = xl.copy(); el_all["animal"] = [i.rsplit("_", 1)[0] if isinstance(i, str) else n for i, n in zip(el_all.ID, el_all["Sample Name"])]
    el_all["week"] = [int(i.rsplit("_", 1)[1]) if isinstance(i, str) else np.nan for i in el_all.ID]
    el_all["group"] = ["Naive" if nv else meta.loc[i, "ART_stage"] for i, nv in zip(el_all.ID, el_all.naive)]
    return d, Y.loc[d.index], ann, el_all

d, Y, ann, el_all = load_data(DATA_DIR)
print(f"proteomics samples: {len(d)} | animals: {d.nhp_id.nunique()} | proteins: {Y.shape[1]} | element table: {len(el_all)} rows ({int(el_all.naive.sum())} naive)")
display(pd.crosstab(d.nhp_id, d.ART_stage))
print("slides per animal:", d.groupby("nhp_id").slide.nunique().to_dict(), "| 3 animals have a single sample (no within-animal information)")
''')

md("""
## 0b. The within-animal engine
Per variable `v` (an element, a ratio, a score…): remove each animal's mean from `v`, from every protein and from the time terms; regress each protein on `v` + time + time²; the coefficient on `v` is the **repeated-measures (within-animal) slope** (log2 protein per 1 SD of `v`). P-values: `v` is shuffled within animals `B_PERM` times and the whole scan repeated; the pooled permutation distribution of |t| gives p for every protein. The same shuffles give null distributions for protein **sets** (mean t = coordinated direction; mean t² = any association).
""")
code('''
class Within:
    def __init__(self, d, Y, by_slide=False, min_n=2):
        g = (d["nhp_id"] + "|" + d["slide"]) if by_slide else d["nhp_id"]
        cnt = g.value_counts(); keep = g.isin(cnt[cnt >= min_n].index).values
        self.g = g.values[keep]; self.groups = sorted(set(self.g)); self.N = int(keep.sum()); self.idx = np.where(keep)[0]
        self.members = {k: np.where(self.g == k)[0] for k in self.groups}
        self.d = d.iloc[self.idx]; self.cols = Y.columns; self.animal = self.d["nhp_id"].values
        self.Yw = self.demean(Y.values[keep]); t = np.log2(self.d["week"].values + 1.0); t = (t - t.mean()) / t.std(ddof=1)
        self.T = np.column_stack([self.demean(t[:, None]), self.demean((t ** 2)[:, None])])
    def demean(self, M):
        M = np.asarray(M, float).copy()
        for r in self.members.values(): M[r] -= M[r].mean(0)
        return M
    def fit(self, v):
        ew = self.demean(((v - v.mean()) / v.std(ddof=1))[:, None]); X = np.column_stack([ew, self.T]); G = np.linalg.inv(X.T @ X); XtY = X.T @ self.Yw
        b = G @ XtY; rss = (self.Yw ** 2).sum(0) - (b * XtY).sum(0); dr = self.N - len(self.groups) - X.shape[1]
        return b[0], b[0] / np.sqrt(rss / dr * G[0, 0]), dr
    def shuffle(self, v, rng):
        o = v.copy()
        for r in self.members.values(): o[r] = rng.permutation(v[r])
        return o
    def scan(self, v, B, set_idx=None, seed=0):
        v = np.asarray(v, float); rng = np.random.default_rng(seed); b0, t0, dr = self.fit(v); F = len(t0)
        if set_idx:
            Wm = np.zeros((len(set_idx), F)); names = list(set_idx)
            for i, k in enumerate(names): Wm[i, set_idx[k]] = 1.0 / len(set_idx[k])
        nt = np.empty((B, F), np.float32); m1, m2 = [], []
        for k in range(B):
            _, t, _ = self.fit(self.shuffle(v, rng)); nt[k] = np.abs(t)
            if set_idx: m1.append(Wm @ t); m2.append(Wm @ (t ** 2))
        ns = np.sort(nt.ravel()); p = (1 + len(ns) - np.searchsorted(ns, np.abs(t0), side="left")) / (1 + len(ns))
        res = pd.DataFrame({"feature": self.cols, "label": ann.loc[self.cols, "label"].values, "slope": b0, "t": t0, "p": p}); res["q"] = bh(p)
        sets = None
        if set_idx:
            m1, m2 = np.array(m1), np.array(m2); o1, o2 = Wm @ t0, Wm @ (t0 ** 2)
            sets = pd.DataFrame({"set": names, "n_proteins": [len(set_idx[k]) for k in names], "mean_t": o1, "p_direction": (1 + (np.abs(m1) >= np.abs(o1)).sum(0)) / (1 + B),
                                 "mean_t2": o2, "p_any_association": (1 + (m2 >= o2).sum(0)) / (1 + B), "frac_positive": [(t0[set_idx[k]] > 0).mean() for k in names]})
        return res.sort_values("p").reset_index(drop=True), sets

W = Within(d, Y, by_slide=BY_SLIDE)
print(f"within-animal engine: {len(W.groups)} baseline groups, N={W.N} samples usable | by_slide={BY_SLIDE}")
CACHE = {}
''')

md("""
# 1. Metal-binding protein sets
**Idea.** A blind scan of 11,011 proteins pays a heavy multiple-testing price. Proteins that *bind or carry* a metal are the ones with a prior reason to track it, so we test them as **sets**.
Sets come from **UniProt keywords** (Copper KW-0186, Iron KW-0408, Zinc KW-0862, Calcium KW-0106, Magnesium KW-0460, Potassium KW-0630, Sodium KW-0915, any metal KW-0479) fetched live, plus **curated** carrier lists and three control sets: red-cell and platelet proteins (cell-content / pre-analytical effects) and interferon-stimulated genes (a known ART response).
If UniProt cannot be reached, the notebook falls back to the curated lists only and says so.
""")
code('''
import requests
METAL_KW = {"Copper": "KW-0186", "Iron": "KW-0408", "Zinc": "KW-0862", "Calcium": "KW-0106", "Magnesium": "KW-0460", "Potassium": "KW-0630", "Sodium": "KW-0915", "Any metal-binding": "KW-0479"}
def uniprot_ids(kw):
    url = f"https://rest.uniprot.org/uniprotkb/stream?query=(keyword:{kw})+AND+(organism_id:9606)+AND+(reviewed:true)&fields=accession&format=list"
    r = requests.get(url, timeout=90); r.raise_for_status(); return set(r.text.split())
CURATED = {
 "Curated: Cu carriers/enzymes": ["CP", "SOD1", "SOD3", "ATOX1", "CCS", "LOXL2", "AOC3", "APP", "ALB"],
 "Curated: Fe carriers/storage": ["TF", "TFRC", "FTL", "FTH1", "HP", "HPX", "LTF", "HAMP", "LCN2", "HMOX1", "BLVRB", "TFR2"],
 "Curated: Zn proteins": ["ALB", "MT1A", "MT2A", "SOD1", "CA1", "CA2", "CA4", "MMP1", "MMP2", "MMP3", "MMP7", "MMP9", "MMP10", "MMP12", "MMP13", "ADAM10", "ADAM17", "S100A8", "S100A9", "S100A12"],
 "Curated: Ca-binding": ["ALB", "CALM1", "CALR", "S100A8", "S100A9", "S100B", "S100A12", "GSN", "F2", "F7", "F9", "F10", "PROC", "PROS1", "CALCA", "PTH", "ANXA1", "ANXA2", "ANXA5", "ANXA11"],
 "Curated: Na/K-ATPase": ["ATP1A1", "ATP1B1", "ATP1B2", "ATP1B3", "ATP2B1", "ATP2B4", "FXYD2"],
 "Control: red-cell proteins": ["HBB", "HBD", "HBA1", "CA1", "CAT", "BLVRB", "FECH", "ALAS2", "SOD1", "G6PD", "PRDX2", "SLC4A1"],
 "Control: platelet proteins": ["PF4", "PPBP", "GP1BA", "GP1BB", "GP9", "ITGA2B", "ITGB3", "SELP", "THBS1", "TUBB1", "CD9", "CLEC1B", "GP6", "FLNA", "PDGFA", "PDGFB", "VCL", "TPM4", "MYL9", "MYL12A"],
 "Control: interferon-stimulated (ART response)": ["ISG15", "MX1", "IFIT1", "IFIT3", "OAS1", "IFI6", "IRF7", "STAT1", "CXCL10", "IFI27", "RSAD2"],
}
up_col = ann["UniProt"].fillna("").astype(str).str.split(r"[|,; ]+")
SETS, SOURCE = {}, {}
for name, kw in METAL_KW.items():
    try:
        ids = uniprot_ids(kw); mask = up_col.map(lambda L: any(x in ids for x in L)).values
        if mask.sum() >= 5: SETS[f"UniProt: {name}"] = np.where(mask)[0]; SOURCE[f"UniProt: {name}"] = "UniProt live"
    except Exception as e:
        print(f"UniProt not reachable for {name} ({type(e).__name__}) -> curated sets only for this metal"); break
lab = ann["label"].values
for name, genes in CURATED.items():
    idx = np.where(np.isin(lab, genes))[0]
    if len(idx) >= 3: SETS[name] = idx; SOURCE[name] = "curated"
# map SOMA feature index -> index in the engine's column order (identical order)
assert list(W.cols) == list(Y.columns)
print(pd.Series({k: len(v) for k, v in SETS.items()}, name="proteins in set (on this panel)").to_string())
HYP = {"Cu": ["UniProt: Copper", "Curated: Cu carriers/enzymes"], "Fe": ["UniProt: Iron", "Curated: Fe carriers/storage"], "Zn": ["UniProt: Zinc", "Curated: Zn proteins"],
       "Ca": ["UniProt: Calcium", "Curated: Ca-binding"], "Mg": ["UniProt: Magnesium", "Control: red-cell proteins", "Control: platelet proteins"], "P": ["Control: red-cell proteins", "Control: platelet proteins"],
       "Na": ["UniProt: Sodium", "Curated: Na/K-ATPase"], "K": ["UniProt: Potassium", "Curated: Na/K-ATPase"]}
''')
code('''
def get_scan(name, v):
    if name not in CACHE:
        t0 = time.time(); CACHE[name] = W.scan(np.asarray(v, float)[W.idx] if len(v) == len(d) else np.asarray(v, float), B_PERM, SETS, seed=sum(map(ord, name)))
        res, st = CACHE[name]; print(f"  scanned {name:16s} {time.time()-t0:5.0f}s | p<0.001: {int((res.p<.001).sum()):4d} (chance {len(res)*.001:.0f}) | q<0.10: {int((res.q<.10).sum())}")
        res.to_csv(f"{OUT}/scan_{name.replace('/','_')}.csv", index=False); st.to_csv(f"{OUT}/sets_{name.replace('/','_')}.csv", index=False)
    return CACHE[name]
print("Scanning the 8 elements (within-animal, permutation-calibrated)...")
for el in ELEMENTS: get_scan(el, d[el].values)
''')
code('''
# ---- set-level results: all sets x all elements, hypothesis-driven pairs flagged
rows = []
for el in ELEMENTS:
    st = CACHE[el][1].copy(); st.insert(0, "element", el); st["hypothesis_pair"] = st.set.isin(HYP.get(el, [])); rows.append(st)
S = pd.concat(rows, ignore_index=True)
S["q_direction_all"] = bh(S.p_direction.values); S["q_any_all"] = bh(S.p_any_association.values)
H = S[S.hypothesis_pair].copy(); H["q_direction_hyp"] = bh(H.p_direction.values); H["q_any_hyp"] = bh(H.p_any_association.values)
S.to_csv(f"{OUT}/1_metal_sets_all.csv", index=False); H.to_csv(f"{OUT}/1_metal_sets_hypothesis_pairs.csv", index=False)
print("HYPOTHESIS-DRIVEN pairs (element x its own metal/cell-content sets), BH within these", len(H), "tests:")
display(H.sort_values("p_any_association")[["element", "set", "n_proteins", "mean_t", "frac_positive", "p_direction", "q_direction_hyp", "p_any_association", "q_any_hyp"]].round(4).head(15))
print("\\nStrongest set-level signals over ALL sets x elements (exploratory):")
display(S.sort_values("p_any_association").head(12)[["element", "set", "n_proteins", "mean_t", "frac_positive", "p_direction", "p_any_association", "q_any_all"]].round(4))
# heat-map of coordinated direction (mean t)
M = S.pivot(index="set", columns="element", values="mean_t")[ELEMENTS]; P = S.pivot(index="set", columns="element", values="p_direction")[ELEMENTS]
fig, ax = plt.subplots(figsize=(8.5, 0.38 * len(M) + 1.5)); im = ax.imshow(M.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto"); ax.grid(False)
ax.set_xticks(range(len(ELEMENTS)), ELEMENTS); ax.set_yticks(range(len(M)), M.index, fontsize=8)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if P.values[i, j] < .05: ax.text(j, i, "*" if P.values[i, j] >= .01 else "**", ha="center", va="center", fontsize=11)
ax.set_title("Protein sets vs elements: mean within-animal t (red = set rises with element)\\n* p<0.05  ** p<0.01 (permutation, not multiplicity-corrected)", fontsize=9); fig.colorbar(im, ax=ax, shrink=.6, label="mean t"); fig.tight_layout(); plt.show()
''')
code('''
# ---- the carrier proteins themselves (single proteins, with their own within-animal slope per element)
carriers = ["CP", "TF", "TFRC", "FTL", "FTH1", "HP", "HPX", "LTF", "HAMP", "ALB", "SOD1", "SOD3", "MT2A", "LCN2", "GSN", "S100A8", "S100A9"]
rows = []
for el in ELEMENTS:
    r = CACHE[el][0]; x = r[r.label.isin(carriers)].copy(); x.insert(0, "element", el); rows.append(x)
car = pd.concat(rows); car["q_within_element_scan"] = car.q
car.to_csv(f"{OUT}/1_carrier_proteins.csv", index=False)
print("Carrier proteins: within-animal slope (log2 protein per SD of element) and permutation p (best 15 over all element x carrier pairs):")
display(car.sort_values("p").head(15)[["element", "label", "slope", "t", "p"]].round(4))
''')

md("""
# 2. Element ratios and the whole ionome
**Idea.** Absolute plasma concentrations share a sample-level scale (dilution, hematocrit, handling), and biology often lives in *ratios* (Cu/Zn is an inflammation marker; Na/K and Ca/Mg reflect cell-membrane and mineral balance). We therefore (a) scan **log2 ratios**, (b) scan the **principal components of the log2 ionome** (PCA on all 37 samples), and (c) fit a **multivariate PLS model** that predicts the *whole within-animal ionome* from the proteome and is scored on **animals the model never saw** (leave-one-animal-out), with a permutation null.
""")
code('''
# (a) ratios and (b) ionome PCs
L_all = np.log2(el_all[ELEMENTS].astype(float)); mu, sd = L_all.mean(), L_all.std(ddof=1); Zall = (L_all - mu) / sd
U, s_, Vt = np.linalg.svd(Zall.values, full_matrices=False); var = s_ ** 2 / (s_ ** 2).sum()
load = pd.DataFrame(Vt[:3].T, index=ELEMENTS, columns=["PC1", "PC2", "PC3"])
for c in load: 
    if load[c].abs().idxmax() and load.loc[load[c].abs().idxmax(), c] < 0: load[c] *= -1; Vt[int(c[-1]) - 1] *= -1
print("ionome PCA variance explained:", (100 * var[:3]).round(1), "%"); display(load.round(2))
Zd = ((np.log2(d[ELEMENTS].astype(float)) - mu) / sd).values; PCs = Zd @ Vt[:3].T
RATIOS = {"Cu/Zn": ("Cu", "Zn"), "Fe/Cu": ("Fe", "Cu"), "Fe/Zn": ("Fe", "Zn"), "Na/K": ("Na", "K"), "Ca/Mg": ("Ca", "Mg"), "Ca/P": ("Ca", "P")}
VARS2 = {}
for k, (a, b) in RATIOS.items(): VARS2[k] = np.log2(d[a].values / d[b].values)
for i in range(3): VARS2[f"ionome PC{i+1}"] = PCs[:, i]
print("Scanning ratios and ionome PCs...")
for k, v in VARS2.items(): get_scan(k, v)
rows = []
for k in list(RATIOS) + [f"ionome PC{i+1}" for i in range(3)]:
    r = CACHE[k][0]; st = CACHE[k][1]; rows.append(dict(variable=k, p_lt_0_001=int((r.p < .001).sum()), expected_by_chance=round(len(r) * .001, 1), q_lt_0_10=int((r.q < .10).sum()), best_set=st.sort_values("p_any_association").iloc[0].set, best_set_p=st.p_any_association.min()))
for el in ELEMENTS:
    r = CACHE[el][0]; st = CACHE[el][1]; rows.append(dict(variable=el, p_lt_0_001=int((r.p < .001).sum()), expected_by_chance=round(len(r) * .001, 1), q_lt_0_10=int((r.q < .10).sum()), best_set=st.sort_values("p_any_association").iloc[0].set, best_set_p=st.p_any_association.min()))
T2 = pd.DataFrame(rows); T2.to_csv(f"{OUT}/2_ratios_ionome_summary.csv", index=False); display(T2.round(4))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
x = np.arange(len(T2)); ax[0].bar(x, T2.p_lt_0_001, color=[BLUE if v in RATIOS or v.startswith("ionome") else LBLUE for v in T2.variable]); ax[0].axhline(11, color="k", ls=":", lw=1)
ax[0].set_xticks(x, T2.variable, rotation=60, ha="right"); ax[0].set_ylabel("# proteins with within-animal p<0.001"); ax[0].set_title("Excess over chance (dotted = 11 expected)\\ndark = ratios / ionome PCs, light = single elements")
for g, c in zip(["Na", "Mg", "P", "K", "Ca", "Fe", "Cu", "Zn"], [GREY, BLUE, ORANGE, "#6250d6", "#1baf7a", "#e34948", "#eda100", NAVY]):
    ax[1].arrow(0, 0, load.loc[g, "PC1"], load.loc[g, "PC2"], color=c, head_width=.02); ax[1].text(load.loc[g, "PC1"] * 1.08, load.loc[g, "PC2"] * 1.08, g, color=c, fontsize=9)
ax[1].set_xlabel(f"ionome PC1 ({100*var[0]:.0f}%)"); ax[1].set_ylabel(f"ionome PC2 ({100*var[1]:.0f}%)"); ax[1].set_title("Ionome loadings"); ax[1].set_xlim(-.9, .9); ax[1].set_ylim(-.9, .9); fig.tight_layout(); plt.show()
''')
code('''
# (c) multivariate PLS: whole within-animal ionome from the proteome, leave-one-animal-out, permutation null
from sklearn.cross_decomposition import PLSRegression
dW = W.d; Ion = (np.log2(dW[ELEMENTS].astype(float)) - mu) / sd; Ionw = W.demean(Ion.values); Ionw = Ionw / Ionw.std(0)
def pls_cv(Ionw, ncomp=2):
    pred = np.full_like(Ionw, np.nan)
    for h in sorted(set(W.animal)):
        te = W.animal == h; tr = ~te
        Bt = np.linalg.lstsq(W.T[tr], np.column_stack([Ionw[tr], W.Yw[tr]]), rcond=None)[0]                 # time trend from training animals only
        Ytr = Ionw[tr] - W.T[tr] @ Bt[:, :Ionw.shape[1]]; Xtr = W.Yw[tr] - W.T[tr] @ Bt[:, Ionw.shape[1]:]; Xte = W.Yw[te] - W.T[te] @ Bt[:, Ionw.shape[1]:]
        m_, s_x = Xtr.mean(0), Xtr.std(0) + 1e-9; pls = PLSRegression(n_components=ncomp, scale=False).fit((Xtr - m_) / s_x, Ytr); pred[te] = pls.predict((Xte - m_) / s_x)
    score = np.empty_like(Ionw)
    for h in sorted(set(W.animal)):
        te = W.animal == h; Bt = np.linalg.lstsq(W.T[~te], Ionw[~te], rcond=None)[0]; score[te] = Ionw[te] - W.T[te] @ Bt
    r = np.array([stats.spearmanr(pred[:, j], score[:, j])[0] for j in range(Ionw.shape[1])]); return r
r_obs = pls_cv(Ionw); rng = np.random.default_rng(1); null = []
for _ in range(B_PLS):
    Ip = Ionw.copy()
    for g, r in W.members.items(): Ip[r] = Ionw[rng.permutation(r)]                                  # shuffle whole ionome rows within animal
    null.append(pls_cv(Ip))
null = np.array(null)
res_pls = pd.DataFrame({"element": ELEMENTS, "spearman_pred_vs_obs": r_obs, "null_95pct": np.quantile(null, .95, axis=0), "perm_p": [(1 + (null[:, j] >= r_obs[j]).sum()) / (1 + B_PLS) for j in range(len(ELEMENTS))]})
mean_r = np.nanmean(r_obs); pmean = (1 + (np.nanmean(null, axis=1) >= mean_r).sum()) / (1 + B_PLS)
res_pls.to_csv(f"{OUT}/2_PLS_ionome_leave_one_animal_out.csv", index=False); display(res_pls.round(3)); print(f"mean Spearman over elements = {mean_r:.2f}  (permutation p = {pmean:.3f})")
fig, ax = plt.subplots(figsize=(7, 3.6)); x = np.arange(len(ELEMENTS)); ax.bar(x - .2, res_pls.null_95pct, .4, color="#c5c4be", label="95th pct of chance"); ax.bar(x + .2, res_pls.spearman_pred_vs_obs, .4, color=BLUE, label="PLS, unseen animals")
ax.set_xticks(x, ELEMENTS); ax.axhline(0, color="k", lw=.7); ax.set_ylabel("Spearman (predicted vs observed)"); ax.legend(frameon=False); ax.set_title("Multivariate: proteome -> within-animal ionome"); plt.show()
''')

md("""
# 3. Recovery toward naive
**Idea.** The 7 naive animals define "normal". For each infected sample (all 30, including the 3 with no proteomics) we compute, per element, the **z-score against the naive animals** (log2 scale) and an overall **distance from naive** = root-mean-square z across the 8 elements. We then ask (a) does the distance fall with ART stage / time / viral load (mixed model, animal random intercept), and (b) which proteins and protein sets track the distance within animals?
*Caveat:* naive animals are a different cohort, so the **absolute** distance contains a cohort offset — interpret **changes** across stage/time within animals, not the level.
""")
code('''
import statsmodels.formula.api as smf
L = np.log2(el_all[ELEMENTS].astype(float)); nv = el_all.naive.values; mu_n, sd_n = L[nv].mean(), L[nv].std(ddof=1)
Zn = (L - mu_n) / sd_n; el_all["dist_naive"] = np.sqrt((Zn ** 2).mean(axis=1))
for e in ELEMENTS: el_all[f"z_{e}"] = Zn[e]
inf_ = el_all[~nv].copy(); inf_["week"] = inf_.week.astype(float); inf_["logw"] = np.log2(inf_.week + 1)
inf_["group"] = pd.Categorical(inf_["group"], STAGES)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    m1 = smf.mixedlm("dist_naive ~ C(group)", inf_, groups=inf_["animal"]).fit(reml=True); m2 = smf.mixedlm("dist_naive ~ logw", inf_, groups=inf_["animal"]).fit(reml=True)
print("Distance from naive (RMS z over elements): naive animals mean =", round(el_all.loc[nv, "dist_naive"].mean(), 2))
display(inf_.groupby("group").dist_naive.agg(["count", "mean", "std"]).round(2)); print(m1.summary().tables[1]); print("\\ntrend with log2(week+1): slope %.3f  p=%.3f (mixed model, animal intercept)" % (m2.params["logw"], m2.pvalues["logw"]))
order = pd.concat([inf_.sort_values(["animal", "week"]), el_all[nv]]); Zh = order[[f"z_{e}" for e in ELEMENTS]].values
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.3, 1]}); im = ax[0].imshow(Zh, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto"); ax[0].grid(False)
ax[0].set_xticks(range(8), ELEMENTS); ax[0].set_yticks(range(len(order)), [f"{a} wk{int(w)}" if not np.isnan(w) else a for a, w in zip(order.animal, order.week)], fontsize=6.5); ax[0].set_title("Element z-score vs the naive animals (log2)"); fig.colorbar(im, ax=ax[0], shrink=.6)
for a_, sub in inf_.groupby("animal"): sub = sub.sort_values("week"); ax[1].plot(sub.logw, sub.dist_naive, "-o", ms=4, lw=1, label=a_)
ax[1].set_xlabel("log2(week+1)"); ax[1].set_ylabel("distance from naive (RMS z)"); ax[1].legend(fontsize=7, frameon=False, ncol=2); ax[1].set_title("Does each animal move toward / away from naive?"); fig.tight_layout(); plt.show()
el_all.to_csv(f"{OUT}/3_recovery_scores_all37.csv", index=False)
''')
code('''
# (b) which proteins / sets track the distance within animals?  (27 samples with proteomics)
dist27 = el_all.set_index("ID").loc[d.index, "dist_naive"].values
res, st = get_scan("distance_from_naive", dist27)
display(st.sort_values("p_any_association").head(10)[["set", "n_proteins", "mean_t", "frac_positive", "p_direction", "p_any_association"]].round(4))
print("top proteins tracking the distance from naive (within-animal):"); display(res.head(15)[["label", "slope", "t", "p", "q"]].round(4))
# does the distance track viral load / stage in the 27 proteomics samples? (within animal)
dw = W.demean(((dist27 - dist27.mean()) / dist27.std())[W.idx][:, None])[:, 0]; vlw = W.demean(((d.Log10VL.values - d.Log10VL.mean()) / d.Log10VL.std())[W.idx][:, None])[:, 0]
print("within-animal correlation distance vs log10 viral load: r=%.2f" % np.corrcoef(dw, vlw)[0, 1])
''')

md("""
# 4. Lead / lag
**Idea.** If the element drives (or reflects an early step of) a protein change, the element at the *previous* sample should predict the protein at the *current* sample beyond the protein's own previous value — and not the reverse. For every consecutive pair of samples within an animal we compute two partial correlations (time trend and animal baseline removed):
`lead_E = r(E[i-1], P[i] | P[i-1])` (element leads protein) and `lead_P = r(P[i-1], E[i] | E[i-1])` (protein leads element). The **asymmetry** `lead_E − lead_P`, signed by the sign of the same-time association, is summarised over the proteins most associated with the element at the same time (top 100) and over metal/cell sets. Null: the **order of the samples is shuffled within each animal** (same shuffle for protein and element), which removes temporal order while keeping everything else.
*Power is very low (≈19 consecutive pairs, irregular gaps) — treat as exploratory.* Positive asymmetry = element leads; negative = protein leads.
""")
code('''
def residualise(M):   # remove time trend (common) from within-animal-demeaned data
    B_ = np.linalg.lstsq(W.T, M, rcond=None)[0]; return M - W.T @ B_
Prs = residualise(W.Yw); order_idx = {g: m[np.argsort(W.d["week"].values[m])] for g, m in W.members.items() if len(m) >= 3}
def pairs_from(order):   # list of (prev, cur) index arrays
    pv, cu = [], []
    for g, m in order.items(): pv += list(m[:-1]); cu += list(m[1:])
    return np.array(pv), np.array(cu)
def pcorr_vec(x, Yv, z):          # partial correlation of x(n,) with Yv(n,F) given z(n,F)
    def c(a, b): a = a - a.mean(0); b = b - b.mean(0); return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0) + 1e-12)
    x2 = np.repeat(x[:, None], Yv.shape[1], 1); rxy, rxz, ryz = c(x2, Yv), c(x2, z), c(Yv, z); return (rxy - rxz * ryz) / np.sqrt(np.clip((1 - rxz ** 2) * (1 - ryz ** 2), 1e-9, None))
def asym_stats(e, cols, order):
    pv, cu = pairs_from(order); P = Prs[:, cols]
    leadE = pcorr_vec(e[pv], P[cu], P[pv])                                   # element(prev) -> protein(cur) | protein(prev)
    # protein(prev) -> element(cur) | element(prev): vectorised over proteins
    def c(a, b): a = a - a.mean(0); b = b - b.mean(0); return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0) + 1e-12)
    ec, ep = e[cu], e[pv]; rxy = c(P[pv], np.repeat(ec[:, None], P.shape[1], 1)); rxz = c(P[pv], np.repeat(ep[:, None], P.shape[1], 1)); ryz = np.corrcoef(ec, ep)[0, 1]
    leadP = (rxy - rxz * ryz) / np.sqrt(np.clip((1 - rxz ** 2) * (1 - ryz ** 2), 1e-9, None)); return leadE - leadP, len(pv)
rng = np.random.default_rng(5); rows = []; groups_for_lag = {"top100 same-time proteins": None, **{k: v for k, v in SETS.items() if k.startswith(("UniProt: Copper", "UniProt: Iron", "UniProt: Zinc", "Curated: Cu", "Curated: Fe", "Control: red", "Control: plat"))}}
for el in ELEMENTS:
    e = residualise(W.demean(zs(W.d[el].values)[:, None]))[:, 0]; r_ = CACHE[el][0]; colpos = {f: i for i, f in enumerate(W.cols)}; top = np.array([colpos[f] for f in r_.feature.values[:100]])
    sgn = np.sign(r_.set_index("feature").loc[W.cols[top], "slope"].values)
    for gname, cols in groups_for_lag.items():
        cols_ = top if cols is None else cols
        sg = sgn if cols is None else np.sign(CACHE[el][0].set_index("feature").loc[W.cols[cols_], "slope"].values)
        a_obs, npairs = asym_stats(e, cols_, order_idx); obs = float(np.mean(a_obs * sg)); nul = []
        for _ in range(B_LAG // 10):                                          # shuffled sample order within animal (10x fewer for speed; scale B_LAG up for final)
            sh = {g: rng.permutation(m) for g, m in order_idx.items()}; an, _ = asym_stats(e, cols_, sh); nul.append(float(np.mean(an * sg)))
        nul = np.array(nul); rows.append(dict(element=el, protein_group=gname, n_pairs=npairs, mean_asymmetry=obs, null_sd=nul.std(), perm_p_two_sided=(1 + (np.abs(nul) >= abs(obs)).sum()) / (1 + len(nul))))
LAG = pd.DataFrame(rows); LAG["q"] = bh(LAG.perm_p_two_sided.values); LAG.to_csv(f"{OUT}/4_lead_lag.csv", index=False)
print("asymmetry > 0: element leads protein; < 0: protein leads element (n_pairs = %d)" % LAG.n_pairs.iloc[0]); display(LAG.sort_values("perm_p_two_sided").head(12).round(4))
Lt = LAG[LAG.protein_group == "top100 same-time proteins"]; fig, ax = plt.subplots(figsize=(7, 3.4)); ax.bar(Lt.element, Lt.mean_asymmetry, color=[BLUE if v > 0 else ORANGE for v in Lt.mean_asymmetry]); ax.errorbar(np.arange(len(Lt)), np.zeros(len(Lt)), yerr=1.96 * Lt.null_sd.values, fmt="none", color="k", capsize=3)
ax.axhline(0, color="k", lw=.7); ax.set_ylabel("asymmetry (element leads  >0)"); ax.set_title("Lead/lag for the 100 proteins most associated with each element\\n(whiskers = 95% range expected by chance)"); plt.show()
''')

md("""
# Summary and downloads
""")
code('''
summary = []
for el in ELEMENTS:
    r = CACHE[el][0]; st = CACHE[el][1]; hyp = st[st.set.isin(HYP.get(el, []))].sort_values("p_any_association")
    ps = res_pls.set_index("element").loc[el]; lg = LAG[(LAG.element == el) & (LAG.protein_group == "top100 same-time proteins")].iloc[0]
    summary.append(dict(element=el, proteins_p_lt_0_001=int((r.p < .001).sum()), chance=round(len(r) * .001), proteins_q_lt_0_10=int((r.q < .10).sum()),
                        best_own_set=(hyp.iloc[0].set if len(hyp) else ""), best_own_set_p=(hyp.iloc[0].p_any_association if len(hyp) else np.nan),
                        PLS_unseen_animals_spearman=ps.spearman_pred_vs_obs, PLS_p=ps.perm_p, lead_lag_asymmetry=lg.mean_asymmetry, lead_lag_p=lg.perm_p_two_sided))
SUM = pd.DataFrame(summary); SUM.to_csv(f"{OUT}/SUMMARY_by_element.csv", index=False); display(SUM.round(3))
shutil.make_archive("elements_proteome_results", "zip", OUT); print("results written to", OUT, "and elements_proteome_results.zip")
if IN_COLAB:
    from google.colab import files; files.download("elements_proteome_results.zip")
''')

nb = nbf.v4.new_notebook(); nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}, "colab": {"provenance": []}}
nbf.write(nb, "Elements_Proteome_Colab.ipynb"); print("written", len(C), "cells")
