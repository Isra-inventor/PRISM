"""
Multi-protein / element relationships (exploratory, all samples kept).

 A. MODULES   co-varying protein modules (hierarchical clustering on |correlation|), module eigengene ~ element
              (M1 stage / M3 week LMMs).  ~30 tests instead of 11,000.
 B. PAIRS     exhaustive screen of ALL protein pairs (A,B): do two proteins together track the element better than
              either alone?  Statistic = R2(E ~ A + B) - max(R2(E~A), R2(E~B))  after removing stage and animal
              (GLS-whitened).  Significance by permutation of the max statistic over all ~60M pairs (FWER).
 C. COUPLING  among the proteins most associated with the element: does the A->B relationship depend on the element?
              B ~ stage + A + E + A:E (+ animal), BH-FDR over pairs.

usage: python tri_analysis.py --data-dir <dir> --out-dir results_v2 [--elements Fe Cu] [--perms 99]
"""
import argparse, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import full_analysis as fa
from lmm_engine import fit_reml, term_tests, bh

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ helpers
def whitener(groups, lam):
    n = len(groups); W = np.eye(n)
    for g in np.unique(groups):
        i = np.where(groups == g)[0]; k = len(i)
        W[np.ix_(i, i)] -= (1 - (1 + lam * k) ** -0.5) / k * np.ones((k, k))
    return W


def residual_space(d, Y, el):
    """whitened + stage-removed element and protein matrices (columns unit norm)."""
    g = d.nhp_id.values; S = fa.stage_dummies(d).values; X = np.column_stack([np.ones(len(d)), S])
    E = d[el].values.astype(float)
    lam = fit_reml(X, E[:, None], g)["lam"][0]
    W = whitener(g, lam); Xw = W @ X
    Q = np.eye(len(d)) - Xw @ np.linalg.pinv(Xw)
    e = Q @ (W @ E); A = Q @ (W @ Y.values)
    e = e / np.linalg.norm(e)
    nrm = np.linalg.norm(A, axis=0); ok = nrm > 1e-8
    A = A[:, ok] / nrm[ok]
    return e, A.astype(np.float32), ok, lam


def pair_stat_max(r, R, mask_abs=0.9, chunk=400, return_top=0):
    """max over pairs of Delta = R2pair - max(rA^2, rB^2); optionally top-k pairs."""
    F = len(r); best = -1.0; tops = []
    r = r.astype(np.float32); r2 = r * r
    for i0 in range(0, F, chunk):
        i1 = min(F, i0 + chunk)
        rab = R[i0:i1, i0:]                    # columns j >= i0 (upper triangle incl. block diagonal)
        rA = r[i0:i1, None]; rB = r[None, i0:]
        den = 1 - rab * rab
        with np.errstate(divide="ignore", invalid="ignore"):
            R2 = (r2[i0:i1, None] + r2[None, i0:] - 2 * rA * rB * rab) / den
        delta = R2 - np.maximum(r2[i0:i1, None], r2[None, i0:])
        bad = np.abs(rab) >= mask_abs
        delta[bad] = -1
        ii = np.arange(i1 - i0)[:, None]; jj = np.arange(i0, F)[None, :]
        delta[jj <= (ii + i0)] = -1           # strict upper triangle
        m = float(delta.max())
        if m > best: best = m
        if return_top:
            flat = np.argpartition(delta.ravel(), -return_top)[-return_top:]
            for f_ in flat:
                a, b = divmod(int(f_), delta.shape[1])
                tops.append((float(delta[a, b]), i0 + a, i0 + b))
    if return_top:
        tops.sort(reverse=True); return best, tops[:return_top]
    return best


def _perm_job(args):
    seed, B = args; rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        out.append(pair_stat_max(_G["A"].T @ rng.permutation(_G["e"]), _G["R"]))
    return out


_G = {}


def pair_screen(d, Y, ann, el, B=99, topk=100, procs=4):
    e, A, ok, lam = residual_space(d, Y, el)
    feats = Y.columns[ok]
    R = (A.T @ A).astype(np.float32); np.fill_diagonal(R, 0)
    r = (A.T @ e).astype(np.float32)
    obs, top = pair_stat_max(r, R, return_top=topk)
    _G.update(A=A, e=e, R=R)
    seeds = [(1000 + k, int(np.ceil(B / procs))) for k in range(procs)]
    with Pool(procs) as pl: nulls = np.concatenate(pl.map(_perm_job, seeds))[:B]
    dfres = len(d) - 1 - fa.stage_dummies(d).shape[1]
    rows = []
    for delta, a, b in top:
        ra, rb, rab = float(r[a]), float(r[b]), float(R[a, b])
        r2 = (ra**2 + rb**2 - 2 * ra * rb * rab) / (1 - rab**2)
        # partial correlations of E with A given B and with B given A
        pa = (ra - rb * rab) / np.sqrt((1 - rb**2) * (1 - rab**2)); pb = (rb - ra * rab) / np.sqrt((1 - ra**2) * (1 - rab**2))
        rows.append(dict(element=el, protein_A=feats[a], label_A=ann.loc[feats[a], "label"], protein_B=feats[b], label_B=ann.loc[feats[b], "label"],
                         r_E_A=ra, r_E_B=rb, r_A_B=rab, R2_pair=r2, delta_R2=delta, partial_r_E_A_given_B=pa, partial_r_E_B_given_A=pb,
                         FWER_p=(1 + int((nulls >= delta).sum())) / (1 + len(nulls))))
    res = pd.DataFrame(rows)
    return res, dict(element=el, lam_E=lam, n_pairs=int(len(feats) * (len(feats) - 1) / 2), obs_max_delta=obs,
                     null_max_median=float(np.median(nulls)), null_max_95=float(np.quantile(nulls, .95)))


# ------------------------------------------------------------------ A. modules
def modules(d, Y, ann, min_size=15, top_var=6000, height=0.75):
    Zc = (Y - Y.mean()) / Y.std()
    keep = Y.var().sort_values(ascending=False).index[:top_var]
    C = np.corrcoef(Zc[keep].values.T); D = 1 - np.abs(C); np.fill_diagonal(D, 0)
    Lk = linkage(squareform(D, checks=False), "average"); lab = fcluster(Lk, height, "distance")
    sizes = pd.Series(lab).value_counts(); mods = [k for k, v in sizes.items() if v >= min_size]
    mem = pd.DataFrame({"feature": keep, "label": ann.loc[keep, "label"].values, "cluster": lab})
    mem = mem[mem.cluster.isin(mods)]
    remap = {k: f"M{i+1:02d}" for i, k in enumerate(sorted(mods, key=lambda k: -sizes[k]))}
    mem["module"] = mem.cluster.map(remap)
    eig = {}
    for m, sub in mem.groupby("module"):
        u, s, vt = np.linalg.svd(Zc[sub.feature].values, full_matrices=False); pc = u[:, 0] * s[0]
        if np.corrcoef(pc, Zc[sub.feature].mean(axis=1))[0, 1] < 0: pc = -pc
        eig[m] = pc
    eig = pd.DataFrame(eig, index=Y.index)
    return mem, eig


def module_tests(d, eig, mem, ann):
    g = d.nhp_id.values; S = fa.stage_dummies(d).values; n = len(d); one = np.ones(n); rows = []
    Ev = eig.apply(fa.z).values
    # module ~ stage / week (element free)
    fs = fit_reml(np.column_stack([one, S]), Ev, g); from lmm_engine import wald_f
    Fv, pF = wald_f(fs, [1, 2, 3], "bw")
    ft = fit_reml(np.column_stack([one, d.tz.values]), Ev, g); bt, st_, tt, dft, pt = term_tests(ft, None, 1, "bw")
    for el in fa.MAIN + ["Ni"]:
        E = fa.z(d[el].values)
        f1 = fit_reml(np.column_stack([one, E, S]), Ev, g); b1, s1, _, _, p1 = term_tests(f1, None, 1, fa.DF_E)
        f3 = fit_reml(np.column_stack([one, E, d.tz.values]), Ev, g); b3, s3, _, _, p3 = term_tests(f3, None, 1, fa.DF_E)
        for j, m in enumerate(eig.columns):
            rows.append(dict(module=m, element=el, beta_E_stage_model=b1[j], p_E_stage_model=p1[j], beta_E_week_model=b3[j], p_E_week_model=p3[j]))
    res = pd.DataFrame(rows)
    res["q_E_stage_model"] = bh(res.p_E_stage_model.values); res["q_E_week_model"] = bh(res.p_E_week_model.values)
    info = pd.DataFrame({"module": eig.columns, "n_proteins": mem.groupby("module").size().reindex(eig.columns).values,
                         "p_stage": pF, "p_week": pt, "beta_week": bt})
    info["q_stage"] = bh(info.p_stage.values); info["q_week"] = bh(info.p_week.values)
    top = {m: ", ".join(sub.label.iloc[:8]) for m, sub in mem.groupby("module")}
    info["example_proteins"] = info.module.map(top)
    return res, info


# ------------------------------------------------------------------ C. element-modulated coupling among top proteins
def coupling(d, Y, ann, el, pool_feats):
    g = d.nhp_id.values; S = fa.stage_dummies(d).values; one = np.ones(len(d)); E = fa.z(d[el].values)
    Yp = Y[pool_feats]; Zp = Yp.apply(fa.z).values; rows = []
    for ia, a in enumerate(pool_feats):
        A = Zp[:, ia]
        X = np.column_stack([one, S, A, E, A * E]); keep = [j for j in range(len(pool_feats)) if j != ia]
        f = fit_reml(X, Yp.values[:, keep], g)
        b, s_, tt, df, p = term_tests(f, X, 6, fa.DF_E); bA = f["beta"][4]
        for k, j in enumerate(keep):
            rows.append(dict(element=el, protein_B_outcome=pool_feats[j], label_B=ann.loc[pool_feats[j], "label"], protein_A_predictor=a,
                             label_A=ann.loc[a, "label"], beta_A=bA[k], beta_AxE=b[k], p_AxE=p[k]))
    res = pd.DataFrame(rows); res["q_AxE"] = bh(res.p_AxE.values)
    return res.sort_values("p_AxE")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default="."); ap.add_argument("--out-dir", default="results_v2")
    ap.add_argument("--elements", nargs="*", default=fa.MAIN); ap.add_argument("--perms", type=int, default=99)
    ap.add_argument("--pool", type=int, default=60); ap.add_argument("--skip-pairs", action="store_true")
    a = ap.parse_args(); out = Path(a.out_dir); (out / "tri").mkdir(parents=True, exist_ok=True)
    d, Y, ann, xl, meta = fa.load(a.data_dir)
    mem, eig = modules(d, Y, ann); mem.to_csv(out / "tri" / "modules_membership.csv", index=False); eig.to_csv(out / "tri" / "module_eigengenes.csv")
    mres, minfo = module_tests(d, eig, mem, ann)
    mres.to_csv(out / "tri" / "modules_vs_elements.csv", index=False, float_format="%.5g"); minfo.to_csv(out / "tri" / "modules_vs_stage_week.csv", index=False, float_format="%.5g")
    print(f"modules: {eig.shape[1]} (proteins in modules: {len(mem)}) | module~element q<.05: {(mres.q_E_stage_model < .05).sum()} (stage model), {(mres.q_E_week_model < .05).sum()} (week model)")
    print(mres.sort_values("p_E_stage_model").head(8).round(4).to_string())
    summ = []
    for el in a.elements:
        pe = pd.read_csv(out / "per_element" / f"proteomics_{el}.csv").sort_values("M1_stage_p_E")
        pool = [f for f in pe.feature.head(a.pool)]
        c = coupling(d, Y, ann, el, pool); c.head(200).to_csv(out / "tri" / f"coupling_{el}.csv", index=False, float_format="%.5g")
        print(f"{el}: element-modulated coupling among top-{a.pool}: pairs q<.05 = {(c.q_AxE < .05).sum()} / {len(c)}")
        if not a.skip_pairs:
            res, info = pair_screen(d, Y, ann, el, a.perms)
            res.to_csv(out / "tri" / f"protein_pairs_{el}.csv", index=False, float_format="%.5g"); summ.append(info)
            print(f"   pair screen: obs max dR2={info['obs_max_delta']:.3f} null95={info['null_max_95']:.3f} top FWER p={res.FWER_p.min():.3f}")
    if summ:
        f = out / "tri" / "pair_screen_summary.csv"; new = pd.DataFrame(summ)
        if f.exists():
            old = pd.read_csv(f); new = pd.concat([old[~old.element.isin(new.element)], new])
        order = {e: i for i, e in enumerate(fa.ELS)}; new = new.sort_values("element", key=lambda s: s.map(order))
        new.to_csv(f, index=False, float_format="%.4g")


if __name__ == "__main__":
    main()
