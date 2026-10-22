"""Out-of-sample test: can the proteome predict an animal's WITHIN-animal element fluctuations in an animal never seen in training?
Outcome and proteins are centred within each animal and shared time trends (t, t^2) are removed; leave-one-animal-out;
in each training fold: top-K proteins by |correlation| (screening done inside the fold) + ridge (lambda by inner leave-one-sample-out).
Significance: element shuffled within animals, whole pipeline repeated (B draws).   usage: python predict_loao.py --data-dir D --out-dir results_v2"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import full_analysis as fa, within_animal_models as W

def ridge_fit_predict(Xtr, ytr, Xte, lambdas=np.logspace(-1, 4, 16)):
    U, s, Vt = np.linalg.svd(Xtr, full_matrices=False); Uy = U.T @ ytr; best = (np.inf, None)
    for lam in lambdas:                                   # closed-form leave-one-out for ridge
        f = s ** 2 / (s ** 2 + lam); H = (U * f) @ U.T; yhat = H @ ytr; loo = ((ytr - yhat) / (1 - np.diag(H) + 1e-9)) ** 2
        if loo.mean() < best[0]: best = (loo.mean(), lam)
    lam = best[1]; beta = Vt.T @ ((s / (s ** 2 + lam)) * Uy); return Xte @ beta, beta

def pipeline(D, e, K=100):
    N, g = D.N, D.g; sd = e.std(ddof=1); yw = D._within((e / sd)[:, None])[:, 0]; Yw = D.Yw; T = D.T
    pred = np.full(N, np.nan); sel = []
    for h in D.animals:
        te = g == h; tr = ~te
        B = np.linalg.lstsq(T[tr], np.column_stack([yw[tr], Yw[tr]]), rcond=None)[0]                  # time trend from training animals only
        ytr = yw[tr] - T[tr] @ B[:, 0]; Xtr = Yw[tr] - T[tr] @ B[:, 1:]; Xte = Yw[te] - T[te] @ B[:, 1:]
        mu, sg = Xtr.mean(0), Xtr.std(0) + 1e-9; Xtr = (Xtr - mu) / sg; Xte = (Xte - mu) / sg
        c = np.abs(Xtr.T @ (ytr - ytr.mean())) / (np.linalg.norm(Xtr, axis=0) * np.linalg.norm(ytr - ytr.mean()) + 1e-12); top = np.argsort(-c)[:K]
        p, beta = ridge_fit_predict(Xtr[:, top], ytr - ytr.mean(), Xte[:, top]); pred[te] = p; sel.append(top)
    yobs = yw - np.concatenate([[0]]) if False else yw
    resid_obs = yw.copy()
    # observed outcome after removing time trend learned on all other animals is what we score against
    score_y = np.empty(N)
    for h in D.animals:
        te = g == h; B = np.linalg.lstsq(T[~te], yw[~te], rcond=None)[0]; score_y[te] = yw[te] - T[te] @ B
    per = [stats.spearmanr(pred[g == h], score_y[g == h])[0] for h in D.animals]
    return stats.spearmanr(pred, score_y)[0], float(np.nanmean(per)), pred, score_y, sel

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir"); ap.add_argument("--out-dir", default="results_v2"); ap.add_argument("--elements", nargs="*", default=fa.ELS)
    ap.add_argument("--B", type=int, default=300); ap.add_argument("--K", type=int, default=100); a = ap.parse_args()
    d, Y, ann, xl, meta = fa.load(a.data_dir); D = W.Data(d, Y); out = Path(a.out_dir) / "within"; rows = []
    for el in a.elements:
        e0 = D.element(el); r_obs, r_per, pred, y, sel = pipeline(D, e0, a.K); rng = np.random.default_rng(sum(map(ord, el))); nul = []
        for _ in range(a.B):
            e = e0.copy()
            for an in D.animals:
                m = D.g == an; e[m] = rng.permutation(e[m])
            nul.append(pipeline(D, e, a.K)[0])
        nul = np.array(nul); p = (1 + (nul >= r_obs).sum()) / (1 + len(nul))
        stab = pd.Series(np.concatenate(sel)).value_counts(); stable = [ann.loc[D.cols[i], "label"] for i in stab[stab >= 4].index][:15]
        rows.append(dict(element=el, pooled_spearman_pred_vs_obs=r_obs, mean_per_animal_spearman=r_per, null_mean=nul.mean(), null_95=np.quantile(nul, .95), perm_p=p, n_proteins_selected_in_4plus_of_5_folds=int((stab >= 4).sum()), examples=", ".join(stable)))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items() if k != "examples"})
        pd.DataFrame({"sample": D.d.index, "animal": D.g, "observed_within_element_time_adjusted": y, "predicted_by_proteome": pred}).to_csv(out / f"prediction_{el}.csv", index=False, float_format="%.4g")
    pd.DataFrame(rows).to_csv(out / "prediction_loao_summary.csv", index=False, float_format="%.4g")
