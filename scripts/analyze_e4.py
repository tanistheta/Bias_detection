"""H6: does the pre-deployment agreement check predict when proxy correction backfires?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, glob, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from fairlab.analysis import choose

R = "results/"
e2 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(R + "e2_*.csv"))])
e2 = e2.drop_duplicates(["dataset", "attr", "seed", "method", "strategy", "nlab", "mu"])
e2 = e2[(e2.method == "erm") | ((e2.method == "proxy_soft") & (e2.strategy == "random"))]
e4 = pd.read_csv(R + "e4b.csv")
e4 = e4[e4.method.isin(["erm", "proxy_soft"])].copy(); e4["strategy"] = e4.method.map({"proxy_soft": "random"})


def diag(df, est_col):
    cos = (df.lab_tpr_s * df.q_tpr_s + df.lab_fpr_s * df.q_fpr_s) / (
        np.hypot(df.lab_tpr_s, df.lab_fpr_s) * np.hypot(df.q_tpr_s, df.q_fpr_s) + 1e-12)
    return pd.DataFrame(dict(dataset=df.dataset, attr=df.attr, seed=df.seed, nlab=df.nlab, cos=cos, est=df[est_col])).drop_duplicates(["dataset", "attr", "seed", "nlab"])


rows = []
for src, d, est_col in ((e2, e2, "est_naive"), (e4, e4, "est_eo_lab")):
    sel = choose(d, 0.01, "T1")
    sel = sel[sel.method == "proxy_soft"]
    dg = diag(d[d.method == "proxy_soft"], est_col)
    rows.append(sel.merge(dg, on=["dataset", "attr", "seed", "nlab"]))
r = pd.concat(rows)
r["t"] = r.dataset + "-" + r.attr
r["harm"] = r.eo_gap > r.eo_gap0 + 0.01
r["gain"] = r.eo_gap0 - r.eo_gap
r["guard"] = (r.cos > 0) & (r.est >= 0.05)
r["eo_guarded"] = np.where(r.guard, r.eo_gap, r.eo_gap0)
r.to_csv(R + "final_e4_rows.csv", index=False)
print("pairs:", r.t.nunique(), "rows:", len(r))
for n in (100, 400):
    g = r[r.nlab == n].dropna(subset=["cos"])
    auc = roc_auc_score(g.harm, -g.cos) if g.harm.nunique() == 2 else np.nan
    pt = g.groupby("t").apply(lambda x: pd.Series(dict(
        red=100 * (1 - x.eo_gap.mean() / x.eo_gap0.mean()), red_g=100 * (1 - x.eo_guarded.mean() / x.eo_gap0.mean()),
        harm=x.harm.mean(), harm_g=(x.eo_guarded > x.eo_gap0 + 0.01).mean(), applied=x.guard.mean(), cos=x.cos.mean())))
    print(f"\n=== n={n}: AUROC(-cos -> harm) = {auc:.3f}; harm rate {g.harm.mean():.3f}")
    print(pt.round(2).to_string())
    keep = pt.red_g.mean() / pt.red.mean() if pt.red.mean() > 0 else np.nan
    print(f"mean reduction unguarded {pt.red.mean():.1f}% -> guarded {pt.red_g.mean():.1f}%  (kept {100*keep:.0f}%);"
          f" harm rate {pt.harm.mean():.3f} -> {pt.harm_g.mean():.3f}")
    print("H6:", "SUPPORTED" if (auc > 0.70 and pt.harm_g.mean() < pt.harm.mean() and keep >= 0.8) else "NOT SUPPORTED",
          f"(AUROC>0.70: {auc > 0.70}; harm cut: {pt.harm_g.mean() < pt.harm.mean()}; kept>=80%: {keep >= 0.8})")
    pt.to_csv(R + f"final_e4_n{n}.csv")
