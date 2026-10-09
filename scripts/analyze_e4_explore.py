"""EXPLORATORY (not pre-registered): guard on the proxy-weighted gap estimate (E5's accurate estimator)."""
import numpy as np, pandas as pd
r = pd.read_csv("results/final_e4_rows.csv")
e2 = pd.concat([pd.read_csv(f) for f in ["results/e2_a.csv", "results/e2_b.csv", "results/e2_c.csv", "results/e2_d.csv"]])
e4 = pd.read_csv("results/e4b.csv")
q = pd.concat([e2[(e2.method == "proxy_soft") & (e2.strategy == "random")], e4[e4.method == "proxy_soft"]])
q = q.assign(est_proxy=np.maximum(q.q_tpr_s.abs(), q.q_fpr_s.abs()))[["dataset", "attr", "seed", "nlab", "est_proxy"]].drop_duplicates(["dataset", "attr", "seed", "nlab"])
r = r.merge(q, on=["dataset", "attr", "seed", "nlab"])
out = []
for n in (100, 400):
    for thr in (0.03, 0.05, 0.08):
        g = r[r.nlab == n].copy()
        g["app"] = g.est_proxy >= thr
        g["eo_g"] = np.where(g.app, g.eo_gap, g.eo_gap0)
        pt = g.groupby("t").apply(lambda x: pd.Series(dict(red=100 * (1 - x.eo_gap.mean() / x.eo_gap0.mean()), red_g=100 * (1 - x.eo_g.mean() / x.eo_gap0.mean()),
                                                          harm=x.harm.mean(), harm_g=(x.eo_g > x.eo_gap0 + 0.01).mean(), applied=x.app.mean())))
        out.append(dict(n=n, thr=thr, red=pt.red.mean(), red_g=pt.red_g.mean(), harm=pt.harm.mean(), harm_g=pt.harm_g.mean(), applied=pt.applied.mean()))
        if thr == 0.05:
            print(f"n={n} thr={thr}"); print(pt.round(2).to_string())
o = pd.DataFrame(out); print(o.round(3).to_string()); o.to_csv("results/explore_gap_guard.csv", index=False)
