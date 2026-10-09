"""Chart data: mean EO-gap reduction over the 7 targets with all budgets, with bootstrap CI over seeds."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, glob, json
from fairlab.analysis import choose
e2 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/e2_*.csv"))]).drop_duplicates(["dataset", "attr", "seed", "method", "strategy", "nlab", "mu"])
e2 = e2[~((e2.dataset == "german") & (e2.attr == "age"))]
sel = choose(e2, 0.01, "T1"); sel["t"] = sel.dataset + "-" + sel.attr
sel["strategy"] = sel.strategy.fillna("-")
rng = np.random.default_rng(1)
def stat(g):
    # g: rows for one arm; per target % reduction then mean over targets; bootstrap seeds within target
    T = sorted(g.t.unique()); per = {t: (g[g.t == t].eo_gap.values, g[g.t == t].eo_gap0.values) for t in T}
    est = np.mean([100 * (1 - a.mean() / b.mean()) for a, b in per.values()])
    bs = []
    for _ in range(2000):
        v = []
        for a, b in per.values():
            i = rng.integers(0, len(a), len(a)); v.append(100 * (1 - a[i].mean() / b[i].mean()))
        bs.append(np.mean(v))
    return round(est, 1), round(np.percentile(bs, 2.5), 1), round(np.percentile(bs, 97.5), 1), len(T)
rows = []
for (m, s, n), g in sel.groupby(["method", "strategy", "nlab"]):
    if m == "proxy_soft" and s in ("random", "random_mnar"):
        e, lo, hi, k = stat(g); rows.append(dict(arm={"random": "Random sample", "random_mnar": "Biased disclosure"}[s], n=int(n), v=e, lo=lo, hi=hi, k=k))
    if m in ("oracle", "cluster_rw", "arl", "cvar", "jtt"):
        e, lo, hi, k = stat(g); rows.append(dict(arm=m, n=0, v=e, lo=lo, hi=hi, k=k))
print(json.dumps(rows))
