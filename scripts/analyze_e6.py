"""H7: neural check. T1 tuning (1-point pool-AUC budget)."""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from fairlab.analysis import choose
d = pd.concat([pd.read_csv(f) for f in ["results/e6_part1.csv", "results/e6_part2.csv"] + sorted(glob.glob("results/e6_[cd].csv"))])
d = d.drop_duplicates(["dataset", "attr", "seed", "method", "nlab", "mu"])
sel = choose(d, 0.01, "T1")
sel["arm"] = np.where(sel.method == "proxy_soft", "proxy n=" + sel.nlab.astype(int).astype(str), sel.method)
sel["t"] = sel.dataset + "-" + sel.attr
r = sel.groupby(["t", "arm"]).apply(lambda g: pd.Series(dict(red=100 * (1 - g.eo_gap.mean() / g.eo_gap0.mean()), dauc=100 * (g.auc - g.auc0).mean(), seeds=len(g)))).reset_index()
pv = r.pivot_table(index="arm", columns="t", values="red")
pv["MEAN"] = pv.mean(axis=1)
print(pv.round(1).to_string())
print(r.groupby("t").seeds.max().to_string())
both = pv.drop(columns="MEAN").loc[["proxy n=400", "arl", "jtt"]].dropna(axis=1)
m = both.mean(axis=1)
print("H7 (matched datasets):", m.round(1).to_dict(), "->", "SUPPORTED" if m["proxy n=400"] > max(m["arl"], m["jtt"]) else "NOT SUPPORTED")
pv.to_csv("results/final_e6.csv")
print(r.pivot_table(index="arm", columns="t", values="dauc").round(2).to_string())
