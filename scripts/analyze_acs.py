"""ACS replication: absolute EO-gap changes, gap-size dependence, backfire guard, E5."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, glob
from scipy.stats import spearmanr
from fairlab.analysis import choose
pd.set_option("display.width", 250)
def load(pat):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/" + pat))])
    return d.drop_duplicates(["dataset", "attr", "seed", "method", "strategy", "nlab", "mu"])
acs, bench = load("acs_*.csv"), load("e2_*.csv")
def arm(r):
    return f"proxy n={int(r.nlab)}" if (r.method == "proxy_soft" and r.strategy == "random") else (r.method if r.method in ("oracle", "arl", "cvar", "jtt", "cluster_rw") else None)
out = []
for name, d in (("ACS", acs), ("Benchmarks", bench)):
    s = choose(d, 0.01, "T1"); s["arm"] = s.apply(arm, axis=1); s = s.dropna(subset=["arm"])
    s["t"] = s.dataset.str.replace("_2018", "") + "-" + s.attr
    s["abs_red"] = s.eo_gap0 - s.eo_gap
    s["harm"] = s.eo_gap > s.eo_gap0 + 0.01
    s["family"] = name
    out.append(s)
s = pd.concat(out)
g = s.groupby(["family", "t", "arm"]).agg(eo0=("eo_gap0", "mean"), abs_red=("abs_red", "mean"), pct=("eo_gap", "mean"), harm=("harm", "mean")).reset_index()
g["pct"] = 100 * (1 - g.pct / g.eo0)
keep = ["cluster_rw", "proxy n=100", "proxy n=400", "oracle"]
print("=== ACS: absolute EO-gap reduction (points x100) and harm rate")
a = g[(g.family == "ACS") & g.arm.isin(keep)]
print((a.pivot_table(index="t", columns="arm", values="abs_red") * 100).round(2).join(a.groupby("t").eo0.first().round(3)).to_string())
print(a.pivot_table(index="t", columns="arm", values="harm").round(2).to_string())
# gap-size dependence across both families (25-ish targets): proxy 400 / oracle success vs baseline gap
for armn in ("proxy n=100", "proxy n=400", "oracle"):
    x = g[g.arm == armn]
    rho = spearmanr(x.eo0, x.abs_red).correlation
    small = x[x.eo0 < 0.05]; big = x[x.eo0 >= 0.05]
    print(f"{armn}: spearman(baseline gap, abs reduction) = {rho:.2f} over {len(x)} targets; "
          f"mean pct reduction gap<0.05: {small.pct.mean():.1f}% (n={len(small)}), gap>=0.05: {big.pct.mean():.1f}% (n={len(big)}); "
          f"harm gap<0.05 {small.harm.mean():.2f}, >=0.05 {big.harm.mean():.2f}")
# ACS excluding the no-gap target (NY, 0.022)
ab = a[a.eo0 >= 0.05].pivot_table(index="t", columns="arm", values="pct")
print("=== ACS targets with baseline gap >= 0.05, % reduction:"); print(ab.round(1).to_string()); print("mean", ab.mean().round(1).to_dict())
# gap guard on ACS (threshold 0.08 fixed before ACS was analysed)
p = acs[(acs.method == "proxy_soft") & (acs.strategy == "random")].copy()
p["est_proxy"] = np.maximum(p.q_tpr_s.abs(), p.q_fpr_s.abs())
est = p.drop_duplicates(["dataset", "attr", "seed", "nlab"])[["dataset", "attr", "seed", "nlab", "est_proxy"]]
sa = s[(s.family == "ACS") & s.arm.str.startswith("proxy")].merge(est, on=["dataset", "attr", "seed", "nlab"])
for n in (100, 400):
    for thr in (0.05, 0.08):
        x = sa[sa.nlab == n].copy(); x["app"] = x.est_proxy >= thr; x["eo_g"] = np.where(x.app, x.eo_gap, x.eo_gap0)
        per = x.groupby("t").apply(lambda q: pd.Series(dict(red=100 * (1 - q.eo_gap.mean() / q.eo_gap0.mean()), red_g=100 * (1 - q.eo_g.mean() / q.eo_gap0.mean()),
                                                           harm=q.harm.mean(), harm_g=(q.eo_g > q.eo_gap0 + 0.01).mean(), applied=q.app.mean())))
        print(f"guard n={n} thr={thr}: mean red {per.red.mean():.1f} -> {per.red_g.mean():.1f}; harm {per.harm.mean():.2f} -> {per.harm_g.mean():.2f}; applied {per.applied.mean():.2f}")
        if thr == 0.08: print(per.round(2).to_string())
# E5 on ACS per n
e5 = p[p.mu == 0.3].drop_duplicates(["dataset", "attr", "seed", "nlab"])
print("=== ACS E5"); print(e5.assign(en=(e5.est_naive - e5.true_eo).abs(), ep=(e5.est_proxy - e5.true_eo).abs()).groupby("nlab")[["en", "ep"]].mean().round(3).to_string())
g.to_csv("results/acs_abs_summary.csv", index=False)
