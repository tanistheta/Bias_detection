"""Confirmatory analysis: H1-H7 and E5. Writes results/final_*.csv and prints a report."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, warnings, glob
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from fairlab.analysis import choose, summarise, holm

R = "results/"
pd.set_option("display.width", 250)
PATTERN = sys.argv[1] if len(sys.argv) > 1 else "e2_*.csv"  # e.g. "acs_*.csv"
e2 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(R + PATTERN))])
e2 = e2.drop_duplicates(["dataset", "attr", "seed", "method", "strategy", "nlab", "mu"])
e2["t"] = e2.dataset + "-" + e2.attr
DFM = ["arl", "cvar", "jtt", "cluster_rw"]


def label(r):
    if r.method == "proxy_soft":
        return f"proxy[{r.strategy}] n={int(r.nlab)}"
    if r.method in ("proxy_confident", "related_feats"):
        return f"{r.method} n={int(r.nlab)}"
    return r.method


out = {}
for budget in (0.01, 0.005, 0.02):
    for rule in ("T1", "T2"):
        sel = choose(e2, budget, rule)
        sm = summarise(sel)
        sm["arm"] = sm.apply(label, axis=1)
        sm["t"] = sm.dataset + "-" + sm.attr
        out[(budget, rule)] = (sel, sm)
        sm.to_csv(R + f"final_summary_{PATTERN.split('_')[0]}_{rule}_{budget}.csv", index=False)

sel, sm = out[(0.01, "T1")]
piv = sm.pivot_table(index="arm", columns="t", values="eo_red")
piv["MEAN"] = piv.mean(axis=1)
piv["MEDIAN"] = piv.drop(columns="MEAN").median(axis=1)
order = DFM + [f"proxy[random] n={n}" for n in (25, 50, 100, 200, 400)] + [f"proxy[strat_y] n={n}" for n in (25, 50, 100, 200, 400)] + \
    [f"proxy[random_mnar] n={n}" for n in (25, 50, 100, 200, 400)] + [f"proxy_confident n={n}" for n in (100, 400)] + \
    [f"related_feats n={n}" for n in (100, 400)] + ["oracle"]
piv = piv.reindex([o for o in order if o in piv.index])
print("=== EO-gap reduction %, T1, AUC budget 0.01 (mean over 30 seeds)")
print(piv.round(1).to_string())
piv.to_csv(R + "final_main_table.csv")

# per-target significance for key arms (Holm over 8 targets)
print("\n=== Significance (one-sided Wilcoxon vs ERM, Holm over targets) + 95% bootstrap CI")
key = ["proxy[random] n=100", "proxy[random] n=400", "oracle"] + DFM
sig = sm[sm.arm.isin(key)].copy()
sig["p_holm"] = np.nan
for arm, g in sig.groupby("arm"):
    sig.loc[g.index, "p_holm"] = holm(g.p.values)
print(sig.pivot_table(index="arm", columns="t", values="p_holm").reindex(key).round(3).to_string())
print("significant targets (p_holm<0.05):", sig[sig.p_holm < 0.05].groupby("arm").size().to_dict())
sig.to_csv(R + "final_significance.csv", index=False)
print(sm[sm.arm.isin(["proxy[random] n=100", "oracle"])][["t", "arm", "eo_red", "ci_lo", "ci_hi", "dauc", "eo0"]].round(2).to_string())

# hypotheses
m = piv["MEAN"]
print("\n=== H1: proxy n=100 mean", round(m["proxy[random] n=100"], 1), "vs DF", {k: round(m[k], 1) for k in DFM},
      "->", "SUPPORTED" if all(m["proxy[random] n=100"] > m[k] for k in DFM) else "NOT SUPPORTED")
both = piv.loc[["proxy[random] n=400", "oracle"]].drop(columns=["MEAN", "MEDIAN"]).dropna(axis=1)
r2 = both.loc["proxy[random] n=400"].mean() / both.loc["oracle"].mean()  # matched targets only (German lacks n=400)
print("=== H2: proxy n=400 / oracle =", round(r2, 2), "->", "SUPPORTED" if r2 >= 0.75 else "NOT SUPPORTED")
d3 = {n: round(m[f"proxy[strat_y] n={n}"] - m[f"proxy[random] n={n}"], 1) for n in (25, 50, 100, 200, 400)}
print("=== H3: strat_y - random by n", d3, "->", "SUPPORTED" if all(v <= 3 for v in d3.values()) else "NOT SUPPORTED")
d4 = {n: round(m[f"proxy[random_mnar] n={n}"] - m[f"proxy[random] n={n}"], 1) for n in (100, 400)}
print("=== H4: mnar - random", d4, "->", "SUPPORTED" if all(v < 0 for v in d4.values()) else "NOT SUPPORTED")
s2 = out[(0.01, "T2")][1]
p2 = s2.pivot_table(index="arm", columns="t", values="eo_red").mean(axis=1)
d5 = {n: round(p2[f"proxy[random] n={n}"] - m[f"proxy[random] n={n}"], 1) for n in (25, 50, 100)}
print("=== H5: T2 - T1 at n<=100", d5, "->", "SUPPORTED (T2 not better)" if all(v <= 0 for v in d5.values()) else "NOT SUPPORTED")

# budget curve rows (for the chart): mean over targets per n and arm family, T1 and T2
curve = []
for (b, rule), (_, s) in out.items():
    pv = s.pivot_table(index="arm", columns="t", values="eo_red")
    for arm, row in pv.iterrows():
        curve.append(dict(budget=b, rule=rule, arm=arm, mean=row.mean(), median=row.median()))
pd.DataFrame(curve).to_csv(R + "final_curve.csv", index=False)
print("\n=== sensitivity: mean reduction by budget (T1)")
print(pd.DataFrame(curve).query("rule=='T1'").pivot_table(index="arm", columns="budget", values="mean").reindex(key).round(1).to_string())

# E5: estimating the unconstrained model's gap
e5 = e2[(e2.method == "proxy_soft") & (e2.mu == 0.3) & (e2.strategy == "random")].drop_duplicates(["t", "seed", "nlab"])
e5 = e5.assign(err_naive=(e5.est_naive - e5.true_eo).abs(), err_proxy=(e5.est_proxy - e5.true_eo).abs(),
               sign_naive=np.sign(e5.lab_tpr_s) == np.sign(e5.true_tpr_s))
print("\n=== E5: mean abs error of gap estimate (true gap in parentheses)")
e5 = e5.assign(bias_naive=e5.est_naive - e5.true_eo, bias_proxy=e5.est_proxy - e5.true_eo)
print(e5.groupby("nlab")[["err_naive", "err_proxy", "bias_naive", "bias_proxy"]].mean().round(3).to_string())
print(e5.groupby("t")[["true_eo", "err_naive", "err_proxy"]].mean().round(3).to_string())
e5.groupby(["t", "nlab"])[["true_eo", "err_naive", "err_proxy"]].mean().to_csv(R + "final_e5.csv")
