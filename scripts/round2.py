import sys, os, glob
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from fairlab.analysis import choose
PAT = sys.argv[1] if len(sys.argv) > 1 else "acs2_*.csv"
files = sorted(glob.glob(os.path.join("results", PAT)))
print("files:", files)
d = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates(["dataset", "attr", "seed", "method", "strategy", "nlab", "mu"])
d["t"] = d.dataset.str.replace("_2018", "") + "-" + d.attr
print(d.groupby("t").seed.nunique().to_string())
p = d[(d.method == "proxy_soft") & (d.strategy == "random")].copy()
e = p[(p.nlab == 100) & (p.mu == 0.3)].drop_duplicates(["t", "seed"])
r1 = e.groupby("t").apply(lambda g: pd.Series(dict(true_gap=g.true_eo.mean(), mae_labelled=(g.est_naive - g.true_eo).abs().mean(), mae_proxy=(g.est_proxy - g.true_eo).abs().mean())))
r1["proxy_better"] = r1.mae_proxy < r1.mae_labelled
print("\n=== R1 (n=100 gap estimation)"); print(r1.round(3).to_string())
k = int(r1.proxy_better.sum()); print(f"R1: proxy better on {k} of {len(r1)} -> {'SUPPORTED' if k >= 4 else 'NOT SUPPORTED'}")
s = choose(d, 0.01, "T1"); s = s[(s.method == "proxy_soft") & (s.strategy == "random")]
s["t"] = s.dataset.str.replace("_2018", "") + "-" + s.attr
p["est_proxy_gap"] = np.maximum(p.q_tpr_s.abs(), p.q_fpr_s.abs())
s = s.merge(p.drop_duplicates(["dataset", "attr", "seed", "nlab"])[["dataset", "attr", "seed", "nlab", "est_proxy_gap"]], on=["dataset", "attr", "seed", "nlab"])
s["eo_g"] = np.where(s.est_proxy_gap >= 0.08, s.eo_gap, s.eo_gap0)
s["harm"] = s.eo_gap > s.eo_gap0 + 0.01; s["harm_g"] = s.eo_g > s.eo_gap0 + 0.01
ok2 = True
for n in (100, 400):
    x = s[s.nlab == n]
    h, hg = x.harm.mean(), x.harm_g.mean()
    ok2 &= hg < h
    print(f"\n=== R2 n={n}: harm {h:.3f} -> guarded {hg:.3f}; applied {np.mean(x.est_proxy_gap >= 0.08):.2f}")
    print(x.groupby("t").apply(lambda q: pd.Series(dict(base_gap=q.eo_gap0.mean(), red=100 * (1 - q.eo_gap.mean() / q.eo_gap0.mean()), red_guarded=100 * (1 - q.eo_g.mean() / q.eo_gap0.mean()), harm=q.harm.mean(), harm_guarded=q.harm_g.mean()))).round(3).to_string())
print("R2:", "SUPPORTED" if ok2 else "NOT SUPPORTED")
x = s[s.nlab == 400]
per = x.groupby("t").apply(lambda q: pd.Series(dict(base=q.eo_gap0.mean(), red_g=100 * (1 - q.eo_g.mean() / q.eo_gap0.mean()))))
big = per[per.base >= 0.05]
m = big.red_g.mean() if len(big) else np.nan
print(f"\n=== R3: guarded 400-label mean reduction on {len(big)} targets with gap >= 0.05: {m:.1f}% -> {'SUPPORTED' if m > 0 else 'NOT SUPPORTED'}")
