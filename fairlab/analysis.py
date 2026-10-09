"""Model selection (tuning) rules and aggregation, shared by the pilot and confirmatory analyses.

T1 (demographic-free): strongest setting whose pool AUC stays within `budget` of ERM's pool AUC.
T2 (label-informed):   among settings within the same AUC budget, the one with the smallest EO gap
                       measured on the labelled pool rows (true A, available only there).
Falling outside the budget -> fall back to ERM (the practitioner would not deploy it).
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

KEY = ["dataset", "attr", "seed"]
METRICS = ["eo_gap", "tpr_gap", "fpr_gap", "dp_gap", "auc", "acc", "worst_tpr"]


def choose(d, budget=0.01, rule="T1"):
    d = d.copy()
    for c in ("strategy", "nlab"):
        if c not in d:
            d[c] = np.nan
    d["strategy"] = d["strategy"].fillna("-")
    d["nlab"] = d["nlab"].fillna(0)
    erm = d[d.method == "erm"].set_index(KEY)
    out = []
    gcols = KEY + ["method", "strategy", "nlab"]
    for k, g in d[d.method != "erm"].groupby(gcols):
        e = erm.loc[tuple(k[:3])]
        ok = g[g.pool_auc >= e.pool_auc - budget]
        if rule == "T2" and "est_eo_lab" in g and ok.est_eo_lab.notna().any():
            pick = ok.loc[ok.est_eo_lab.idxmin()]
        elif len(ok):
            pick = ok.loc[ok.mu.idxmax()]
        else:
            pick = e
        r = {c: pick[c] for c in METRICS}
        r.update(dict(zip(gcols, k)), mu=pick["mu"] if len(ok) else 0)
        for c in METRICS:
            r[c + "0"] = e[c]
        out.append(r)
    return pd.DataFrame(out)


def summarise(sel, by=("method", "strategy", "nlab")):
    rows = []
    for k, g in sel.groupby(["dataset", "attr"] + list(by)):
        red = 100 * (1 - g.eo_gap.mean() / g.eo_gap0.mean())
        diff = g.eo_gap - g.eo_gap0
        try:
            p = wilcoxon(g.eo_gap, g.eo_gap0, alternative="less").pvalue if (diff != 0).any() else 1.0
        except ValueError:
            p = np.nan
        # bootstrap CI over seeds for the % reduction
        rng = np.random.default_rng(0)
        bs = []
        a, b = g.eo_gap.values, g.eo_gap0.values
        for _ in range(1000):
            i = rng.integers(0, len(a), len(a))
            bs.append(100 * (1 - a[i].mean() / b[i].mean()))
        rows.append(dict(zip(["dataset", "attr"] + list(by), k), eo_red=red, ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                         p=p, dauc=100 * (g.auc - g.auc0).mean(), dacc=100 * (g.acc - g.acc0).mean(),
                         tpr_red=100 * (1 - g.tpr_gap.mean() / g.tpr_gap0.mean()), dp_red=100 * (1 - g.dp_gap.mean() / g.dp_gap0.mean()),
                         eo0=g.eo_gap0.mean(), n_seeds=len(g), harm_rate=(g.eo_gap > g.eo_gap0 + 0.01).mean()))
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = np.sum(~np.isnan(p))
    adj = np.full(len(p), np.nan)
    run = 0
    for r, i in enumerate(o):
        if np.isnan(p[i]):
            continue
        run = max(run, min(1, (m - r) * p[i]))
        adj[i] = run
    return adj
