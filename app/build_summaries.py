"""Condense the raw 30-seed results into the small tables the web app shows (app/data/*.csv).

Run from the repository root:  python app/build_summaries.py
Uses fairlab.analysis.choose and the same filters as scripts/analyze_final.py and scripts/round2.py.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fairlab.analysis import choose  # noqa: E402

R = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "app", "data")
CLASSIC = ["e2_a", "e2_b", "e2_c", "e2_d"]
CENSUS1 = ["acs_a", "acs_b"]
CENSUS2 = ["acs2_a", "acs2_b"]
KEYS = ["dataset", "attr", "seed", "method", "strategy", "nlab", "mu"]


def load(names):
    d = pd.concat([pd.read_csv(os.path.join(R, f + ".csv")) for f in names]).drop_duplicates(KEYS)
    d["t"] = d.dataset.str.replace("_2018", "") + "-" + d.attr
    return d


def target_name(t):
    return t.replace("acs_", "").replace("_ACSIncome", " income").replace("_ACSEmployment", " employment")


def audit():
    rows = []
    for fam, f in (("Classic datasets", "e1_audit.csv"), ("US census (round 1)", "acs_audit.csv")):
        d = pd.read_csv(os.path.join(R, f))
        d["rank"] = d.groupby(["dataset", "attr", "seed"]).eo_gap.rank(ascending=False, method="min")
        a = d[d.is_A].copy()
        n = d.groupby(["dataset", "attr", "seed"]).size().rename("candidates").reset_index()
        a = a.merge(n, on=["dataset", "attr", "seed"])
        for (ds, attr), g in a.groupby(["dataset", "attr"]):
            rows.append(dict(family=fam, target=target_name(f"{ds.replace('_2018', '')}-{attr}"), runs=len(g),
                             hidden_first=int((g["rank"] == 1).sum()), median_rank=g["rank"].median(),
                             candidates=int(g.candidates.median()), hidden_gap=g.eo_gap.mean()))
    return pd.DataFrame(rows)


def e5(d, fam):
    e = d[(d.method == "proxy_soft") & (d.mu == 0.3) & (d.strategy == "random")].drop_duplicates(["t", "seed", "nlab"])
    e = e.assign(err_labelled=(e.est_naive - e.true_eo).abs(), err_proxy=(e.est_proxy - e.true_eo).abs(),
                 bias_labelled=e.est_naive - e.true_eo, bias_proxy=e.est_proxy - e.true_eo)
    out = e.groupby("nlab")[["err_labelled", "err_proxy", "bias_labelled", "bias_proxy"]].mean().reset_index()
    out.insert(0, "family", fam)
    return out


def guard(d, fam, thr=0.08):
    """round2.py: correct only if the proxy-estimated EO gap of the unconstrained model is >= thr."""
    p = d[(d.method == "proxy_soft") & (d.strategy == "random")].copy()
    s = choose(d, 0.01, "T1")
    s = s[(s.method == "proxy_soft") & (s.strategy == "random")]
    p["est_proxy_gap"] = np.maximum(p.q_tpr_s.abs(), p.q_fpr_s.abs())
    s = s.merge(p.drop_duplicates(["dataset", "attr", "seed", "nlab"])[["dataset", "attr", "seed", "nlab", "est_proxy_gap"]],
                on=["dataset", "attr", "seed", "nlab"])
    s["eo_g"] = np.where(s.est_proxy_gap >= thr, s.eo_gap, s.eo_gap0)
    rows = []
    for n in (100, 400):
        x = s[s.nlab == n]
        rows.append(dict(family=fam, nlab=n, harm=(x.eo_gap > x.eo_gap0 + 0.01).mean(),
                         harm_guarded=(x.eo_g > x.eo_gap0 + 0.01).mean(), applied=(x.est_proxy_gap >= thr).mean(),
                         red=100 * (1 - x.eo_gap.mean() / x.eo_gap0.mean()),
                         red_guarded=100 * (1 - x.eo_g.mean() / x.eo_gap0.mean())))
    return pd.DataFrame(rows)


def fix_tables():
    keep = ["t", "arm", "eo_red", "ci_lo", "ci_hi", "p", "dauc", "eo0", "harm_rate", "n_seeds"]
    a = pd.read_csv(os.path.join(R, "final_summary_T1_0.01.csv"))[keep].assign(family="Classic datasets")
    b = pd.read_csv(os.path.join(R, "final_summary_acs_T1_0.01.csv"))[keep].assign(family="US census (round 1)")
    out = pd.concat([a, b])
    out["target"] = out.t.map(target_name)
    return out


def hypotheses(fix):
    c = fix[fix.family == "Classic datasets"].pivot_table(index="arm", columns="t", values="eo_red")
    m = c.mean(axis=1)
    t2 = pd.read_csv(os.path.join(R, "final_summary_T2_0.01.csv")).pivot_table(index="arm", columns="t", values="eo_red").mean(axis=1)
    both = c.loc[["proxy[random] n=400", "oracle"]].dropna(axis=1)
    df = {k: m[k] for k in ("arl", "cvar", "jtt", "cluster_rw")}
    h3 = max(m[f"proxy[strat_y] n={n}"] - m[f"proxy[random] n={n}"] for n in (25, 50, 100, 200, 400))
    h4 = {n: m[f"proxy[random_mnar] n={n}"] - m[f"proxy[random] n={n}"] for n in (100, 400)}
    h5 = max(t2[f"proxy[random] n={n}"] - m[f"proxy[random] n={n}"] for n in (25, 50, 100))
    ratio = both.loc["proxy[random] n=400"].mean() / both.loc["oracle"].mean()
    return pd.DataFrame([
        ("H1", "100 labelled people beat every demographic-free method",
         f"{m['proxy[random] n=100']:.1f}% vs at most {max(df.values()):.1f}%", all(m["proxy[random] n=100"] > v for v in df.values())),
        ("H2", "400 people reach 75% of the full-knowledge (oracle) reduction", f"{ratio:.0%} of the oracle", ratio >= 0.75),
        ("H3", "Choosing rows by outcome is no better than random (<= 3 points)", f"largest advantage {h3:+.1f} points", h3 <= 3),
        ("H4", "When the group discloses 3x less often, results get worse",
         f"{h4[100]:+.1f} points at n=100, {h4[400]:+.1f} at n=400", all(v < 0 for v in h4.values())),
        ("H5", "Tuning on the labelled rows does not beat the AUC-only rule", f"largest gain {h5:+.1f} points", h5 <= 0),
        ("H6", "A direction check (cosine) predicts harmful fixes", "AUROC 0.58 at n=100 (needed > 0.70)", False),
        ("H7", "Results hold for a neural network", "7.9% vs ARL 7.8% with only 2-5 seeds per target: inconclusive", None),
    ], columns=["id", "prediction", "result", "supported"])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    classic, c1, c2 = load(CLASSIC), load(CENSUS1), load(CENSUS2)
    fix = fix_tables()
    tables = {
        "audit": audit(),
        "e5": pd.concat([e5(classic, "Classic datasets"), e5(c1, "US census (round 1)"), e5(c2, "US census (round 2)")]),
        "guard": pd.concat([guard(c1, "US census (round 1)"), guard(c2, "US census (round 2)")]),
        "fix": fix,
        "hypotheses": hypotheses(fix),
    }
    for name, t in tables.items():
        t.to_csv(os.path.join(OUT, name + ".csv"), index=False, float_format="%.6g")
        print(f"{name}: {len(t)} rows")
    a = tables["audit"]
    print("audit runs", a.runs.sum(), "hidden first", a.hidden_first.sum())
    print(tables["guard"].round(3).to_string())
    print(tables["e5"].round(3).to_string())
    print(tables["hypotheses"].to_string())
