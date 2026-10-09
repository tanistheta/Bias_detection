"""E1: where does the hidden protected attribute rank in a demographic-free disparity audit, and does
the score-correlation identity explain it?

For every binary candidate group G (each observed column split at its median / category) and for the
hidden attribute A we record, on the test split of an unconstrained (ERM) logistic model:
  - thresholded EO gap (what an auditor ranks by),
  - mean-score gap within each label, and the identity value sd(s|y) * corr(s,G|y) / sqrt(pi(1-pi)),
  - the model's reliance on the column (|standardised coefficient|).
usage: python scripts/exp_audit.py SEED_FROM SEED_TO OUT.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from fairlab.data import load_capped, ALL_PAIRS
from fairlab.learn import FairLR
from fairlab.metrics import evaluate, rate_threshold
from scripts.exp_main import splits


def groups_of(Xtr, Xte, cols, min_frac=0.1):
    out = []
    for j, c in enumerate(cols):
        u = np.unique(Xtr[:, j])
        if len(u) <= 2:
            g_tr, g_te = (Xtr[:, j] >= u.max()).astype(int), (Xte[:, j] >= u.max()).astype(int)
        else:
            med = np.median(Xtr[:, j])
            g_tr, g_te = (Xtr[:, j] > med).astype(int), (Xte[:, j] > med).astype(int)
        if min_frac <= g_tr.mean() <= 1 - min_frac:
            out.append((c, j, g_te))
    return out


def score_stats(s, y, g):
    res = {}
    for yv in (0, 1):
        m = y == yv
        gs, ss = g[m], s[m]
        pi = gs.mean()
        gap = ss[gs == 1].mean() - ss[gs == 0].mean()
        ident = ss.std() * np.corrcoef(ss, gs)[0, 1] / np.sqrt(pi * (1 - pi)) if 0 < pi < 1 else np.nan
        res[yv] = (gap, ident, np.corrcoef(ss, gs)[0, 1] if 0 < pi < 1 else np.nan)
    return res


def main():
    s0, s1, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    rows = []
    pairs = [t.split(":") for t in sys.argv[4].split(",")] if len(sys.argv) > 4 else ALL_PAIRS
    for ds, attr in pairs:
        X, y, A = load_capped(ds)
        cols = list(X.columns)
        X = X.values
        a = A[attr].values
        for seed in range(s0, s1):
            tr, po, te = splits(len(y), seed, y * 2 + a)
            m = FairLR().fit(X[tr], y[tr])
            p = m.predict_proba(X[te])
            s = np.log(p / (1 - p))  # score on the logit scale
            thr = rate_threshold(p, y[tr].mean())
            coef = np.abs(m.th[:-1])
            cand = [(c, j, g) for c, j, g in groups_of(X[tr], X[te], cols)] + [("__A__", -1, a[te])]
            for c, j, g in cand:
                ev = evaluate(p, y[te], g, thr)
                st = score_stats(s, y[te], g)
                rows.append(dict(dataset=ds, attr=attr, seed=seed, group=c, is_A=c == "__A__", eo_gap=ev["eo_gap"],
                                 gap_y0=st[0][0], ident_y0=st[0][1], rho_y0=st[0][2], gap_y1=st[1][0], ident_y1=st[1][1], rho_y1=st[1][2],
                                 reliance=coef[j] if j >= 0 else np.nan, frac=g.mean()))
        print(ds, attr, flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
