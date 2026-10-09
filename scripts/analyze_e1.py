"""E1 summary (paper Table I): rank of the hidden attribute in the demographic-free audit, per pair.

usage: python scripts/analyze_e1.py [results/e1_audit.csv] [results/e1_summary.csv]
Columns: eo_A = mean EO gap of the hidden group; rank_A = median rank (1 = largest EO gap); n = median number of
candidate groups including A; top = share of audits where A ranked first; rhoA = mean over seeds of
max_y |corr(score, A | y)|; rho_feat = mean over seeds of the largest |corr(score, G | y)| over feature groups;
sp1 = mean Spearman correlation between the ranking by max_y |mean score gap| and the ranking by EO gap;
sp2 = mean Spearman correlation between model reliance and EO gap over feature groups.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
src = sys.argv[1] if len(sys.argv) > 1 else "results/e1_audit.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "results/e1_summary.csv"
d = pd.read_csv(src)
rows, ident = [], 0.0
for (ds, attr), g in d.groupby(["dataset", "attr"], sort=True):
    per = []
    for _, s in g.groupby("seed"):
        rank = s.eo_gap.rank(ascending=False, method="min")
        a, f = s[s.is_A].iloc[0], s[~s.is_A]
        gap = np.maximum(s.gap_y0.abs(), s.gap_y1.abs())
        ident = max(ident, np.nanmax(np.abs(np.r_[s.gap_y0 - s.ident_y0, s.gap_y1 - s.ident_y1])))
        per.append(dict(eo_A=a.eo_gap, rank_A=rank[s.is_A].iloc[0], n=len(s), top=float(rank[s.is_A].iloc[0] == 1),
                        rhoA=np.nanmax([abs(a.rho_y0), abs(a.rho_y1)]),
                        rho_feat=np.nanmax(np.r_[f.rho_y0.abs(), f.rho_y1.abs()]),
                        sp1=spearmanr(gap, s.eo_gap).statistic, sp2=spearmanr(f.reliance, f.eo_gap).statistic))
    p = pd.DataFrame(per)
    rows.append(dict(dataset=ds, attr=attr, eo_A=p.eo_A.mean(), rank_A=p.rank_A.median(), n=int(p.n.median()),
                     top=p.top.mean(), rhoA=p.rhoA.mean(), rho_feat=p.rho_feat.mean(), sp1=p.sp1.mean(),
                     sp2=p.sp2.mean(), audits=len(p)))
t = pd.DataFrame(rows)
t.drop(columns="audits").round(3).to_csv(out, index=False)
print(t.round(3).to_string(index=False))
print(f"audits {t.audits.sum()}, A ranked first {int((t.top * t.audits).round().sum())}, identity max error {ident:.1e}")
