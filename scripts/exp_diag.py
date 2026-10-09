"""E4: when does proxy-based correction backfire? Runs ERM, oracle, and random-n proxy correction on all
17 dataset-attribute pairs and stores pre-deployment diagnostics computable from the labelled rows only:
  - signed (TPR, FPR) gap of ERM on the true groups of the labelled rows,
  - signed (TPR, FPR) gap of ERM on the soft proxy groups over the whole pool,
  - their cosine ('agreement'): if correcting proxy groups pushes the gaps in a different direction from
    the true-group gaps, correction can reverse or worsen the true gap.
usage: python scripts/exp_diag.py SEED_FROM SEED_TO OUT.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from fairlab.data import load_capped, ALL_PAIRS
from fairlab.learn import FairLR
from fairlab.metrics import evaluate, rate_threshold, group_rates
from fairlab.proxy import select, fit_proxy
from scripts.exp_main import splits, MUS

s0, s1, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
pairs = ALL_PAIRS if len(sys.argv) < 5 else [t.split(":") for t in sys.argv[4].split(",")]
rows = []
for ds, attr in pairs:
    X, y, A = load_capped(ds); X = X.values; a = A[attr].values
    for seed in range(s0, s1):
        tr, po, te = splits(len(y), seed, y * 2 + a); base = y[tr].mean()
        def rec(method, mu, m, ex=None):
            ppo, pte = m.predict_proba(X[po]), m.predict_proba(X[te])
            r = evaluate(pte, y[te], a[te], rate_threshold(pte, base))
            r.update(dataset=ds, attr=attr, seed=seed, method=method, mu=mu, pool_auc=roc_auc_score(y[po], ppo))
            if ex: r.update(ex)
            rows.append(r)
        m0 = FairLR().fit(X[tr], y[tr]); rec("erm", 0, m0)
        if "--oracle" in sys.argv:
            for mu in MUS:
                rec("oracle", mu, FairLR(mu).fit(X[tr], y[tr], a[tr].astype(float), init=m0))
        p0 = m0.predict_proba(X[po]); yh = (p0 >= rate_threshold(p0, base)).astype(int)
        for n in (100, 400):
            if n > len(po) // 2: continue
            lab, _ = select("random", n, X[po], y[po], a[po], np.random.default_rng(seed * 101 + n))
            px = fit_proxy(X[po], a[po], lab)
            q, qpo = px.predict(X[tr]), px.predict(X[po])
            gl = group_rates(yh[lab], y[po][lab], a[po][lab]); gq = group_rates(yh, y[po], qpo)
            vl, vq = np.array([gl["tpr"], gl["fpr"]]), np.array([gq["tpr"], gq["fpr"]])
            cos = float(vl @ vq / (np.linalg.norm(vl) * np.linalg.norm(vq) + 1e-12))
            ex = dict(nlab=n, proxy_auc=roc_auc_score(a[tr], q), lab_tpr_s=gl["tpr"], lab_fpr_s=gl["fpr"], q_tpr_s=gq["tpr"], q_fpr_s=gq["fpr"],
                      agree_cos=cos, est_eo_lab=float(np.nanmax(np.abs(vl))))
            for mu in MUS:
                rec("proxy_soft", mu, FairLR(mu).fit(X[tr], y[tr], q, init=m0), ex)
        print(ds, attr, seed, flush=True); pd.DataFrame(rows).to_csv(out, index=False)
