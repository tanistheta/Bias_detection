"""E6: does the label-budget result hold for a neural model? MLP learner; ERM, ARL (faithful-style),
JTT, proxy (random n labels) and oracle with the same soft-EO penalty.
usage: python scripts/exp_mlp.py SEED_FROM SEED_TO OUT.csv ds:attr,...
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from fairlab.data import load_capped
from fairlab.learn import FairMLP, ARLMLP
from fairlab.metrics import evaluate, rate_threshold
from fairlab.proxy import select, fit_proxy
from scripts.exp_main import splits

MUS = [1, 3, 10]
s0, s1, out, targets = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], [t.split(":") for t in sys.argv[4].split(",")]
rows = []
for ds, attr in targets:
    X, y, A = load_capped(ds); X = X.values; a = A[attr].values
    for seed in range(s0, s1):
        tr, po, te = splits(len(y), seed, y * 2 + a)
        base = y[tr].mean()
        def rec(method, mu, m, nlab=0):
            ppo, pte = m.predict_proba(X[po]), m.predict_proba(X[te])
            r = evaluate(pte, y[te], a[te], rate_threshold(pte, base))
            r.update(dataset=ds, attr=attr, seed=seed, method=method, mu=mu, nlab=nlab, pool_auc=roc_auc_score(y[po], ppo)); rows.append(r)
        m0 = FairMLP(seed=seed).fit(X[tr], y[tr]); rec("erm", 0, m0)
        for i, al in enumerate([0.005, 0.02, 0.1]):
            rec("arl", i + 1, ARLMLP(adv_lr=al, seed=seed).fit(X[tr], y[tr]))
        p1 = m0.predict_proba(X[tr]); err = ((p1 >= 0.5).astype(int) != y[tr])
        for i, lam in enumerate([1.5, 2, 3]):
            rec("jtt", i + 1, FairMLP(seed=seed).fit(X[tr], y[tr], sample_weight=np.where(err, lam, 1.0)))
        for mu in MUS:
            rec("oracle", mu, FairMLP(mu, seed=seed).fit(X[tr], y[tr], a[tr].astype(float), init=m0))
        for n in (100, 400):
            if n > len(po) // 2: continue
            lab, _ = select("random", n, X[po], y[po], a[po], np.random.default_rng(seed * 31 + n))
            q = fit_proxy(X[po], a[po], lab).predict(X[tr])
            for mu in MUS:
                rec("proxy_soft", mu, FairMLP(mu, seed=seed).fit(X[tr], y[tr], q, init=m0), n)
        print(ds, attr, seed, flush=True); pd.DataFrame(rows).to_csv(out, index=False)
