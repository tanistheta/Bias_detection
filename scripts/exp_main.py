"""E2/E3/E5 main experiment: label budget x selection strategy x method, plus demographic-free baselines.

usage: python scripts/exp_main.py SEED_FROM SEED_TO OUT.csv ds:attr,ds:attr [--no-df]
Splits per seed: train 50% / labelling pool 20% (outcomes known, demographics revealed only for
selected rows) / test 30%, stratified by label x attribute.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from fairlab.data import load_capped
from fairlab.learn import FairLR
from fairlab.metrics import evaluate, rate_threshold, group_rates
from fairlab.proxy import select, fit_proxy
from fairlab import baselines as B

MUS = [0.3, 1, 3, 10, 30]
NS = [25, 50, 100, 200, 400]
STRATS = ["random", "strat_y"]  # frozen after pilot (seeds 100-105); see PREREGISTRATION.md
DF = [("jtt", B.jtt, "lam", [1.5, 2, 3, 5]), ("cvar", B.cvar_dro, "alpha", [0.5, 0.2, 0.1]),
      ("arl", B.arl, "adv_lr", [0.05, 0.2, 0.5]), ("cluster_rw", B.cluster_reweigh, "gamma", [0.25, 0.5, 1.0])]


def splits(n, seed, strat):
    idx = np.arange(n)
    tr, rest = train_test_split(idx, test_size=0.5, random_state=seed, stratify=strat)
    po, te = train_test_split(rest, test_size=0.6, random_state=seed, stratify=strat[rest])
    return tr, po, te


def main():
    s0, s1, out, targets = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], [t.split(":") for t in sys.argv[4].split(",")]
    run_df = "--no-df" not in sys.argv
    rows = []
    for ds, attr in targets:
        X, y, A = load_capped(ds)
        X = X.values
        a = A[attr].values
        cols = None
        for seed in range(s0, s1):
            tr, po, te = splits(len(y), seed, y * 2 + a)
            Xtr, ytr, Xpo, ypo, apo, Xte, yte, ate = X[tr], y[tr], X[po], y[po], a[po], X[te], y[te], a[te]
            base = ytr.mean()
            rng_master = np.random.default_rng(seed * 7919 + 17)

            def record(method, mu, p_po, p_te, extra=None, lab=None):
                thr = rate_threshold(p_te, base)
                r = evaluate(p_te, yte, ate, thr)
                r.update(dataset=ds, attr=attr, seed=seed, method=method, mu=mu, pool_auc=roc_auc_score(ypo, p_po))
                if lab is not None and len(np.unique(apo[lab])) == 2:
                    thp = rate_threshold(p_po, base)
                    yh = (p_po[lab] >= thp).astype(int)
                    gr = group_rates(yh, ypo[lab], apo[lab])
                    r["est_eo_lab"] = np.nanmax([abs(gr["tpr"]), abs(gr["fpr"])])
                if extra:
                    r.update(extra)
                rows.append(r)

            m0 = FairLR().fit(Xtr, ytr)
            p0po, p0te = m0.predict_proba(Xpo), m0.predict_proba(Xte)
            record("erm", 0, p0po, p0te)
            # true gap of ERM on test (for estimation study)
            true_eo = evaluate(p0te, yte, ate, rate_threshold(p0te, base))
            # oracle: true A on all training rows
            for mu in MUS:
                m = FairLR(mu).fit(Xtr, ytr, ate[:0] if False else a[tr].astype(float), init=m0)
                record("oracle", mu, m.predict_proba(Xpo), m.predict_proba(Xte))
            # proxies
            for strat in STRATS:
                for n in NS:
                    if n > len(po) // 2:
                        continue
                    rng = np.random.default_rng(rng_master.integers(1 << 31))
                    lab, prior = select(strat, n, Xpo, ypo, apo, rng)
                    px = fit_proxy(Xpo, apo, lab, prior)
                    q = px.predict(Xtr)
                    qpo = px.predict(Xpo)
                    ex = dict(nlab=n, strategy=strat, proxy_auc=roc_auc_score(a[tr], q), lab_minfrac=apo[lab].mean(), nlab_actual=len(lab))
                    # E5: estimating ERM's gap with n labels (naive on labelled rows vs proxy-weighted on full pool)
                    thp = rate_threshold(p0po, base)
                    yh0 = (p0po >= thp).astype(int)
                    gl = group_rates(yh0[lab], ypo[lab], apo[lab])
                    gq = group_rates(yh0, ypo, qpo)
                    ex.update(true_eo=true_eo["eo_gap"], true_tpr_s=true_eo["tpr_signed"], true_fpr_s=true_eo["fpr_signed"],
                              est_naive=np.nanmax([abs(gl["tpr"]), abs(gl["fpr"])]), est_proxy=max(abs(gq["tpr"]), abs(gq["fpr"])),
                              lab_tpr_s=gl["tpr"], lab_fpr_s=gl["fpr"], q_tpr_s=gq["tpr"], q_fpr_s=gq["fpr"])
                    variants = [("proxy_soft", q, None)]
                    if strat == "random":
                        # FairDSR-style: hard proxy groups, penalty only on the more confident half of rows
                        conf = np.abs(q - 0.5)
                        hard = (q >= 0.5).astype(float)
                        variants.append(("proxy_confident", hard, (conf >= np.median(conf)).astype(float)))
                        # FairRF-style: related features = top-5 features most correlated with A on labelled rows
                        Zl = Xpo[lab]
                        sd = Zl.std(0)
                        cor = np.nan_to_num(np.abs(np.array([np.corrcoef(Zl[:, j], apo[lab])[0, 1] if sd[j] > 0 else 0 for j in range(X.shape[1])])))
                        top = np.argsort(-cor)[:5]
                        G = np.column_stack([(Xtr[:, j] > np.median(Xtr[:, j])).astype(float) if len(np.unique(Xtr[:, j])) > 2 else (Xtr[:, j] >= Xtr[:, j].max()).astype(float) for j in top])
                        variants.append(("related_feats", G, None))
                    for name, G, mask in variants:
                        for mu in MUS:
                            m = FairLR(mu).fit(Xtr, ytr, G, mask=mask, init=m0)
                            record(name, mu, m.predict_proba(Xpo), m.predict_proba(Xte), ex, lab)
            # MNAR robustness: group 1 discloses 3x less often (random selection otherwise)
            for n in NS:
                if n > len(po) // 2:
                    continue
                rng = np.random.default_rng(rng_master.integers(1 << 31))
                lab, _ = select("random", n, Xpo, ypo, apo, rng, mnar_ratio=1 / 3)
                q = fit_proxy(Xpo, apo, lab).predict(Xtr)
                ex = dict(nlab=n, strategy="random_mnar", proxy_auc=roc_auc_score(a[tr], q), lab_minfrac=apo[lab].mean())
                for mu in MUS:
                    m = FairLR(mu).fit(Xtr, ytr, q, init=m0)
                    record("proxy_soft", mu, m.predict_proba(Xpo), m.predict_proba(Xte), ex, lab)
            # demographic-free baselines (strength index stored in mu)
            if run_df:
                XX = np.r_[Xpo, Xte]
                for name, fn, par, vals in DF:
                    for i, v in enumerate(vals):
                        p = fn("lr", Xtr, ytr, XX, seed, **{par: v})
                        record(name, i + 1, p[: len(po)], p[len(po):], dict(param=v))
            print(ds, attr, seed, len(rows), flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
