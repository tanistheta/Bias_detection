"""Single-seed versions of the paper's experiments for the web app.

Everything is built from fairlab and from the code paths of scripts/exp_main.py and scripts/exp_audit.py
(same splits, same random draws, same tuning rule), so a run in the app for seed s reproduces the rows that
the 30-seed study stored in results/ for that seed. Nothing here depends on Streamlit.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fairlab import baselines as B  # noqa: E402
from fairlab.analysis import choose  # noqa: E402
from fairlab.data import ALL_PAIRS, TARGETS, load_capped  # noqa: E402
from fairlab.learn import FairLR  # noqa: E402
from fairlab.metrics import evaluate, group_rates, rate_threshold  # noqa: E402
from fairlab.proxy import fit_proxy, select  # noqa: E402
from scripts.exp_audit import groups_of, score_stats  # noqa: E402
from scripts.exp_main import DF, MUS, NS, splits  # noqa: E402

warnings.filterwarnings("ignore")

GUARD = 0.08  # gap guard: correct only if the proxy-estimated gap is at least this large
BUDGET = 0.01  # T1 tuning: strongest penalty whose pool AUC stays within this of the unconstrained model
HARM = 0.01  # a correction "harms" if the test EO gap rises by more than this

LABELS = {
    "adult": "Adult income", "compas": "COMPAS recidivism", "law": "Law School bar exam",
    "credit": "Taiwan credit default", "oulad": "OULAD online course", "dutch": "Dutch census",
    "german": "German credit", "bank": "Bank marketing", "diabetes": "Diabetes readmission",
}
GROUPS = {
    ("adult", "sex"): "women", ("adult", "race"): "non-white people", ("compas", "race"): "African-American people",
    ("compas", "sex"): "women", ("law", "race"): "non-white students", ("law", "sex"): "women",
    ("credit", "age"): "people under 25", ("credit", "sex"): "women", ("oulad", "disability"): "disabled students",
    ("oulad", "sex"): "women", ("dutch", "sex"): "women", ("german", "age"): "people under 25",
    ("german", "sex"): "women", ("bank", "age"): "people under 30", ("bank", "marital"): "single people",
    ("diabetes", "race"): "African-American patients", ("diabetes", "sex"): "women",
}


def pair_label(ds, attr):
    return f"{LABELS[ds]} · hidden {attr} ({GROUPS[(ds, attr)]})"


class Run:
    """One seed of one dataset-attribute pair: the paper's 50/20/30 split and the unconstrained model."""

    def __init__(self, ds, attr, seed):
        X, y, A = load_capped(ds)
        self.ds, self.attr, self.seed = ds, attr, int(seed)
        self.cols = list(X.columns)
        X = X.values
        a = A[attr].values
        tr, po, te = splits(len(y), self.seed, y * 2 + a)
        self.Xtr, self.ytr, self.atr = X[tr], y[tr], a[tr]
        self.Xpo, self.ypo, self.apo = X[po], y[po], a[po]
        self.Xte, self.yte, self.ate = X[te], y[te], a[te]
        self.base = self.ytr.mean()
        self.m0 = FairLR().fit(self.Xtr, self.ytr)
        self.p0po, self.p0te = self.m0.predict_proba(self.Xpo), self.m0.predict_proba(self.Xte)
        self.n_rows, self.n_features, self.group_share = len(y), X.shape[1], a.mean()

    def budgets(self):
        return [n for n in NS if n <= len(self.ypo) // 2]


def label_rng(seed, n, pool_size, draw=0):
    """Random generator for choosing which pool rows reveal the group.

    draw 0 reproduces the paper: exp_main draws one generator per budget from a per-seed master, random
    strategy first, skipping budgets larger than half the pool. Other draws are fresh, reproducible ones."""
    if draw:
        return np.random.default_rng([int(seed), int(n), int(draw)])
    master = np.random.default_rng(int(seed) * 7919 + 17)
    for nn in NS:
        if nn > pool_size // 2:
            continue
        r = np.random.default_rng(master.integers(1 << 31))
        if nn == n:
            return r
    raise ValueError(f"budget {n} is larger than half the labelling pool")


# ------------------------------------------------------------------ FIND
def audit(run):
    """E1 for one seed: every candidate group (observed columns split at the median) plus the hidden group,
    ranked by the unconstrained model's equalized-odds gap on the test split."""
    m = run.m0
    p = run.p0te
    s = np.log(p / (1 - p))
    thr = rate_threshold(p, run.base)
    coef = np.abs(m.th[:-1])
    cand = groups_of(run.Xtr, run.Xte, run.cols) + [("__A__", -1, run.ate)]
    rows = []
    for c, j, g in cand:
        ev = evaluate(p, run.yte, g, thr)
        st = score_stats(s, run.yte, g)
        # groups_of splits a 0/1 column at its value 1 and any other column at the training median
        label = "" if j < 0 else (f"{c} = 1" if len(np.unique(run.Xtr[:, j])) <= 2 else f"{c} above median")
        rows.append(dict(group=c, label=label, is_A=c == "__A__", eo_gap=ev["eo_gap"], gap_y0=st[0][0], ident_y0=st[0][1],
                         rho_y0=st[0][2], gap_y1=st[1][0], ident_y1=st[1][1], rho_y1=st[1][2],
                         reliance=coef[j] if j >= 0 else np.nan, frac=g.mean()))
    out = pd.DataFrame(rows).sort_values("eo_gap", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


# ------------------------------------------------------------------ MEASURE
def measure(run, n, draw=0):
    """E5 for one seed: the unconstrained model's EO gap on test (answer key) against two estimates made
    with n revealed labels: labelled rows only (naive) and the whole pool weighted by the proxy."""
    rng = label_rng(run.seed, n, len(run.ypo), draw)
    lab, prior = select("random", n, run.Xpo, run.ypo, run.apo, rng)
    px = fit_proxy(run.Xpo, run.apo, lab, prior)
    q, qpo = px.predict(run.Xtr), px.predict(run.Xpo)
    yh0 = (run.p0po >= rate_threshold(run.p0po, run.base)).astype(int)
    gl = group_rates(yh0[lab], run.ypo[lab], run.apo[lab])
    gq = group_rates(yh0, run.ypo, qpo)
    true = evaluate(run.p0te, run.yte, run.ate, rate_threshold(run.p0te, run.base))
    return dict(true_eo=true["eo_gap"], est_naive=np.nanmax([abs(gl["tpr"]), abs(gl["fpr"])]),
                est_proxy=max(abs(gq["tpr"]), abs(gq["fpr"])), proxy_auc=roc_auc_score(run.atr, q),
                n_labelled=len(lab), n_group_labelled=int(run.apo[lab].sum()), lab=lab, q=q)


def measure_draws(run, n, k=30):
    """The same estimate for k different random choices of who reveals their group (draw 0 = the paper's)."""
    rows = []
    for d in range(k):
        m = measure(run, n, d)
        rows.append(dict(draw=d, true_eo=m["true_eo"], labelled_only=m["est_naive"], proxy=m["est_proxy"]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ FIX
def fix(run, n, draw=0, guard=GUARD, budget=BUDGET, with_baseline=True):
    """E2 for one seed: unconstrained model, proxy correction with n labels, the best demographic-free method
    (cluster reweighting) and the oracle, each tuned by the paper's T1 rule; then the gap guard."""
    rows = []

    def record(method, mu, p_po, p_te, **extra):
        r = evaluate(p_te, run.yte, run.ate, rate_threshold(p_te, run.base))
        r.update(dataset=run.ds, attr=run.attr, seed=run.seed, method=method, mu=mu,
                 pool_auc=roc_auc_score(run.ypo, p_po), **extra)
        rows.append(r)

    record("erm", 0, run.p0po, run.p0te)
    for mu in MUS:
        m = FairLR(mu).fit(run.Xtr, run.ytr, run.atr.astype(float), init=run.m0)
        record("oracle", mu, m.predict_proba(run.Xpo), m.predict_proba(run.Xte))
    meas = measure(run, n, draw)
    for mu in MUS:
        m = FairLR(mu).fit(run.Xtr, run.ytr, meas["q"], init=run.m0)
        record("proxy_soft", mu, m.predict_proba(run.Xpo), m.predict_proba(run.Xte), strategy="random", nlab=n)
    if with_baseline:
        name, fn, par, vals = next(d for d in DF if d[0] == "cluster_rw")
        XX = np.r_[run.Xpo, run.Xte]
        for i, v in enumerate(vals):
            p = fn("lr", run.Xtr, run.ytr, XX, run.seed, **{par: v})
            record(name, i + 1, p[: len(run.ypo)], p[len(run.ypo):])
    raw = pd.DataFrame(rows)
    sel = choose(raw, budget, "T1").set_index("method")
    erm = raw[raw.method == "erm"].iloc[0]
    applied = meas["est_proxy"] >= guard
    proxy = sel.loc["proxy_soft"]
    deployed = proxy if applied else None
    return dict(raw=raw, sel=sel, erm=erm, measure=meas, applied=applied, guard=guard,
                proxy=proxy, deployed=deployed, harm=bool(proxy.eo_gap > proxy.eo_gap0 + HARM))
