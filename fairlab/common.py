import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def split(n, seed, strat):
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=0.3, random_state=seed, stratify=strat)
    tr, va = train_test_split(tr, test_size=0.2 / 0.7, random_state=seed, stratify=strat[tr])
    return tr, va, te


def make_model(kind, seed=0):
    if kind == "lr":
        return LogisticRegression(C=1.0, max_iter=2000)
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
    raise ValueError(kind)


def rates(yhat, y, a):
    out = {}
    for g in (0, 1):
        m = a == g
        pos, neg = m & (y == 1), m & (y == 0)
        out[g] = dict(sel=yhat[m].mean(), tpr=yhat[pos].mean() if pos.any() else np.nan,
                      fpr=yhat[neg].mean() if neg.any() else np.nan, acc=(yhat[m] == y[m]).mean(), n=m.sum())
    return out


def fairness(yhat, y, a, p=None):
    r = rates(yhat, y, a)
    tpr_gap = abs(r[0]["tpr"] - r[1]["tpr"])
    fpr_gap = abs(r[0]["fpr"] - r[1]["fpr"])
    res = dict(acc=(yhat == y).mean(),
               bal_acc=0.5 * (yhat[y == 1].mean() + (1 - yhat[y == 0]).mean()),
               dp_gap=abs(r[0]["sel"] - r[1]["sel"]),
               eo_gap=max(tpr_gap, fpr_gap), tpr_gap=tpr_gap, fpr_gap=fpr_gap,
               avg_odds=0.5 * (tpr_gap + fpr_gap),
               worst_acc=min(r[0]["acc"], r[1]["acc"]),
               worst_tpr=min(r[0]["tpr"], r[1]["tpr"]))
    if p is not None:
        res["auc"] = roc_auc_score(y, p)
    return res


class Scaled:
    """Standardise numeric matrix then fit model with optional sample weights."""

    def __init__(self, model):
        self.model = model
        self.sc = StandardScaler()

    def fit(self, X, y, w=None):
        Xs = self.sc.fit_transform(X)
        self.model.fit(Xs, y, sample_weight=w)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self.sc.transform(X))[:, 1]
