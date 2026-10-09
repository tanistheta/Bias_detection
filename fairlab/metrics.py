import numpy as np
from sklearn.metrics import roc_auc_score


def group_rates(yhat, y, a, w=None):
    """Signed rates for group 1 minus group 0. a may be soft (probabilities) -> weighted rates."""
    a = np.asarray(a, float)
    w = np.ones(len(y)) if w is None else w
    out = {}
    for name, sel in (("tpr", y == 1), ("fpr", y == 0), ("sel", np.ones(len(y), bool))):
        r = []
        for g in (a, 1 - a):
            ww = g * w * sel
            r.append((ww * yhat).sum() / ww.sum() if ww.sum() > 0 else np.nan)
        out[name] = r[0] - r[1]
    return out


def evaluate(p, y, a, thr):
    yhat = (p >= thr).astype(int)
    r = group_rates(yhat, y, a)
    acc = (yhat == y).mean()
    tg, fg = abs(r["tpr"]), abs(r["fpr"])
    a = np.asarray(a)
    tprs = [yhat[(a == g) & (y == 1)].mean() for g in (0, 1)]
    return dict(auc=roc_auc_score(y, p), acc=acc, bal_acc=0.5 * (yhat[y == 1].mean() + 1 - yhat[y == 0].mean()),
                eo_gap=max(tg, fg), tpr_gap=tg, fpr_gap=fg, dp_gap=abs(r["sel"]),
                tpr_signed=r["tpr"], fpr_signed=r["fpr"], worst_tpr=min(tprs))


def rate_threshold(p, base):
    """Threshold giving selection rate = base (training base rate)."""
    return np.quantile(p, 1 - base)
