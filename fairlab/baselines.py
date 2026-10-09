"""Demographic-free baselines (never see A). Each takes (kind, Xtr, ytr, Xeval, seed, **strength)."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .common import Scaled, make_model


def kc_weights(g, y):
    """Kamiran-Calders reweighing: w(g,y) = P(g)P(y)/P(g,y). g may be soft (prob of group 1)."""
    g = np.asarray(g, float)
    w = np.zeros(len(y))
    for gv in (0, 1):
        mg = g if gv == 1 else 1 - g
        for yv in (0, 1):
            my = (y == yv).astype(float)
            pgy = (mg * my).mean()
            if pgy > 0:
                w += mg * my * (mg.mean() * my.mean() / pgy)
    return w / w.mean()


def kc_weights_multi(c, y):
    w = np.zeros(len(y))
    for cv in np.unique(c):
        for yv in (0, 1):
            m = (c == cv) & (y == yv)
            if m.any():
                w[m] = (c == cv).mean() * (y == yv).mean() / m.mean()
    return w / w.mean()


# ---------------- demographic-free ----------------
def erm(kind, Xtr, ytr, Xte, seed, **_):
    return Scaled(make_model(kind, seed)).fit(Xtr, ytr).predict_proba(Xte)


def jtt(kind, Xtr, ytr, Xte, seed, lam=10, **_):
    """Just Train Twice (Liu et al. 2021): upweight training errors of an ERM model."""
    m1 = Scaled(make_model(kind, seed)).fit(Xtr, ytr)
    err = (m1.predict_proba(Xtr) >= 0.5).astype(int) != ytr
    w = np.where(err, lam, 1.0)
    return Scaled(make_model(kind, seed)).fit(Xtr, ytr, w / w.mean()).predict_proba(Xte)


def cvar_dro(kind, Xtr, ytr, Xte, seed, alpha=0.2, l2=1e-3, **_):
    """CVaR-DRO logistic regression (Hashimoto/Levy style): minimise mean loss of the worst alpha-fraction.
    Linear model only; smooth hinge via softplus. kind is ignored (always linear)."""
    sc = StandardScaler().fit(Xtr)
    Z = np.c_[sc.transform(Xtr), np.ones(len(Xtr))]
    s = 2 * ytr - 1
    k = 20.0

    def obj(v):
        th, eta = v[:-1], v[-1]
        mgn = s * (Z @ th)
        loss = np.logaddexp(0, -mgn)
        u = k * (loss - eta)
        sp = np.logaddexp(0, u) / k
        f = eta + sp.mean() / alpha + l2 * th[:-1] @ th[:-1]
        sig_u = expit(u)
        dloss = -s * expit(-mgn)
        gth = Z.T @ (sig_u * dloss) / (alpha * len(s))
        gth[:-1] += 2 * l2 * th[:-1]
        geta = 1 - sig_u.mean() / alpha
        return f, np.r_[gth, geta]

    v0 = np.zeros(Z.shape[1] + 1)
    r = minimize(obj, v0, jac=True, method="L-BFGS-B", options=dict(maxiter=500))
    th = r.x[:-1]
    Zt = np.c_[sc.transform(Xte), np.ones(len(Xte))]
    return expit(Zt @ th)


def cluster_reweigh(kind, Xtr, ytr, Xte, seed, k=6, gamma=1.0, **_):
    """Unsupervised pseudo-groups (k-means on features), then Kamiran-Calders reweighing over (cluster, y)."""
    Zs = StandardScaler().fit_transform(Xtr)
    c = KMeans(k, n_init=4, random_state=seed).fit_predict(Zs)
    w = kc_weights_multi(c, ytr) ** gamma
    w = w / w.mean()
    return Scaled(make_model(kind, seed)).fit(Xtr, ytr, w).predict_proba(Xte)


def arl(kind, Xtr, ytr, Xte, seed, steps=300, lr=0.05, adv_lr=0.05, l2=1e-3, **_):
    """Adversarially Reweighted Learning (Lahoti et al. 2020), linear learner + linear adversary on (x, y).
    lambda_i = 1 + n * sigmoid(f_adv(x_i,y_i)) / sum_j sigmoid(f_adv(x_j,y_j)). Full-batch alternating
    Adagrad updates. kind is ignored (linear)."""
    rng = np.random.default_rng(seed)
    sc = StandardScaler().fit(Xtr)
    Z = np.c_[sc.transform(Xtr), np.ones(len(Xtr))]
    Za = np.c_[Z, ytr * 2 - 1.0]
    n, d = Z.shape
    th = np.zeros(d)
    ph = rng.normal(0, 0.01, Za.shape[1])
    s = 2 * ytr - 1
    gth2 = np.full(d, 1e-8)
    gph2 = np.full(Za.shape[1], 1e-8)
    for t in range(steps):
        a = expit(Za @ ph)
        lam = 1 + n * a / a.sum()
        mgn = s * (Z @ th)
        loss = np.logaddexp(0, -mgn)
        # learner minimises mean(lam * loss)
        g = Z.T @ (lam * -s * expit(-mgn)) / n
        g[:-1] += 2 * l2 * th[:-1]
        gth2 += g * g
        th -= lr * g / np.sqrt(gth2)
        # adversary maximises mean(lam * loss); d lam_i / d ph via softmax-like normalisation
        S = a.sum()
        da = a * (1 - a)
        # J = sum_i loss_i * n * a_i / S / n = sum_i loss_i a_i / S
        dJ_da = loss / S - (loss * a).sum() / S ** 2
        gp = Za.T @ (dJ_da * da)
        gph2 += gp * gp
        ph += adv_lr * gp / np.sqrt(gph2)
    Zt = np.c_[sc.transform(Xte), np.ones(len(Xte))]
    return expit(Zt @ th)


