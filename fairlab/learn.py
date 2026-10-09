"""Learners with a soft equalized-odds penalty.

Objective:  mean logloss + l2*||w||^2 + mu * sum_k sum_{y in {0,1}} ( E_w[p | g_k=1, y] - E_w[p | g_k=0, y] )^2
where g_k in [0,1] are (possibly soft) group memberships and an optional row mask m in [0,1]
restricts which rows enter the penalty (used by the FairDSR-style confident-proxy variant).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler


def _cells(G, y, mask=None):
    G = np.asarray(G, float)
    if G.ndim == 1:
        G = G[:, None]
    m = np.ones(len(y)) if mask is None else np.asarray(mask, float)
    cells = []
    for k in range(G.shape[1]):
        g = G[:, k]
        for yv in (0, 1):
            my = (y == yv) * m
            w1, w0 = g * my, (1 - g) * my
            if w1.sum() > 1 and w0.sum() > 1:
                cells.append(w1 / w1.sum() - w0 / w0.sum())
    return np.array(cells) if cells else np.zeros((0, len(y)))


class FairLR:
    def __init__(self, mu=0.0, l2=1e-3):  # 1e-3 frozen after pilot (conditioning); pilot used 1e-4
        self.mu, self.l2 = mu, l2

    def fit(self, X, y, G=None, mask=None, init=None):
        self.sc = StandardScaler().fit(X)
        Z = np.c_[self.sc.transform(X), np.ones(len(X))]
        n = len(y)
        C = _cells(G, y, mask) if (G is not None and self.mu > 0) else np.zeros((0, n))
        mu, l2 = self.mu, self.l2

        def obj(th):
            s = Z @ th
            p = expit(s)
            f = np.mean(np.logaddexp(0, s) - y * s) + l2 * th[:-1] @ th[:-1]
            g = Z.T @ (p - y) / n
            g[:-1] += 2 * l2 * th[:-1]
            if len(C):
                d = C @ p
                f += mu * d @ d
                g += Z.T @ (p * (1 - p) * (2 * mu * (C.T @ d)))
            return f, g

        th0 = np.zeros(Z.shape[1]) if init is None else init.th
        self.th = minimize(obj, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=1000)).x
        return self

    def predict_proba(self, X):
        return expit(np.c_[self.sc.transform(X), np.ones(len(X))] @ self.th)


class FairMLP:
    """One-hidden-layer MLP (tanh) with the same penalty, full-batch Adam via autograd."""

    def __init__(self, mu=0.0, hidden=32, steps=400, lr=0.01, l2=1e-4, seed=0):
        self.mu, self.h, self.steps, self.lr, self.l2, self.seed = mu, hidden, steps, lr, l2, seed

    def fit(self, X, y, G=None, mask=None, init=None, sample_weight=None):
        import autograd.numpy as anp
        from autograd import grad
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        n, d = Z.shape
        C = _cells(G, y, mask) if (G is not None and self.mu > 0) else np.zeros((0, n))
        w = np.ones(n) if sample_weight is None else sample_weight / np.mean(sample_weight)
        rng = np.random.default_rng(self.seed)
        if init is not None:
            P = [p.copy() for p in init.P]
        else:
            P = [rng.normal(0, 1 / np.sqrt(d), (d, self.h)), np.zeros(self.h), rng.normal(0, 1 / np.sqrt(self.h), self.h), np.zeros(1)]
        mu, l2 = self.mu, self.l2

        def loss(P):
            hdn = anp.tanh(anp.dot(Z, P[0]) + P[1])
            s = anp.dot(hdn, P[2]) + P[3][0]
            p = 1 / (1 + anp.exp(-s))
            f = anp.mean(w * (anp.logaddexp(0, s) - y * s)) + l2 * (anp.sum(P[0] ** 2) + anp.sum(P[2] ** 2))
            if len(C):
                dvec = anp.dot(C, p)
                f = f + mu * anp.sum(dvec ** 2)
            return f

        g = grad(loss)
        m = [np.zeros_like(p) for p in P]
        v = [np.zeros_like(p) for p in P]
        for t in range(1, self.steps + 1):
            gr = g(P)
            for i in range(4):
                m[i] = 0.9 * m[i] + 0.1 * gr[i]
                v[i] = 0.999 * v[i] + 0.001 * gr[i] ** 2
                P[i] = P[i] - self.lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        self.P = P
        return self

    def predict_proba(self, X):
        Z = self.sc.transform(X)
        s = np.tanh(Z @ self.P[0] + self.P[1]) @ self.P[2] + self.P[3][0]
        return expit(s)


class ARLMLP:
    """Adversarially Reweighted Learning (Lahoti et al., 2020) with an MLP learner and a linear adversary on
    (x, y), as in the original paper's tabular setup. Example weights lambda_i = 1 + n * a_i / sum_j a_j,
    a_i = sigmoid(adversary(x_i, y_i)). Alternating full-batch Adam steps (learner minimises, adversary maximises)."""

    def __init__(self, hidden=32, steps=400, lr=0.01, adv_lr=0.01, l2=1e-4, seed=0):
        self.h, self.steps, self.lr, self.adv_lr, self.l2, self.seed = hidden, steps, lr, adv_lr, l2, seed

    def fit(self, X, y, **_):
        import autograd.numpy as anp
        from autograd import grad
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        n, d = Z.shape
        Za = np.c_[Z, 2 * y - 1.0]
        rng = np.random.default_rng(self.seed)
        P = [rng.normal(0, 1 / np.sqrt(d), (d, self.h)), np.zeros(self.h), rng.normal(0, 1 / np.sqrt(self.h), self.h), np.zeros(1)]
        Q = [rng.normal(0, 0.01, Za.shape[1]), np.zeros(1)]
        l2 = self.l2

        def losses(P, Q):
            hdn = anp.tanh(anp.dot(Z, P[0]) + P[1])
            s = anp.dot(hdn, P[2]) + P[3][0]
            ll = anp.logaddexp(0, s) - y * s
            a = 1 / (1 + anp.exp(-(anp.dot(Za, Q[0]) + Q[1][0])))
            lam = 1 + n * a / anp.sum(a)
            return anp.mean(lam * ll), l2 * (anp.sum(P[0] ** 2) + anp.sum(P[2] ** 2))

        gl = grad(lambda P, Q: sum(losses(P, Q)), 0)
        ga = grad(lambda P, Q: -losses(P, Q)[0], 1)

        def adam(params, grads, state, lr, t):
            m, v = state
            for i in range(len(params)):
                m[i] = 0.9 * m[i] + 0.1 * grads[i]
                v[i] = 0.999 * v[i] + 0.001 * grads[i] ** 2
                params[i] = params[i] - lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        sP = ([np.zeros_like(p) for p in P], [np.zeros_like(p) for p in P])
        sQ = ([np.zeros_like(q) for q in Q], [np.zeros_like(q) for q in Q])
        for t in range(1, self.steps + 1):
            adam(P, gl(P, Q), sP, self.lr, t)
            adam(Q, ga(P, Q), sQ, self.adv_lr, t)
        self.P = P
        return self

    predict_proba = FairMLP.predict_proba
