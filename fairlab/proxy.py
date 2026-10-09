"""Choosing which rows get a demographic label, and fitting the proxy P(A|X) from them.

All strategies select from a labelling POOL (a held-out split with outcomes but, by default, no
demographics). Only selected rows reveal A.
"""
import numpy as np
from scipy.special import expit, logit
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class Proxy:
    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, a, prior=None):
        self.sc = StandardScaler().fit(X)
        if len(np.unique(a)) < 2:
            self.const = float(np.mean(a))
            return self
        self.const = None
        self.m = LogisticRegression(C=self.C, max_iter=2000).fit(self.sc.transform(X), a)
        self.shift = 0.0
        if prior is not None:  # recalibrate intercept so the pool-average prediction matches the prior
            self.shift = 0.0
        return self

    def calibrate_to(self, Xpool, prior):
        """Shift the logit so mean prediction over Xpool equals prior (corrects selection-induced prior shift)."""
        if self.const is not None:
            return self
        s = self.m.decision_function(self.sc.transform(Xpool))
        lo, hi = -10.0, 10.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if expit(s + mid).mean() > prior:
                hi = mid
            else:
                lo = mid
        self.shift = (lo + hi) / 2
        return self

    def predict(self, X):
        if self.const is not None:
            return np.full(len(X), self.const)
        return expit(self.m.decision_function(self.sc.transform(X)) + getattr(self, "shift", 0.0))


def _ensure_both(idx, a_pool, rng, pool_n):
    """Keep sampling random rows until the labelled set contains both groups (a real labeller would)."""
    idx = list(idx)
    tries = 0
    while len(np.unique(a_pool[idx])) < 2 and tries < pool_n:
        cand = rng.integers(pool_n)
        if cand not in idx:
            idx.append(cand)
        tries += 1
    return np.array(idx)


def select(strategy, n, Xpool, ypool, a_pool, rng, mnar_ratio=None):
    """Return (indices into pool, prior estimate or None). a_pool is only *read* for selected rows."""
    N = len(ypool)
    n = min(n, N // 2)
    if strategy == "random":
        if mnar_ratio is not None:  # group 1 discloses with relative probability mnar_ratio
            w = np.where(a_pool == 1, mnar_ratio, 1.0)
            idx = rng.choice(N, n, replace=False, p=w / w.sum())
        else:
            idx = rng.choice(N, n, replace=False)
        return _ensure_both(idx, a_pool, rng, N), None
    if strategy == "strat_y":
        idx = np.concatenate([rng.choice(np.flatnonzero(ypool == v), n // 2, replace=False) for v in (0, 1)])
        return _ensure_both(idx, a_pool, rng, N), None
    if strategy == "diverse":
        Z = StandardScaler().fit_transform(Xpool)
        km = KMeans(n, n_init=1, random_state=int(rng.integers(1 << 30))).fit(Z)
        d = km.transform(Z)
        idx, used = [], set()
        for c in range(n):
            for j in np.argsort(d[:, c]):
                if j not in used:
                    used.add(j)
                    idx.append(j)
                    break
        return _ensure_both(idx, a_pool, rng, N), None
    if strategy in ("uncertainty", "minority"):
        n0 = max(10, n // 4)
        idx = list(_ensure_both(rng.choice(N, n0, replace=False), a_pool, rng, N))
        prior = a_pool[idx[:n0]].mean()  # from the random seed batch only
        nb = 3
        per = int(np.ceil((n - len(idx)) / nb)) if n > len(idx) else 0
        for _ in range(nb):
            if len(idx) >= n or per <= 0:
                break
            q = Proxy().fit(Xpool[idx], a_pool[idx]).calibrate_to(Xpool, max(prior, 1e-3)).predict(Xpool)
            q[idx] = np.nan
            minority = 1 if prior < 0.5 else 0
            if strategy == "uncertainty":
                score = -np.abs(q - 0.5)
            else:
                score = q if minority == 1 else 1 - q
            score = np.where(np.isnan(score), -np.inf, score)
            take = [j for j in np.argsort(-score) if j not in set(idx)][: min(per, n - len(idx))]
            idx += take
        return np.array(idx), prior
    raise ValueError(strategy)


def fit_proxy(Xpool, a_pool, idx, prior=None):
    pr = Proxy().fit(Xpool[idx], a_pool[idx])
    if prior is not None:
        pr.calibrate_to(Xpool, prior)
    return pr
