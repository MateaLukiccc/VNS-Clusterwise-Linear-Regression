import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, log_loss
from metrics import rmse


def _move(a, i, new_cluster, sigma, kappa):
    """Reassign point i to new_cluster. Updates sigma in place. Returns updated kappa."""
    old_cluster = a[i]
    if old_cluster == new_cluster:
        return kappa
    if sigma[old_cluster] == 1:
        kappa += 1                          # old cluster becomes empty
    if sigma[new_cluster] == 0:
        kappa -= 1                          # new cluster was empty
    sigma[old_cluster] -= 1
    sigma[new_cluster] += 1
    a[i] = new_cluster
    return kappa


def _train_regression(X, y, a, K):
    """One OLS fit per cluster. Returns W of shape (K, D+1) with intercept in column 0."""
    # TODO: add lambda_r / p_reg to switch in Ridge or Lasso.
    _, D = X.shape
    W = np.zeros((K, D + 1))
    for k in range(K):
        idx = np.where(a == k)[0]
        if len(idx) == 0:
            raise ValueError("Empty cluster {} during regression fit".format(k))
        reg = LinearRegression().fit(X[idx], y[idx])
        W[k, 0]  = reg.intercept_
        W[k, 1:] = reg.coef_
    return W


def _train_classification(X, a, K):
    """Multinomial logistic regression on (X, a). Returns a FRESH classifier every call."""
    # TODO: add lambda_c (set sklearn C = 1.0 / lambda_c).
    if K == 1 or len(np.unique(a)) < 2:
        return None
    return LogisticRegression().fit(X, a)


def _predict_reg(X, W, a):
    """For each point, predict using its assigned cluster's linear model."""
    X_ext = np.column_stack([np.ones(X.shape[0]), X])
    return np.sum(X_ext * W[a], axis=1)


def _reg_error(X, y, W, a):
    # TODO: add + lambda_r * sum_k ||w_k||_p^p
    return mean_squared_error(y, _predict_reg(X, W, a))


def _cls_error(X, a, Theta):
    # TODO: add + lambda_c * ||Theta||_p^p
    if Theta is None:
        raise ValueError("Classifier is None but classification error was requested")
    return log_loss(a, Theta.predict_proba(X), labels=Theta.classes_)


def _error(X, y, Theta, W, a, alpha):
    r = _reg_error(X, y, W, a)
    if alpha == 1.0:
        return r
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return alpha * r + (1.0 - alpha) * _cls_error(X, a, Theta)


class _StopFirstImprovement(Exception):
    pass


def _find_best_neighbor(X, y, a, sigma, kappa, i, p, best, K, alpha, strategy):
    """
    Recursive enumeration of N_{p,i}(a). Modifies a and sigma in place
    and restores them on backtrack. best is [err, a_copy, W].
    Theta is NOT stored here, it is refit once at the end of VND.
    """
    N = X.shape[0]

    if p == 0:
        if kappa == 0:
            W = _train_regression(X, y, a, K)
            if alpha < 1.0:
                Theta = _train_classification(X, a, K)
            else:
                Theta = None                 # not needed when classification ignored
            err = _error(X, y, Theta, W, a, alpha)
            if err < best[0]:
                best[0] = err
                best[1] = a.copy()
                best[2] = W
                if strategy == "first":
                    raise _StopFirstImprovement
        return

    if p > N - i:
        return

    # Option A: leave i unchanged
    _find_best_neighbor(X, y, a, sigma, kappa, i + 1, p, best, K, alpha, strategy)

    # Option B: move i to each of the K-1 other clusters
    old_k = a[i]
    for offset in range(1, K):
        new_k = (old_k + offset) % K
        kappa = _move(a, i, new_k, sigma, kappa)
        _find_best_neighbor(X, y, a, sigma, kappa, i + 1, p - 1, best, K, alpha, strategy)
        kappa = _move(a, i, old_k, sigma, kappa)


def _vnd(X, y, W_init, a_init, l_max, K, alpha, strategy):
    a_star = a_init.copy()
    W_star = W_init.copy()
    Theta_star = _train_classification(X, a_star, K) if alpha < 1.0 else None
    err_star = _error(X, y, Theta_star, W_star, a_star, alpha)

    l = 1
    while l <= l_max:
        sigma = np.bincount(a_star, minlength=K).astype(int)
        kappa = int(np.sum(sigma == 0))
        if kappa != 0:
            raise ValueError("Initial solution contains an empty cluster")

        best = [err_star, a_star.copy(), W_star.copy()]   # err, a, W only
        try:
            _find_best_neighbor(X, y, a_star.copy(), sigma, kappa, 0, l,
                                best, K, alpha, strategy)
        except _StopFirstImprovement:
            pass

        if best[0] < err_star - 1e-10:
            err_star = best[0]
            a_star   = best[1]
            W_star   = best[2]
            l = 1
        else:
            l += 1

    # Refit the classifier from the FINAL a_star.
    Theta_star = _train_classification(X, a_star, K)
    return Theta_star, W_star, a_star, err_star


class CLR_VND:
    """
    Clusterwise Linear Regression trained with Variable Neighborhood Descent.
    Bare-bones: no regularization, no shaking, no GVNS.
    """

    def __init__(self, K=2, l_max=1, alpha=1.0, strategy="best", random_state=42):
        self.K = K
        self.l_max = l_max
        self.alpha = alpha
        self.strategy = strategy
        self.random_state = random_state
        # TODO: lambda_r, p_reg, lambda_c when regularization is added.

        self.Theta_ = None
        self.W_     = None
        self.a_     = None
        self.err_   = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # init: clustering followed by linear regression
        km = KMeans(n_clusters=self.K, random_state=self.random_state, n_init=10)
        a  = km.fit_predict(X).astype(int)
        W  = _train_regression(X, y, a, self.K)

        self.Theta_, self.W_, self.a_, self.err_ = _vnd(
            X, y, W, a, self.l_max, self.K, self.alpha, self.strategy
        )
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.Theta_ is not None:
            a_pred = self.Theta_.predict(X).astype(int)
        else:
            a_pred = np.zeros(X.shape[0], dtype=int)
        return _predict_reg(X, self.W_, a_pred)

    def cluster_sizes(self):
        return np.bincount(self.a_, minlength=self.K)

    def __repr__(self):
        return ("CLR_VND(K={}, l_max={}, alpha={}, strategy={!r})"
                .format(self.K, self.l_max, self.alpha, self.strategy))


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    N = 600
    X1 = rng.uniform(0, 5,  (N // 2, 2))
    y1 = 2 * X1[:, 0] - 1 * X1[:, 1] + rng.normal(0, 0.3, N // 2)
    X2 = rng.uniform(5, 10, (N // 2, 2))
    y2 = -1 * X2[:, 0] + 3 * X2[:, 1] + rng.normal(0, 0.3, N // 2)
    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])

    model = CLR_VND(K=2, l_max=1, strategy="first", random_state=42).fit(X, y)
    lin   = LinearRegression().fit(X, y)

    print("CLR_VND RMSE :", round(rmse(y, model.predict(X)), 4))
    print("MLR RMSE     :", round(rmse(y, lin.predict(X)), 4))
    print("Cluster sizes:", model.cluster_sizes())