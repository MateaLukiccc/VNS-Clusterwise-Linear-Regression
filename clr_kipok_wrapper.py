from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.base import clone

# ── Import the real clr core (patch the two NumPy 1.24 deprecations) ─────────
# Apply monkey-patch before importing so the module loads cleanly.
import builtins, importlib, types, sys

def _load_patched_clr(path: str = "clr") -> types.ModuleType:
    """
    Import clr.py with np.int / np.float replaced at source level so it
    works under NumPy ≥ 1.24 without touching the original file.
    """
    import importlib.util, pathlib, re
    spec = importlib.util.spec_from_file_location("clr_orig", f"{path}.py")
    src  = pathlib.Path(f"{path}.py").read_text()
    src  = re.sub(r'\bnp\.int\b',   'int',   src)
    src  = re.sub(r'\bnp\.float\b', 'float', src)
    mod  = types.ModuleType("clr_orig")
    exec(compile(src, f"{path}.py", "exec"), mod.__dict__)
    return mod

_clr_mod  = _load_patched_clr("clr")
_best_clr = _clr_mod.best_clr          # the real best_clr


# ── Shared low-level helpers ──────────────────────────────────────────────────

def _models_to_W(models, D: int) -> np.ndarray:
    """List of fitted sklearn regressors → W of shape (K, D+1)."""
    K = len(models)
    W = np.zeros((K, D + 1))
    for k, m in enumerate(models):
        W[k, 0]  = float(m.intercept_) if np.ndim(m.intercept_) == 0 \
                   else float(m.intercept_[0])
        W[k, 1:] = m.coef_
    return W


def _linear_pred(x: np.ndarray, w: np.ndarray) -> float:
    return w[0] + np.dot(w[1:], x)


def _predict_all_clusters(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Returns (N, K) matrix of every cluster's prediction for every point."""
    X_ext = np.column_stack([np.ones(X.shape[0]), X])
    return X_ext @ W.T


# ── sklearn-compatible wrapper ────────────────────────────────────────────────

class CLR_Kipok:
    """
    Clusterwise Linear Regression (standard alternating-minimisation,
    Gitman et al. 2018) with multiple inference strategies.

    Parameters
    ----------
    K            : number of clusters
    num_tries    : restarts of the alternating optimiser
    max_iter     : EM iterations per restart
    kmeans_X     : k-means regulariser weight on X (0 = pure CLR)
    lr           : sklearn regressor per cluster (default: Ridge α=1e-5)
    n_estimators : trees in the RandomForest label predictor
    random_state : RNG seed
    """

    def __init__(
        self,
        K: int            = 2,
        num_tries: int    = 10,
        max_iter: int     = 5,
        kmeans_X: float   = 0.0,
        lr                = None,
        n_estimators: int = 20,
        random_state: int = 42,
    ):
        self.K            = K
        self.num_tries    = num_tries
        self.max_iter     = max_iter
        self.kmeans_X     = kmeans_X
        self.lr           = lr
        self.n_estimators = n_estimators
        self.random_state = random_state

        # set after fit()
        self.W_:       np.ndarray | None            = None
        self.a_:       np.ndarray | None            = None
        self.clf_:     RandomForestClassifier | None = None
        self.X_train_: np.ndarray | None            = None
        self.err_:     float | None                  = None

    # ── Training ──────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray) -> "CLR_Kipok":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        np.random.seed(self.random_state)

        labels, models, _, obj = _best_clr(
            X, y, k=self.K,
            kmeans_X=self.kmeans_X,
            max_iter=self.max_iter,
            num_tries=self.num_tries,
            lr=self.lr,
        )

        self.a_       = labels.astype(int)
        self.W_       = _models_to_W(models, X.shape[1])
        self.err_     = float(obj)
        self.X_train_ = X.copy()

        # CLRp classifier — RandomForest, matching original CLRpRegressor
        labels_for_fit = self.a_.copy()
        if len(np.unique(labels_for_fit)) < 2:
            # force at least 2 classes so RF doesn't crash
            labels_for_fit[0] = 1 - labels_for_fit[0]

        self.clf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        ).fit(X, labels_for_fit)

        return self

    # ── Prediction dispatcher ─────────────────────────────────────────────────
    def predict(self, X: np.ndarray, method: str = "clrp",
                K_neighbors: int = 5) -> np.ndarray:
        """
        Parameters
        ----------
        X            : test features
        method       : one of 'clrp', 'largest', 'weighted', 'knn',
                       'knn_weighted', 'centroid', 'distance', 'rmse_local'
        K_neighbors  : used by knn / knn_weighted / rmse_local
        """
        X = np.asarray(X, dtype=float)

        if method == "clrp":
            return self._predict_clrp(X)
        if method == "largest":
            return self._predict_largest(X)
        if method == "weighted":
            return self._predict_weighted(X)
        if method == "knn":
            return self._predict_knn(X, K_neighbors, weighted=False)
        if method == "knn_weighted":
            return self._predict_knn(X, K_neighbors, weighted=True)
        if method == "centroid":
            return self._predict_centroid(X)
        if method == "distance":
            return self._predict_distance(X)
        if method == "rmse_local":
            return self._predict_rmse_local(X, K_neighbors)
        raise ValueError(f"Unknown method '{method}'. Choose from: "
                         "clrp, largest, weighted, knn, knn_weighted, "
                         "centroid, distance, rmse_local")

    # ── Inference strategies ──────────────────────────────────────────────────

    def _predict_clrp(self, X: np.ndarray) -> np.ndarray:
        """RandomForest classifier → cluster regression (original CLRp)."""
        a_pred = self.clf_.predict(X).astype(int)
        X_ext  = np.column_stack([np.ones(X.shape[0]), X])
        return np.sum(X_ext * self.W_[a_pred], axis=1)

    def _predict_largest(self, X: np.ndarray) -> np.ndarray:
        """Use the regression of the most populated training cluster."""
        q     = int(np.argmax(np.bincount(self.a_, minlength=self.K)))
        X_ext = np.column_stack([np.ones(X.shape[0]), X])
        return X_ext @ self.W_[q]

    def _predict_weighted(self, X: np.ndarray) -> np.ndarray:
        """Cluster-size weighted average over all regressions."""
        weights = np.bincount(self.a_, minlength=self.K) / len(self.a_)
        Z       = _predict_all_clusters(X, self.W_)   # (N, K)
        return Z @ weights

    def _predict_knn(self, X: np.ndarray, K_neighbors: int,
                     weighted: bool) -> np.ndarray:
        """k-NN: majority vote (weighted=False) or weighted average (True)."""
        preds = np.empty(X.shape[0])
        for i, x in enumerate(X):
            dists      = np.linalg.norm(self.X_train_ - x, axis=1)
            nn_idx     = np.argsort(dists)[:K_neighbors]
            nn_clusters = self.a_[nn_idx]
            counts     = np.bincount(nn_clusters, minlength=self.K)
            if weighted:
                w = counts / K_neighbors
                z = np.array([_linear_pred(x, self.W_[k]) for k in range(self.K)])
                preds[i] = np.dot(w, z)
            else:
                q       = int(np.argmax(counts))
                preds[i] = _linear_pred(x, self.W_[q])
        return preds

    def _predict_centroid(self, X: np.ndarray) -> np.ndarray:
        """Assign each test point to the nearest cluster centroid."""
        centers = np.array([
            self.X_train_[self.a_ == k].mean(axis=0) for k in range(self.K)
        ])
        preds = np.empty(X.shape[0])
        for i, x in enumerate(X):
            q       = int(np.argmin(np.linalg.norm(centers - x, axis=1)))
            preds[i] = _linear_pred(x, self.W_[q])
        return preds

    def _predict_distance(self, X: np.ndarray) -> np.ndarray:
        """Inverse-squared centroid distance weighting over all regressions."""
        centers = np.array([
            self.X_train_[self.a_ == k].mean(axis=0) for k in range(self.K)
        ])
        preds = np.empty(X.shape[0])
        for i, x in enumerate(X):
            dists   = np.maximum(np.linalg.norm(centers - x, axis=1), 1e-10)
            inv_sq  = 1.0 / dists ** 2
            w       = inv_sq / inv_sq.sum()
            z       = np.array([_linear_pred(x, self.W_[k]) for k in range(self.K)])
            preds[i] = np.dot(w, z)
        return preds

    def _predict_rmse_local(self, X: np.ndarray,
                            K_neighbors: int) -> np.ndarray:
        """RMSE-based local weighting (Guo et al. variant)."""
        preds = np.empty(X.shape[0])
        for i, x in enumerate(X):
            dists      = np.linalg.norm(self.X_train_ - x, axis=1)
            nn_idx     = np.argsort(dists)[:K_neighbors]
            nn_clusters = self.a_[nn_idx]
            rmse_h     = dists[nn_idx]
            r_bar      = rmse_h.sum()

            if r_bar == 0:
                counts = np.bincount(nn_clusters, minlength=self.K)
                w = counts / K_neighbors
            else:
                denom = (K_neighbors - 1) * r_bar
                w = np.zeros(self.K)
                for h, k in enumerate(nn_clusters):
                    w[k] += (r_bar - rmse_h[h]) / denom
                w = np.maximum(w, 0.0)
                s = w.sum()
                w = w / s if s > 0 else np.ones(self.K) / self.K

            z       = np.array([_linear_pred(x, self.W_[k]) for k in range(self.K)])
            preds[i] = np.dot(w, z)
        return preds

    # ── Utilities ─────────────────────────────────────────────────────────────
    def cluster_sizes(self) -> np.ndarray:
        return np.bincount(self.a_, minlength=self.K)

    def __repr__(self) -> str:
        return (f"CLR_Kipok(K={self.K}, num_tries={self.num_tries}, "
                f"max_iter={self.max_iter}, kmeans_X={self.kmeans_X})")
