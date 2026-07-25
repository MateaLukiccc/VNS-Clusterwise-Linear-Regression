from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ..core.gitman_clr import best_clr


def _models_to_weights(models, D: int) -> np.ndarray:
    """Convert list of fitted sklearn regressors → weights array of shape (K, D+1)."""
    K = len(models)
    weights = np.zeros((K, D + 1))
    for k, model in enumerate(models):
        intercept = model.intercept_
        weights[k, 0]  = float(intercept) if np.ndim(intercept) == 0 else float(intercept[0])
        weights[k, 1:] = model.coef_
    return weights


def _linear_predict(x: np.ndarray, w: np.ndarray) -> float:
    """Predict for a single row x using weight vector w = [intercept, coef...]."""
    return w[0] + np.dot(w[1:], x)


def _predict_all_clusters(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Returns (N, K) matrix: prediction of every cluster for every point."""
    X_ext = np.column_stack([np.ones(X.shape[0]), X])
    return X_ext @ weights.T


class CLR_Kipok:
    """Clusterwise Linear Regression (standard alternating-minimisation).
    
    Parameters
    ----------
    K            : Number of clusters.
    num_tries    : Restarts of the alternating optimiser.
    max_iter     : EM iterations per restart.
    kmeans_X     : k-means regulariser weight on X (0 = pure CLR).
    lr           : sklearn regressor per cluster (default: Ridge α=1e-5).
    n_estimators : Trees in the RandomForest label predictor.
    random_state : RNG seed.
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

        self.weights_:       np.ndarray | None             = None
        self.assignments_:   np.ndarray | None             = None
        self.classifier_:    RandomForestClassifier | None = None
        self.X_train_:       np.ndarray | None             = None
        self.err_:           float | None                  = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CLR_Kipok":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        np.random.seed(self.random_state)

        labels, models, _, obj = best_clr(
            X, y, k=self.K,
            kmeans_X=self.kmeans_X,
            max_iter=self.max_iter,
            num_tries=self.num_tries,
            lr=self.lr,
        )

        self.assignments_   = labels.astype(int)
        self.weights_       = _models_to_weights(models, X.shape[1])
        self.err_           = float(obj)
        self.X_train_       = X.copy()

        labels_for_fit = self.assignments_.copy()
        if len(np.unique(labels_for_fit)) < 2:
            labels_for_fit[0] = 1 - labels_for_fit[0]  # Force at least 2 classes

        self.classifier_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        ).fit(X, labels_for_fit)

        return self

    def predict(self, X: np.ndarray, method: str = "clrp", K_neighbors: int = 5) -> np.ndarray:
        """Dispatch prediction based on inference strategy."""
        X = np.asarray(X, dtype=float)

        if method == "clrp":
            return self._predict_random_forest(X)
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
            
        raise ValueError(f"Unknown method '{method}'.")

    def _predict_random_forest(self, X: np.ndarray) -> np.ndarray:
        """RandomForest classifier → cluster regression (original CLRp)."""
        assignments_pred = self.classifier_.predict(X).astype(int)
        X_ext  = np.column_stack([np.ones(X.shape[0]), X])
        return np.sum(X_ext * self.weights_[assignments_pred], axis=1)

    def _predict_largest(self, X: np.ndarray) -> np.ndarray:
        """Use the regression of the most populated training cluster."""
        dominant_cluster = int(np.argmax(np.bincount(self.assignments_, minlength=self.K)))
        X_ext = np.column_stack([np.ones(X.shape[0]), X])
        return X_ext @ self.weights_[dominant_cluster]

    def _predict_weighted(self, X: np.ndarray) -> np.ndarray:
        """Cluster-size weighted average over all regressions."""
        weights = np.bincount(self.assignments_, minlength=self.K) / len(self.assignments_)
        all_preds = _predict_all_clusters(X, self.weights_)
        return all_preds @ weights

    def _predict_knn(self, X: np.ndarray, K_neighbors: int, weighted: bool) -> np.ndarray:
        """k-NN: majority vote (weighted=False) or weighted average (True)."""
        preds = np.empty(X.shape[0])
        for i, x in enumerate(X):
            dists       = np.linalg.norm(self.X_train_ - x, axis=1)
            nn_idx      = np.argsort(dists)[:K_neighbors]
            nn_clusters = self.assignments_[nn_idx]
            counts      = np.bincount(nn_clusters, minlength=self.K)
            
            if weighted:
                cluster_weights = counts / K_neighbors
                cluster_preds   = np.array([_linear_predict(x, self.weights_[k]) for k in range(self.K)])
                preds[i]        = np.dot(cluster_weights, cluster_preds)
            else:
                best_cluster = int(np.argmax(counts))
                preds[i]     = _linear_predict(x, self.weights_[best_cluster])
        return preds

    def _predict_centroid(self, X: np.ndarray) -> np.ndarray:
        """Assign each test point to the nearest cluster centroid."""
        centers = np.array([
            self.X_train_[self.assignments_ == k].mean(axis=0) for k in range(self.K)
        ])
        preds = np.empty(X.shape[0])
        for i, x in enumerate(X):
            best_cluster = int(np.argmin(np.linalg.norm(centers - x, axis=1)))
            preds[i]     = _linear_predict(x, self.weights_[best_cluster])
        return preds

    def _predict_distance(self, X: np.ndarray) -> np.ndarray:
        """Inverse-squared centroid distance weighting over all regressions."""
        centers = np.array([
            self.X_train_[self.assignments_ == k].mean(axis=0) for k in range(self.K)
        ])
        preds = np.empty(X.shape[0])
        for i, x in enumerate(X):
            dists          = np.maximum(np.linalg.norm(centers - x, axis=1), 1e-10)
            inv_sq         = 1.0 / dists ** 2
            cluster_weights = inv_sq / inv_sq.sum()
            cluster_preds   = np.array([_linear_predict(x, self.weights_[k]) for k in range(self.K)])
            preds[i]        = np.dot(cluster_weights, cluster_preds)
        return preds

    def _predict_rmse_local(self, X: np.ndarray, K_neighbors: int) -> np.ndarray:
        """RMSE-based local weighting (Guo et al. variant)."""
        preds = np.empty(X.shape[0])
        for i, x in enumerate(X):
            dists       = np.linalg.norm(self.X_train_ - x, axis=1)
            nn_idx      = np.argsort(dists)[:K_neighbors]
            nn_clusters = self.assignments_[nn_idx]
            rmse_h      = dists[nn_idx]
            r_bar       = rmse_h.sum()

            if r_bar == 0:
                counts = np.bincount(nn_clusters, minlength=self.K)
                cluster_weights = counts / K_neighbors
            else:
                denom = (K_neighbors - 1) * r_bar
                cluster_weights = np.zeros(self.K)
                for h, k in enumerate(nn_clusters):
                    cluster_weights[k] += (r_bar - rmse_h[h]) / denom
                cluster_weights = np.maximum(cluster_weights, 0.0)
                s = cluster_weights.sum()
                cluster_weights = cluster_weights / s if s > 0 else np.ones(self.K) / self.K

            cluster_preds = np.array([_linear_predict(x, self.weights_[k]) for k in range(self.K)])
            preds[i]      = np.dot(cluster_weights, cluster_preds)
        return preds

    def cluster_sizes(self) -> np.ndarray:
        return np.bincount(self.assignments_, minlength=self.K)

    def __repr__(self) -> str:
        return (f"CLR_Kipok(K={self.K}, num_tries={self.num_tries}, "
                f"max_iter={self.max_iter}, kmeans_X={self.kmeans_X})")