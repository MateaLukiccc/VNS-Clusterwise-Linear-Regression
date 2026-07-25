import numpy as np
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss, mean_squared_error


class BaseCLR(ABC):
    """Base class for Clusterwise Linear Regression models."""
    
    def __init__(self, K: int = 2, l_max: int = 1, alpha: float = 1.0, 
                 strategy: str = "best", random_state: int = 42):
        self.K = K
        self.l_max = l_max
        self.alpha = alpha
        self.strategy = strategy
        self.random_state = random_state
        
        self.classifier_ = None
        self.weights_ = None
        self.assignments_ = None
        self.err_ = None

    # ── Shared Static Math Helpers ────────────────────────────────────────────
    @staticmethod
    def _train_regression(X, y, assignments, K):
        _, D = X.shape
        weights = np.zeros((K, D + 1))
        for k in range(K):
            idx = np.where(assignments == k)[0]
            if len(idx) == 0:
                raise ValueError(f"Empty cluster {k} during regression fit")
            reg = LinearRegression().fit(X[idx], y[idx])
            weights[k, 0]  = reg.intercept_
            weights[k, 1:] = reg.coef_
        return weights

    @staticmethod
    def _predict_regression(X, weights, assignments):
        X_ext = np.column_stack([np.ones(X.shape[0]), X])
        return np.sum(X_ext * weights[assignments], axis=1)

    @staticmethod
    def _regression_error(X, y, weights, assignments):
        return mean_squared_error(y, BaseCLR._predict_regression(X, weights, assignments))

    @staticmethod
    def _classification_error(X, assignments, classifier):
        if classifier is None:
            raise ValueError("Classifier is None but classification error requested")
        return log_loss(assignments, classifier.predict_proba(X), labels=classifier.classes_)

    @staticmethod
    def _total_error(X, y, classifier, weights, assignments, alpha):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        reg_err = BaseCLR._regression_error(X, y, weights, assignments)
        if alpha == 1.0:
            return reg_err
        return alpha * reg_err + (1.0 - alpha) * BaseCLR._classification_error(X, assignments, classifier)

    @staticmethod
    def _reassign_point(assignments, point_idx, new_cluster, cluster_sizes, empty_count):
        old_cluster = assignments[point_idx]
        if old_cluster == new_cluster:
            return empty_count
        if cluster_sizes[old_cluster] == 1:
            empty_count += 1
        if cluster_sizes[new_cluster] == 0:
            empty_count -= 1
        cluster_sizes[old_cluster] -= 1
        cluster_sizes[new_cluster] += 1
        assignments[point_idx] = new_cluster
        return empty_count

    # ── Abstract Methods (To be implemented by subclasses) ────────────────────
    @abstractmethod
    def _train_classifier(self, X, assignments):
        """Train the cluster assignment classifier."""
        pass

    @abstractmethod
    def _run_vnd(self, X, y, weights_init, assignments_init):
        """Execute the Variable Neighborhood Descent search."""
        pass

    # ── Public API ────────────────────────────────────────────────────────────
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        km = KMeans(n_clusters=self.K, random_state=self.random_state, n_init=10)
        assignments = km.fit_predict(X).astype(int)
        weights = self._train_regression(X, y, assignments, self.K)
        
        self.classifier_, self.weights_, self.assignments_, self.err_ = self._run_vnd(
            X, y, weights, assignments
        )
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.classifier_ is not None:
            assignments_pred = self.classifier_.predict(X).astype(int)
        else:
            assignments_pred = np.zeros(X.shape[0], dtype=int)
        return self._predict_regression(X, self.weights_, assignments_pred)

    def cluster_sizes(self):
        return np.bincount(self.assignments_, minlength=self.K)