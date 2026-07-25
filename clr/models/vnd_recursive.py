import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .base import BaseCLR


class _StopFirstImprovement(Exception):
    pass


class CLR_VND_Recursive(BaseCLR):
    """Base for recursive VND variants. Implements the shared recursive search."""
    
    def _search_neighborhood(self, X, y, assignments, cluster_sizes, empty_count, point_idx, 
                             moves_left, best_candidate):
        N = X.shape[0]
        if moves_left == 0:
            if empty_count == 0:
                weights = self._train_regression(X, y, assignments, self.K)
                classifier = self._train_classifier(X, assignments) if self.alpha < 1.0 else None
                err = self._total_error(X, y, classifier, weights, assignments, self.alpha)
                
                if err < best_candidate["error"]:
                    best_candidate["error"] = err
                    best_candidate["assignments"] = assignments.copy()
                    best_candidate["weights"] = weights
                    if self.strategy == "first":
                        raise _StopFirstImprovement
            return

        if moves_left > N - point_idx:
            return

        self._search_neighborhood(X, y, assignments, cluster_sizes, empty_count, 
                                  point_idx + 1, moves_left, best_candidate)
        
        old_cluster = assignments[point_idx]
        for offset in range(1, self.K):
            new_cluster = (old_cluster + offset) % self.K
            empty_count = self._reassign_point(assignments, point_idx, new_cluster, cluster_sizes, empty_count)
            self._search_neighborhood(X, y, assignments, cluster_sizes, empty_count, 
                                      point_idx + 1, moves_left - 1, best_candidate)
            empty_count = self._reassign_point(assignments, point_idx, old_cluster, cluster_sizes, empty_count)

    def _run_vnd(self, X, y, weights_init, assignments_init):
        assignments_star = assignments_init.copy()
        weights_star = weights_init.copy()
        classifier_star = self._train_classifier(X, assignments_star) if self.alpha < 1.0 else None
        err_star = self._total_error(X, y, classifier_star, weights_star, assignments_star, self.alpha)

        neighborhood_size = 1
        while neighborhood_size <= self.l_max:
            cluster_sizes = np.bincount(assignments_star, minlength=self.K).astype(int)
            empty_count = int(np.sum(cluster_sizes == 0))
            if empty_count != 0:
                raise ValueError("Initial solution contains an empty cluster")

            best_candidate = {
                "error": err_star,
                "assignments": assignments_star.copy(),
                "weights": weights_star.copy()
            }
            try:
                self._search_neighborhood(X, y, assignments_star.copy(), cluster_sizes, 
                                          empty_count, 0, neighborhood_size, best_candidate)
            except _StopFirstImprovement:
                pass

            if best_candidate["error"] < err_star - 1e-10:
                err_star = best_candidate["error"]
                assignments_star = best_candidate["assignments"]
                weights_star = best_candidate["weights"]
                neighborhood_size = 1
            else:
                neighborhood_size += 1

        classifier_star = self._train_classifier(X, assignments_star)
        return classifier_star, weights_star, assignments_star, err_star


class CLR_VND_Logistic(CLR_VND_Recursive):
    """Recursive VND using Logistic Regression for cluster assignment."""
    def _train_classifier(self, X, assignments):
        if self.K == 1 or len(np.unique(assignments)) < 2:
            return None
        return LogisticRegression().fit(X, assignments)

    def __repr__(self):
        return (f"CLR_VND_Logistic(K={self.K}, l_max={self.l_max}, "
                f"alpha={self.alpha}, strategy={self.strategy!r})")


class CLR_VND_Tree(CLR_VND_Recursive):
    """Recursive VND using Random Forest for cluster assignment."""
    def _train_classifier(self, X, assignments):
        if self.K == 1 or len(np.unique(assignments)) < 2:
            return None
        return RandomForestClassifier(
            n_estimators=20, random_state=self.random_state, n_jobs=-1
        ).fit(X, assignments)

    def __repr__(self):
        return (f"CLR_VND_Tree(K={self.K}, l_max={self.l_max}, "
                f"alpha={self.alpha}, strategy={self.strategy!r})")