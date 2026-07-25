import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base import BaseCLR

VND_TIME_LIMIT_SECONDS = 30 * 60  # 30 minutes


class CLR_VND_TreeIterative(BaseCLR):
    """Iterative VND using Random Forest for cluster assignment. Bounded by a time limit."""
    
    def _train_classifier(self, X, assignments):
        if self.K == 1 or len(np.unique(assignments)) < 2:
            return None
        return RandomForestClassifier(
            n_estimators=20, random_state=self.random_state, n_jobs=-1
        ).fit(X, assignments)

    def _find_best_neighbor_iterative(self, X, y, assignments, deadline):
        N = X.shape[0]
        weights_cur = self._train_regression(X, y, assignments, self.K)
        classifier_cur = self._train_classifier(X, assignments) if self.alpha < 1.0 else None
        err_cur = self._total_error(X, y, classifier_cur, weights_cur, assignments, self.alpha)

        best_err = err_cur
        best_assignments = assignments.copy()
        best_weights = weights_cur.copy()
        improved = False

        for i in range(N):
            if time.monotonic() >= deadline:
                print(f"    [VND] Time limit reached at point {i}/{N} — stopping search", flush=True)
                break

            original_cluster = int(assignments[i])
            for offset in range(1, self.K):
                new_cluster = (original_cluster + offset) % self.K
                assignments[i] = new_cluster

                if np.sum(assignments == original_cluster) == 0:
                    assignments[i] = original_cluster
                    continue

                try:
                    weights_cand = self._train_regression(X, y, assignments, self.K)
                    classifier_cand = self._train_classifier(X, assignments) if self.alpha < 1.0 else None
                    err_cand = self._total_error(X, y, classifier_cand, weights_cand, assignments, self.alpha)
                except Exception:
                    assignments[i] = original_cluster
                    continue

                if err_cand < best_err - 1e-10:
                    best_err = err_cand
                    best_assignments = assignments.copy()
                    best_weights = weights_cand.copy()
                    improved = True

                    if self.strategy == "first":
                        assignments[i] = original_cluster
                        return improved, best_err, best_assignments, best_weights

                assignments[i] = original_cluster

        return improved, best_err, best_assignments, best_weights

    def _run_vnd(self, X, y, weights_init, assignments_init):
        deadline = time.monotonic() + VND_TIME_LIMIT_SECONDS
        
        assignments_star = assignments_init.copy()
        weights_star = weights_init.copy()
        classifier_star = self._train_classifier(X, assignments_star) if self.alpha < 1.0 else None
        err_star = self._total_error(X, y, classifier_star, weights_star, assignments_star, self.alpha)

        neighborhood_size = 1
        while neighborhood_size <= self.l_max:
            if time.monotonic() >= deadline:
                print("    [VND] Overall time limit reached — exiting VND loop", flush=True)
                break

            improved, new_err, new_assignments, new_weights = self._find_best_neighbor_iterative(
                X, y, assignments_star.copy(), deadline
            )

            if improved:
                err_star = new_err
                assignments_star = new_assignments
                weights_star = new_weights
                neighborhood_size = 1
            else:
                neighborhood_size += 1

        classifier_star = self._train_classifier(X, assignments_star)
        return classifier_star, weights_star, assignments_star, err_star

    def __repr__(self):
        return (f"CLR_VND_TreeIterative(K={self.K}, l_max={self.l_max}, "
                f"alpha={self.alpha}, strategy={self.strategy!r})")