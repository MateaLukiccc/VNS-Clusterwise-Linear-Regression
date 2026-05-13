import numpy as np

def _linear_pred(x, w):
    """Predict for a single row x using weight vector w = [intercept, coef...]."""
    return w[0] + np.dot(w[1:], x)


def _predict_all_clusters(X, W):
    """Returns (N, K) matrix: prediction of every cluster for every point."""
    X_ext = np.column_stack([np.ones(X.shape[0]), X])
    return X_ext @ W.T          # shape (N, K)


def predict_largest_cluster(model, X_test):
    """Use the regression of the cluster with the most training points."""
    sizes = np.bincount(model.a_, minlength=model.K)
    q = int(np.argmax(sizes))
    W_q = model.W_[q]
    X_ext = np.column_stack([np.ones(X_test.shape[0]), X_test])
    return X_ext @ W_q


def predict_simple_weighting(model, X_test):
    """Weighted average of all clusters, weights = cluster size / total."""
    m = len(model.a_)
    sizes = np.bincount(model.a_, minlength=model.K)
    weights = sizes / m                             # shape (K,)
    Z = _predict_all_clusters(X_test, model.W_)     # (N_test, K)
    return Z @ weights


def predict_knn(model, X_train, X_test, K_neighbors=5):
    """
    For each test point find K nearest training neighbors, take majority cluster,
    use that cluster's regression.
    """
    preds = np.empty(X_test.shape[0])
    for i, x in enumerate(X_test):
        dists = np.linalg.norm(X_train - x, axis=1)
        nn_idx = np.argsort(dists)[:K_neighbors]
        nn_clusters = model.a_[nn_idx]
        counts = np.bincount(nn_clusters, minlength=model.K)
        q = int(np.argmax(counts))
        preds[i] = _linear_pred(x, model.W_[q])
    return preds


def predict_local_weighting(model, X_train, X_test, K_neighbors=5):
    """
    K-NN but instead of majority vote, weighted average where
    weight_j = (number of neighbors in cluster j) / K_neighbors.
    """
    preds = np.empty(X_test.shape[0])
    for i, x in enumerate(X_test):
        dists = np.linalg.norm(X_train - x, axis=1)
        nn_idx = np.argsort(dists)[:K_neighbors]
        nn_clusters = model.a_[nn_idx]
        counts = np.bincount(nn_clusters, minlength=model.K)
        weights = counts / K_neighbors              # shape (K,)
        z = np.array([_linear_pred(x, model.W_[k]) for k in range(model.K)])
        preds[i] = np.dot(weights, z)
    return preds


def predict_distance(model, X_train, X_test):
    """
    Weights = inverse squared distance to cluster centroids.
    """
    # Compute cluster centers from training data
    centers = np.array([
        X_train[model.a_ == k].mean(axis=0) for k in range(model.K)
    ])

    preds = np.empty(X_test.shape[0])
    for i, x in enumerate(X_test):
        dists = np.linalg.norm(centers - x, axis=1)
        # Avoid division by zero
        dists = np.maximum(dists, 1e-10)
        inv_sq = 1.0 / dists**2
        weights = inv_sq / inv_sq.sum()
        z = np.array([_linear_pred(x, model.W_[k]) for k in range(model.K)])
        preds[i] = np.dot(weights, z)
    return preds


def predict_rmse_local(model, X_train, _, X_test, K_neighbors=5):
    """
    RMSE between test point and each neighbor used as distance.
    Weight of cluster j = sum over neighbors in j of (r_bar - rmse_h) / ((K-1) * r_bar).
    Falls back to simple count ratio if r_bar == 0.
    """
    preds = np.empty(X_test.shape[0])

    for i, x in enumerate(X_test):
        dists = np.linalg.norm(X_train - x, axis=1)
        nn_idx = np.argsort(dists)[:K_neighbors]
        nn_clusters = model.a_[nn_idx]

        # rmse_h = euclidean distance (features only, as proxy, per paper)
        rmse_h = dists[nn_idx]
        r_bar = rmse_h.sum()

        weights = np.zeros(model.K)
        if r_bar == 0:
            counts = np.bincount(nn_clusters, minlength=model.K)
            weights = counts / K_neighbors
        else:
            denom = (K_neighbors - 1) * r_bar
            for h, k in enumerate(nn_clusters):
                weights[k] += (r_bar - rmse_h[h]) / denom
            # clip negatives (can happen when one rmse_h > r_bar)
            weights = np.maximum(weights, 0.0)
            s = weights.sum()
            weights = weights / s if s > 0 else np.ones(model.K) / model.K

        z = np.array([_linear_pred(x, model.W_[k]) for k in range(model.K)])
        preds[i] = np.dot(weights, z)
    return preds


def predict_cluster_centers(model, X_train, X_test):
    """Assign test point to nearest cluster center, use its regression."""
    centers = np.array([
        X_train[model.a_ == k].mean(axis=0) for k in range(model.K)
    ])
    preds = np.empty(X_test.shape[0])
    for i, x in enumerate(X_test):
        dists = np.linalg.norm(centers - x, axis=1)
        q = int(np.argmin(dists))
        preds[i] = _linear_pred(x, model.W_[q])
    return preds
