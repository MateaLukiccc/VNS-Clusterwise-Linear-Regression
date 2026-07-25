import numpy as np

def _linear_predict(x, weights):
    """Predict for a single row x using weight vector w = [intercept, coef...]."""
    return weights[0] + np.dot(weights[1:], x)

def _predict_all_clusters(X, weights):
    """Returns (N, K) matrix: prediction of every cluster for every point."""
    X_ext = np.column_stack([np.ones(X.shape[0]), X])
    return X_ext @ weights.T

def predict_largest_cluster(model, X_test):
    """Use the regression of the cluster with the most training points."""
    sizes = np.bincount(model.assignments_, minlength=model.K)
    dominant_cluster = int(np.argmax(sizes))
    W_q = model.weights_[dominant_cluster]
    X_ext = np.column_stack([np.ones(X_test.shape[0]), X_test])
    return X_ext @ W_q

def predict_simple_weighting(model, X_test):
    """Weighted average of all clusters, weights = cluster size / total."""
    m = len(model.assignments_)
    sizes = np.bincount(model.assignments_, minlength=model.K)
    cluster_weights = sizes / m
    all_preds = _predict_all_clusters(X_test, model.weights_)
    return all_preds @ cluster_weights

def predict_knn(model, X_train, X_test, K_neighbors=5):
    """For each test point find K nearest training neighbors, take majority cluster."""
    preds = np.empty(X_test.shape[0])
    for i, x in enumerate(X_test):
        dists = np.linalg.norm(X_train - x, axis=1)
        nn_idx = np.argsort(dists)[:K_neighbors]
        nn_clusters = model.assignments_[nn_idx]
        counts = np.bincount(nn_clusters, minlength=model.K)
        best_cluster = int(np.argmax(counts))
        preds[i] = _linear_predict(x, model.weights_[best_cluster])
    return preds

def predict_local_weighting(model, X_train, X_test, K_neighbors=5):
    """K-NN but instead of majority vote, weighted average."""
    preds = np.empty(X_test.shape[0])
    for i, x in enumerate(X_test):
        dists = np.linalg.norm(X_train - x, axis=1)
        nn_idx = np.argsort(dists)[:K_neighbors]
        nn_clusters = model.assignments_[nn_idx]
        counts = np.bincount(nn_clusters, minlength=model.K)
        cluster_weights = counts / K_neighbors
        cluster_preds = np.array([_linear_predict(x, model.weights_[k]) for k in range(model.K)])
        preds[i] = np.dot(cluster_weights, cluster_preds)
    return preds

def predict_distance(model, X_train, X_test):
    """Weights = inverse squared distance to cluster centroids."""
    centers = np.array([
        X_train[model.assignments_ == k].mean(axis=0) for k in range(model.K)
    ])
    preds = np.empty(X_test.shape[0])
    for i, x in enumerate(X_test):
        dists = np.linalg.norm(centers - x, axis=1)
        dists = np.maximum(dists, 1e-10)
        inv_sq = 1.0 / dists**2
        cluster_weights = inv_sq / inv_sq.sum()
        cluster_preds = np.array([_linear_predict(x, model.weights_[k]) for k in range(model.K)])
        preds[i] = np.dot(cluster_weights, cluster_preds)
    return preds

def predict_rmse_local(model, X_train, _, X_test, K_neighbors=5):
    """RMSE between test point and each neighbor used as distance."""
    preds = np.empty(X_test.shape[0])
    for i, x in enumerate(X_test):
        dists = np.linalg.norm(X_train - x, axis=1)
        nn_idx = np.argsort(dists)[:K_neighbors]
        nn_clusters = model.assignments_[nn_idx]
        rmse_h = dists[nn_idx]
        r_bar = rmse_h.sum()
        cluster_weights = np.zeros(model.K)
        
        if r_bar == 0:
            counts = np.bincount(nn_clusters, minlength=model.K)
            cluster_weights = counts / K_neighbors
        else:
            denom = (K_neighbors - 1) * r_bar
            for h, k in enumerate(nn_clusters):
                cluster_weights[k] += (r_bar - rmse_h[h]) / denom
            cluster_weights = np.maximum(cluster_weights, 0.0)
            s = cluster_weights.sum()
            cluster_weights = cluster_weights / s if s > 0 else np.ones(model.K) / model.K
            
        cluster_preds = np.array([_linear_predict(x, model.weights_[k]) for k in range(model.K)])
        preds[i] = np.dot(cluster_weights, cluster_preds)
    return preds

def predict_cluster_centers(model, X_train, X_test):
    """Assign test point to nearest cluster center, use its regression."""
    centers = np.array([
        X_train[model.assignments_ == k].mean(axis=0) for k in range(model.K)
    ])
    preds = np.empty(X_test.shape[0])
    for i, x in enumerate(X_test):
        dists = np.linalg.norm(centers - x, axis=1)
        best_cluster = int(np.argmin(dists))
        preds[i] = _linear_predict(x, model.weights_[best_cluster])
    return preds
