"""
Five datasets differ only in covariance matrices:
  DS1 — baseline: spherical equal-variance clusters
  DS2 — X2 has 15x higher variance than X1 (global scale difference)
  DS3 — different covariance per class, no cross-covariance (local relevance)
  DS4 — same non-zero cross-covariance for all classes (correlated predictors)
  DS5 — different covariance per class + cross-covariance (hardest)
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from alg import CLR_VND

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_PER_CLASS = 100
K_TRUE = 3
K_VALUES = [3]
K_NEIGHBORS = 5
ALPHA_VALUES = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1]

MU = [
    np.array([-4.0, -4.0]),
    np.array([0.0, 0.0]),
    np.array([4.0, -4.0]),
]

SIGMAS = {
    "DS1": [
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    ],
    "DS2": [
        np.array([[1.0, 0.0], [0.0, 15.0]]),
        np.array([[1.0, 0.0], [0.0, 15.0]]),
        np.array([[1.0, 0.0], [0.0, 15.0]]),
    ],
    "DS3": [
        np.array([[0.01, 0.0], [0.0, 15.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[15.0, 0.0], [0.0, 0.01]]),
    ],
    "DS4": [
        np.array([[4.30, -8.27], [-8.27, 15.89]]),
        np.array([[4.30, -8.27], [-8.27, 15.89]]),
        np.array([[4.30, -8.27], [-8.27, 15.89]]),
    ],
    "DS5": [
        np.array([[4.30, -8.27], [-8.27, 15.89]]),
        np.array([[1.0, -1.0], [-1.0, 1.0]]),
        np.array([[15.89, 8.27], [8.27, 4.30]]),
    ],
}

DESCRIPTIONS = {
    "DS1": "Baseline — spherical equal-variance clusters",
    "DS2": "X2 has 15x higher variance (global scale difference)",
    "DS3": "Different relevant variable per class (local relevance)",
    "DS4": "Same non-zero cross-covariance for all classes (correlated)",
    "DS5": "Different covariance per class + cross-covariance (hardest)",
}


def generate_dataset(sigmas, n_per_class=N_PER_CLASS, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    coef = [
        np.array([1.0, 1.0, -1.0]),
        np.array([1.0, -1.0, 1.0]),
        np.array([-1.0, 1.0, 1.0]),
    ]
    X_parts, y_parts = [], []
    for k in range(3):
        Xk = rng.multivariate_normal(MU[k], sigmas[k], size=n_per_class)
        yk = (
            coef[k][0]
            + coef[k][1] * Xk[:, 0]
            + coef[k][2] * Xk[:, 1]
            + rng.randn(n_per_class)
        )
        X_parts.append(Xk)
        y_parts.append(yk)
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    X -= X.min(axis=0)
    mx = X.max(axis=0)
    mx[mx == 0] = 1.0
    X = X / mx * 2.0 - 1.0
    return X, y


def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2


def print_table(all_rows, baseline_row, sort_by="rmse"):
    title = "sorted by RMSE" if sort_by == "rmse" else "sorted by R²"
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

    hdr = f"{'Rank':<5} {'alpha':<7} {'Method':<32} | {'RMSE':>8} | {'R²':>8}"
    sep = "-" * len(hdr)

    name, rmse, r2 = baseline_row
    print(hdr)
    print(sep)
    print(f"{'—':<5} {'—':<7} {name:<32} | {rmse:>8.4f} | {r2:>8.4f}")

    for K in K_VALUES:
        print(f"\n  ── K={K} {'─' * 56}")
        # collect all (alpha, method_name, rmse, r2) for this K
        k_rows = []
        for alpha in ALPHA_VALUES:
            for name, rm, r2 in all_rows.get((K, alpha), []):
                k_rows.append((alpha, name, rm, r2))
        if not k_rows:
            continue
        if sort_by == "rmse":
            k_rows = sorted(k_rows, key=lambda r: r[2])
        else:
            k_rows = sorted(k_rows, key=lambda r: r[3], reverse=True)
        for rank, (alpha, name, rm, r2) in enumerate(k_rows, 1):
            print(f"  {rank:<5} {alpha:<7} {name:<32} | {rm:>8.4f} | {r2:>8.4f}")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    for ds_name, sigmas in SIGMAS.items():
        X, y = generate_dataset(sigmas)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE
        )

        # MLR baseline
        mlr = LinearRegression().fit(X_train, y_train)
        baseline_row = ("MLR baseline", *metrics(y_test, mlr.predict(X_test)))

        print(f"\n{'#' * 72}")
        print(f"  {ds_name}: {DESCRIPTIONS[ds_name]}")
        print(f"  Train: {X_train.shape[0]} pts | Test: {X_test.shape[0]} pts")
        print(f"{'#' * 72}")

        # all_rows[(K, alpha)] = list of (method_name, rmse, r2)
        all_rows = {}

        total = len(K_VALUES) * len(ALPHA_VALUES)
        done = 0
        for K in K_VALUES:
            for alpha in ALPHA_VALUES:
                done += 1
                print(f"  [{done:>2}/{total}] K={K}  α={alpha} ...", flush=True)
                try:
                    model = CLR_VND(
                        K=K,
                        l_max=1,
                        alpha=alpha,
                        strategy="first",
                        random_state=RANDOM_STATE,
                    )
                    model.fit(X_train, y_train)
                except Exception as e:
                    print(f"    FAILED: {e}")
                    continue

                methods = [
                    ("★ Logistic clf", model.predict(X_test)),
                ]

                all_rows[(K, alpha)] = [
                    (name, *metrics(y_test, preds)) for name, preds in methods
                ]

        print_table(all_rows, baseline_row, sort_by="rmse")
        print_table(all_rows, baseline_row, sort_by="r2")
