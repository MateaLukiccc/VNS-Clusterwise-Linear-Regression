"""
Evaluation of CLR_VND prediction methods on auto-mpg dataset.
Uses 80/20 train/test split.
Loops over K values and alpha values.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from alg import CLR_VND
from clr_pred_methods import (
    predict_largest_cluster,
    predict_simple_weighting,
    predict_knn,
    predict_local_weighting,
    predict_distance,
    predict_rmse_local,
    predict_cluster_centers,
)

RANDOM_STATE = 42
K_NEIGHBORS  = 5
K_VALUES     = [2, 3, 4]
ALPHA_VALUES = [1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5,
                0.45, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]


def load_auto_mpg(path="auto-mpg.data"):
    data = pd.read_csv(path, header=None, sep=r"\s+", na_values="?")
    data = data.dropna()
    X = pd.get_dummies(data.iloc[:, 1:-1], columns=[7]).values.astype(float)
    y = data[0].values.astype(float)
    X -= X.min(axis=0, keepdims=True)
    mx = X.max(axis=0, keepdims=True)
    mx[mx == 0] = 1.0
    X = X / mx * 2.0 - 1.0
    return X, y


def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return rmse, r2


def print_detail_table(all_rows, baseline_row, sort_by="rmse"):
    title = "FULL DETAIL — sorted by RMSE" if sort_by == "rmse" \
            else "FULL DETAIL — sorted by R²"
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")

    hdr = f"{'Rank':<5} {'Method':<35} | {'RMSE':>8} | {'R²':>8}"
    sep = "-" * len(hdr)

    name, rmse, r2 = baseline_row
    print(hdr)
    print(sep)
    print(f"{'—':<5} {name:<35} | {rmse:>8.4f} | {r2:>8.4f}")

    for K in K_VALUES:
        print(f"\n  ── K={K} {'─' * 58}")
        for alpha in ALPHA_VALUES:
            rows = all_rows.get((K, alpha), [])
            if not rows:
                continue
            rows = sorted(rows, key=lambda r: r[1] if sort_by == "rmse"
                          else -r[2])
            print(f"\n    α={alpha}")
            print(f"    {hdr}")
            print(f"    {sep}")
            for rank, (n, rm, r2) in enumerate(rows, 1):
                print(f"    {rank:<5} {n:<35} | {rm:>8.4f} | {r2:>8.4f}")


def print_summary_table(all_rows, baseline_row, sort_by="rmse"):
    title = "SUMMARY — best method per (K, α)" + \
            (" sorted by RMSE" if sort_by == "rmse" else " sorted by R²")
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")

    hdr = f"{'K':<5} {'alpha':<7} {'Best method':<35} | {'RMSE':>8} | {'R²':>8}"
    print(hdr)
    print("-" * len(hdr))

    name, rmse, r2 = baseline_row
    print(f"{'1':<5} {'—':<7} {name:<35} | {rmse:>8.4f} | {r2:>8.4f}")
    print("-" * len(hdr))

    for K in K_VALUES:
        for alpha in ALPHA_VALUES:
            rows = all_rows.get((K, alpha), [])
            if not rows:
                continue
            best = (min(rows, key=lambda r: r[1]) if sort_by == "rmse"
                    else max(rows, key=lambda r: r[2]))
            print(f"{K:<5} {alpha:<7} {best[0]:<35} | {best[1]:>8.4f} | {best[2]:>8.4f}")


if __name__ == "__main__":
    X, y = load_auto_mpg()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]} pts | Test: {X_test.shape[0]} pts "
          f"| Features: {X.shape[1]}")

    mlr = LinearRegression().fit(X_train, y_train)
    baseline_row = ("K=1  MLR (baseline)", *metrics(y_test, mlr.predict(X_test)))

    # all_rows[(K, alpha)] = list of (method_name, rmse, r2)
    all_rows = {}

    total = len(K_VALUES) * len(ALPHA_VALUES)
    done  = 0
    for K in K_VALUES:
        for alpha in ALPHA_VALUES:
            done += 1
            print(f"  [{done:>2}/{total}] K={K}  α={alpha} ...", flush=True)
            try:
                model = CLR_VND(K=K, l_max=1, alpha=alpha, strategy="first",
                                random_state=RANDOM_STATE)
                model.fit(X_train, y_train)
            except Exception as e:
                print(f"    FAILED: {e}")
                continue

            methods = [
                ("1. Largest cluster",
                 predict_largest_cluster(model, X_test)),
                ("2. Simple weighting",
                 predict_simple_weighting(model, X_test)),
                (f"3. KNN (k={K_NEIGHBORS}) majority",
                 predict_knn(model, X_train, X_test, K_NEIGHBORS)),
                (f"4. KNN (k={K_NEIGHBORS}) local weight",
                 predict_local_weighting(model, X_train, X_test, K_NEIGHBORS)),
                ("5. Distance (inv-sq to center)",
                 predict_distance(model, X_train, X_test)),
                (f"6. RMSE-based local (k={K_NEIGHBORS})",
                 predict_rmse_local(model, X_train, y_train, X_test, K_NEIGHBORS)),
                ("7. Cluster centers",
                 predict_cluster_centers(model, X_train, X_test)),
                ("★ Logistic clf",
                 model.predict(X_test)),
            ]

            all_rows[(K, alpha)] = [
                (name, *metrics(y_test, preds)) for name, preds in methods
            ]

    print_detail_table(all_rows, baseline_row, sort_by="rmse")
    print_detail_table(all_rows, baseline_row, sort_by="r2")
    print_summary_table(all_rows, baseline_row, sort_by="rmse")
    print_summary_table(all_rows, baseline_row, sort_by="r2")