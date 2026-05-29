import warnings
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression

from alg_tree import CLR_VND
from clr_kipok_wrapper import CLR_Kipok

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE  = 42
K_VALUES      = [2, 3, 4]
ALPHA_VALUES  = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1]
KIPOK_TRIES   = 1
KIPOK_ITER    = 5
KIPOK_METHODS = ["clrp", "largest", "weighted", "centroid",
                 "distance", "knn", "knn_weighted", "rmse_local"]
KNN_K         = 3
N_FOLDS       = 5


# ── Data loading ──────────────────────────────────────────────────────────────
def load_car_data(data_path="auto-mpg.data"):
    data = pd.read_csv(data_path, header=None, sep=r"\s+", na_values="?")
    data = data.dropna()
    X = pd.get_dummies(data.iloc[:, 1:-1], columns=[7]).values.astype(float)
    y = data[0].values.astype(float)
    X -= X.min(axis=0, keepdims=True)
    max_val = X.max(axis=0, keepdims=True)
    max_val[max_val == 0] = 1.0
    X = X / max_val * 2.0 - 1.0
    return X, y


# ── Metrics ───────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return rmse, r2


# ── Cross-validation (training split only — no test leakage) ─────────────────
def cross_validate(fit_fn, predict_fn, X, y, n_splits=N_FOLDS):
    """
    X and y must be the TRAINING split only.
    Returns (mean_rmse, std_rmse, mean_r2, std_r2).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rmses, r2s = [], []
    for fold_idx, (tr, te) in enumerate(kf.split(X)):
        try:
            m      = fit_fn(X[tr], y[tr])
            preds  = predict_fn(m, X[te])
            rm, r2 = metrics(y[te], preds)
            rmses.append(rm)
            r2s.append(r2)
        except Exception as e:
            print(f"      [CV fold {fold_idx}] skipped — {e}")
    if not rmses:
        return np.nan, np.nan, np.nan, np.nan
    return (float(np.mean(rmses)), float(np.std(rmses, ddof=0)),
            float(np.mean(r2s)),   float(np.std(r2s,   ddof=0)))


# ── Pretty printing ───────────────────────────────────────────────────────────
def _cv_str(cv_results, key):
    if key not in cv_results or any(np.isnan(cv_results[key])):
        return f"{'—':>16}", f"{'—':>16}"
    mr, sr, mr2, sr2 = cv_results[key]
    return f"{mr:.4f}±{sr:.4f}", f"{mr2:.4f}±{sr2:.4f}"


def print_results(results, cv_results):
    W = 106
    print(f"\n{'═'*W}")
    print("  Auto-MPG Car Dataset")
    print(f"{'═'*W}")
    hdr = (f"  {'Method':<38}  {'Test RMSE':>9}  {'Test R²':>8}  "
           f"{'CV RMSE (mean±σ)':>16}  {'CV R² (mean±σ)':>16}")
    print(hdr)
    print(f"  {'-'*98}")

    # MLR baseline
    rm, r2 = results["MLR"]
    cr, cr2 = _cv_str(cv_results, "MLR")
    print(f"  {'MLR (baseline)':<38}  {rm:>9.4f}  {r2:>8.4f}  {cr:>16}  {cr2:>16}")

    # CLR_Kipok — one sorted block per K
    for K in K_VALUES:
        print()
        kipok_rows = sorted(
            [(m, *results[f"CLR_Kipok/K={K}/{m}"])
             for m in KIPOK_METHODS
             if f"CLR_Kipok/K={K}/{m}" in results],
            key=lambda r: r[1],
        )
        if not kipok_rows:
            continue
        print(f"  CLR_Kipok  K={K}  ({KIPOK_TRIES} restart, {KIPOK_ITER} iters)"
              f"  — sorted by test RMSE")
        print(f"  {'inference':<20}  {'Test RMSE':>9}  {'Test R²':>8}  "
              f"{'CV RMSE (mean±σ)':>16}  {'CV R² (mean±σ)':>16}")
        for method, rm, r2 in kipok_rows:
            key = f"CLR_Kipok/K={K}/{method}"
            cr, cr2 = _cv_str(cv_results, key)
            marker = " ◄ best" if method == kipok_rows[0][0] else ""
            print(f"  {method:<20}  {rm:>9.4f}  {r2:>8.4f}  {cr:>16}  {cr2:>16}{marker}")

    # CLR_VND — one sorted block per K
    for K in K_VALUES:
        print()
        vnd_rows = sorted(
            [(alpha, *results[(K, alpha)])
             for alpha in ALPHA_VALUES
             if (K, alpha) in results],
            key=lambda r: r[1],
        )
        if not vnd_rows:
            continue
        print(f"  CLR_VND  K={K}  (sorted by test RMSE)")
        print(f"  {'alpha':<20}  {'Test RMSE':>9}  {'Test R²':>8}  "
              f"{'CV RMSE (mean±σ)':>16}  {'CV R² (mean±σ)':>16}")
        for alpha, rm, r2 in vnd_rows:
            cr, cr2 = _cv_str(cv_results, (K, alpha))
            marker = " ◄ best" if alpha == vnd_rows[0][0] else ""
            print(f"  {alpha:<20.2f}  {rm:>9.4f}  {r2:>8.4f}  {cr:>16}  {cr2:>16}{marker}")

    print()

    # Grand summary — best per family per K
    print(f"\n{'═'*W}")
    print("  SUMMARY  —  best configuration per method family")
    print(f"{'═'*W}")
    hdr = (f"  {'Method':<36}  {'RMSE_test':>9}  {'R²_test':>8}  "
           f"{'CV RMSE mean':>12}  {'CV RMSE σ':>9}  {'CV R² mean':>10}  {'CV R² σ':>8}")
    print(hdr)
    print(f"  {'-'*104}")

    # MLR
    rm, r2 = results["MLR"]
    mr, sr, mr2, sr2 = cv_results.get("MLR", (np.nan,)*4)
    print(f"  {'MLR':<36}  {rm:>9.4f}  {r2:>8.4f}  "
          f"{mr:>12.4f}  {sr:>9.4f}  {mr2:>10.4f}  {sr2:>8.4f}")

    # Best Kipok per K
    for K in K_VALUES:
        candidates = [
            (m, *results[f"CLR_Kipok/K={K}/{m}"])
            for m in KIPOK_METHODS
            if f"CLR_Kipok/K={K}/{m}" in results
        ]
        if not candidates:
            continue
        best_m, best_rm, best_r2 = min(candidates, key=lambda r: r[1])
        key = f"CLR_Kipok/K={K}/{best_m}"
        mr, sr, mr2, sr2 = cv_results.get(key, (np.nan,)*4)
        label = f"Kipok K={K}/{best_m}"
        print(f"  {label:<36}  {best_rm:>9.4f}  {best_r2:>8.4f}  "
              f"{mr:>12.4f}  {sr:>9.4f}  {mr2:>10.4f}  {sr2:>8.4f}")

    # Best VND per K
    for K in K_VALUES:
        candidates = [
            (alpha, *results[(K, alpha)])
            for alpha in ALPHA_VALUES
            if (K, alpha) in results
        ]
        if not candidates:
            continue
        best_alpha, best_rm, best_r2 = min(candidates, key=lambda r: r[1])
        key = (K, best_alpha)
        mr, sr, mr2, sr2 = cv_results.get(key, (np.nan,)*4)
        label = f"CLR_VND K={K} α={best_alpha:.2f}"
        print(f"  {label:<36}  {best_rm:>9.4f}  {best_r2:>8.4f}  "
              f"{mr:>12.4f}  {sr:>9.4f}  {mr2:>10.4f}  {sr2:>8.4f}")

    print(f"{'═'*W}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y = load_car_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    results    = {}
    cv_results = {}

    # ── MLR baseline ──────────────────────────────────────────────────────────
    mlr = LinearRegression().fit(X_tr, y_tr)
    results["MLR"] = metrics(y_te, mlr.predict(X_te))

    print("\n[Car] CV: MLR …", flush=True)
    cv_results["MLR"] = cross_validate(
        lambda Xtr, ytr: LinearRegression().fit(Xtr, ytr),
        lambda m, Xte:   m.predict(Xte),
        X_tr, y_tr,
    )

    # ── CLR_Kipok ─────────────────────────────────────────────────────────────
    for K in K_VALUES:
        print(f"\n[Car] Fitting CLR_Kipok "
              f"(K={K}, {KIPOK_TRIES} restart, {KIPOK_ITER} iters)…", flush=True)
        kipok = CLR_Kipok(
            K=K, num_tries=KIPOK_TRIES, max_iter=KIPOK_ITER,
            random_state=RANDOM_STATE,
        ).fit(X_tr, y_tr)
        print(f"         cluster sizes: {kipok.cluster_sizes()}", flush=True)

        for method in KIPOK_METHODS:
            key   = f"CLR_Kipok/K={K}/{method}"
            preds = kipok.predict(X_te, method=method, K_neighbors=KNN_K)
            results[key] = metrics(y_te, preds)
            rm, r2 = results[key]
            print(f"         {method:<16} RMSE={rm:.4f}  R²={r2:.4f}", flush=True)

            print(f"           CV ({method}) …", flush=True)
            cv_results[key] = cross_validate(
                lambda Xtr, ytr, _K=K: CLR_Kipok(
                    K=_K, num_tries=KIPOK_TRIES, max_iter=KIPOK_ITER,
                    random_state=RANDOM_STATE,
                ).fit(Xtr, ytr),
                lambda m, Xte, _met=method: m.predict(
                    Xte, method=_met, K_neighbors=KNN_K),
                X_tr, y_tr,
            )

    # ── CLR_VND ───────────────────────────────────────────────────────────────
    total = len(K_VALUES) * len(ALPHA_VALUES)
    done  = 0

    for K in K_VALUES:
        for alpha in ALPHA_VALUES:
            done += 1
            print(f"  [{done:>2}/{total}] [Car] CLR_VND K={K}  α={alpha} …",
                  flush=True)
            try:
                vnd = CLR_VND(
                    K=K, l_max=1, alpha=alpha,
                    strategy="first", random_state=RANDOM_STATE,
                ).fit(X_tr, y_tr)
                results[(K, alpha)] = metrics(y_te, vnd.predict(X_te))

                cv_results[(K, alpha)] = cross_validate(
                    lambda Xtr, ytr, _K=K, _a=alpha: CLR_VND(
                        K=_K, l_max=1, alpha=_a,
                        strategy="first", random_state=RANDOM_STATE,
                    ).fit(Xtr, ytr),
                    lambda m, Xte: m.predict(Xte),
                    X_tr, y_tr,
                )
            except Exception as e:
                print(f"    FAILED: {e}")

    print_results(results, cv_results)