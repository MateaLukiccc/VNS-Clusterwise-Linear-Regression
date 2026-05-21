import warnings
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from alg import CLR_VND
from clr_kipok_wrapper import CLR_Kipok

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE  = 42
N_PER_CLASS   = 100
K_VALUES      = [2, 3, 4]
ALPHA_VALUES  = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1]
KIPOK_TRIES   = 1
KIPOK_ITER    = 5
KIPOK_METHODS = ["clrp", "largest", "weighted", "centroid",
                 "distance", "knn", "knn_weighted", "rmse_local"]
KNN_K         = 3

MU = [
    np.array([-4.0, -4.0]),
    np.array([ 0.0,  0.0]),
    np.array([ 4.0, -4.0]),
]

SIGMAS = {
    "DS1": [
        np.array([[1.0,  0.0], [0.0,  1.0]]),
        np.array([[1.0,  0.0], [0.0,  1.0]]),
        np.array([[1.0,  0.0], [0.0,  1.0]]),
    ],
    "DS2": [
        np.array([[1.0,  0.0], [0.0, 15.0]]),
        np.array([[1.0,  0.0], [0.0, 15.0]]),
        np.array([[1.0,  0.0], [0.0, 15.0]]),
    ],
    "DS3": [
        np.array([[0.01, 0.0], [0.0, 15.0]]),
        np.array([[1.0,  0.0], [0.0,  1.0]]),
        np.array([[15.0, 0.0], [0.0,  0.01]]),
    ],
    "DS4": [
        np.array([[4.30, -8.27], [-8.27, 15.89]]),
        np.array([[4.30, -8.27], [-8.27, 15.89]]),
        np.array([[4.30, -8.27], [-8.27, 15.89]]),
    ],
    "DS5": [
        np.array([[4.30, -8.27], [-8.27, 15.89]]),
        np.array([[1.0,  -1.0], [-1.0,   1.0]]),
        np.array([[15.89, 8.27], [8.27,  4.30]]),
    ],
}

DESCRIPTIONS = {
    "DS1": "Baseline — spherical equal-variance clusters",
    "DS2": "X2 has 15x higher variance (global scale difference)",
    "DS3": "Different relevant variable per class (local relevance)",
    "DS4": "Same non-zero cross-covariance for all classes (correlated)",
    "DS5": "Different covariance per class + cross-covariance (hardest)",
}


# ── Data generation ───────────────────────────────────────────────────────────
def generate_dataset(sigmas, n_per_class=N_PER_CLASS, seed=RANDOM_STATE):
    rng  = np.random.RandomState(seed)
    coef = [
        np.array([ 1.0,  1.0, -1.0]),
        np.array([ 1.0, -1.0,  1.0]),
        np.array([-1.0,  1.0,  1.0]),
    ]
    X_parts, y_parts = [], []
    for k in range(3):
        Xk = rng.multivariate_normal(MU[k], sigmas[k], size=n_per_class)
        yk = (coef[k][0] + coef[k][1]*Xk[:, 0] + coef[k][2]*Xk[:, 1]
              + rng.randn(n_per_class))
        X_parts.append(Xk)
        y_parts.append(yk)
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    X -= X.min(axis=0)
    mx = X.max(axis=0); mx[mx == 0] = 1.0
    X  = X / mx * 2.0 - 1.0
    return X, y


def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return rmse, r2


# ── Pretty printing ───────────────────────────────────────────────────────────
def print_comparison(results: dict, ds_name: str) -> None:
    W = 72
    print(f"\n{'═'*W}")
    print(f"  {ds_name}: {DESCRIPTIONS[ds_name]}")
    print(f"{'═'*W}")
    print(f"  {'Method':<40}  {'RMSE':>8}  {'R²':>8}")
    print(f"  {'-'*60}")

    # MLR baseline
    rm, r2 = results['MLR']
    print(f"  {'MLR (baseline)':<40}  {rm:>8.4f}  {r2:>8.4f}")

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
              f" — sorted by RMSE")
        print(f"  {'inference':<20}  {'RMSE':>8}  {'R²':>8}")
        for method, rm, r2 in kipok_rows:
            marker = " ◄ best" if method == kipok_rows[0][0] else ""
            print(f"  {method:<20}  {rm:>8.4f}  {r2:>8.4f}{marker}")

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
        print(f"  CLR_VND  K={K}  (sorted by RMSE)")
        print(f"  {'alpha':<20}  {'RMSE':>8}  {'R²':>8}")
        for alpha, rm, r2 in vnd_rows:
            marker = " ◄ best" if alpha == vnd_rows[0][0] else ""
            print(f"  {alpha:<20.2f}  {rm:>8.4f}  {r2:>8.4f}{marker}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    summary_rows = []

    for ds_name, sigmas in SIGMAS.items():
        X, y = generate_dataset(sigmas)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE
        )
        results = {}

        # ── MLR baseline ──────────────────────────────────────────────
        mlr = LinearRegression().fit(X_tr, y_tr)
        results['MLR'] = metrics(y_te, mlr.predict(X_te))

        # ── CLR_Kipok — fit once per K, evaluate all inference methods ─
        for K in K_VALUES:
            print(f"\n[{ds_name}] Fitting CLR_Kipok "
                  f"(K={K}, {KIPOK_TRIES} restarts, {KIPOK_ITER} iters)…",
                  flush=True)
            kipok = CLR_Kipok(
                K=K, num_tries=KIPOK_TRIES, max_iter=KIPOK_ITER,
                random_state=RANDOM_STATE,
            ).fit(X_tr, y_tr)
            print(f"         cluster sizes: {kipok.cluster_sizes()}", flush=True)

            for method in KIPOK_METHODS:
                preds = kipok.predict(X_te, method=method, K_neighbors=KNN_K)
                key   = f"CLR_Kipok/K={K}/{method}"
                results[key] = metrics(y_te, preds)
                rm, r2 = results[key]
                print(f"         {method:<16} RMSE={rm:.4f}  R²={r2:.4f}",
                      flush=True)

        # ── CLR_VND ───────────────────────────────────────────────────
        total = len(K_VALUES) * len(ALPHA_VALUES)
        done  = 0
        for K in K_VALUES:
            for alpha in ALPHA_VALUES:
                done += 1
                print(f"  [{done:>2}/{total}] CLR_VND K={K}  α={alpha} …",
                      flush=True)
                try:
                    vnd = CLR_VND(
                        K=K, l_max=1, alpha=alpha,
                        strategy="first", random_state=RANDOM_STATE,
                    ).fit(X_tr, y_tr)
                    results[(K, alpha)] = metrics(y_te, vnd.predict(X_te))
                except Exception as e:
                    print(f"    FAILED: {e}")

        print_comparison(results, ds_name)

        # ── Collect for grand summary ──────────────────────────────────
        summary_rows.append((ds_name, 'MLR', *results['MLR']))

        # All Kipok methods, all K values
        for K in K_VALUES:
            for method in KIPOK_METHODS:
                key = f"CLR_Kipok/K={K}/{method}"
                if key in results:
                    summary_rows.append(
                        (ds_name, f"Kipok K={K}/{method}", *results[key])
                    )

        # Best CLR_VND per K
        for K in K_VALUES:
            vnd_candidates = [
                (alpha, *results[(K, alpha)])
                for alpha in ALPHA_VALUES
                if (K, alpha) in results
            ]
            if vnd_candidates:
                best = min(vnd_candidates, key=lambda r: r[1])
                summary_rows.append(
                    (ds_name, f"CLR_VND K={K} α={best[0]:.2f}",
                     best[1], best[2])
                )

    # ── Grand summary ─────────────────────────────────────────────────────────
    W = 82
    print(f"\n\n{'═'*W}")
    print("  GRAND SUMMARY  —  all methods per dataset")
    print(f"{'═'*W}")
    print(f"  {'Dataset':<6}  {'Method':<36}  {'RMSE':>8}  {'R²':>8}")
    print(f"  {'-'*76}")
    prev_ds = None
    for ds, method, rm, r2 in summary_rows:
        if ds != prev_ds and prev_ds is not None:
            print()
        prev_ds = ds
        print(f"  {ds:<6}  {method:<36}  {rm:>8.4f}  {r2:>8.4f}")
    print(f"{'═'*W}\n")