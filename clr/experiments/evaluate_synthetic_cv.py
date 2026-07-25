import warnings
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from clr.models import CLR_Kipok, CLR_VND_TreeIterative
from clr.experiments.experiment_utils import (
    SIGMAS, generate_dataset, calc_metrics, 
    cross_validate, plot_cluster_boundaries, print_grand_summary
)

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

if __name__ == "__main__":
    summary_rows = []

    for ds_name, sigmas in SIGMAS.items():
        X, y = generate_dataset(sigmas)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)
        results, cv_results = {}, {}

        # ── MLR baseline ──────────────────────────────────────────────────────
        print(f"\n[{ds_name}] MLR …", flush=True)
        mlr = LinearRegression().fit(X_tr, y_tr)
        results['MLR'] = calc_metrics(y_te, mlr.predict(X_te))
        cv_results['MLR'] = cross_validate(
            lambda Xtr, ytr: LinearRegression().fit(Xtr, ytr),
            lambda m, Xte: m.predict(Xte), X_tr, y_tr, random_state=RANDOM_STATE
        )

        # ── CLR_Kipok ─────────────────────────────────────────────────────────
        for K in K_VALUES:
            print(f"\n[{ds_name}] Fitting CLR_Kipok (K={K}) …", flush=True)
            kipok = CLR_Kipok(K=K, num_tries=KIPOK_TRIES, max_iter=KIPOK_ITER, random_state=RANDOM_STATE).fit(X_tr, y_tr)
            if K == 3: plot_cluster_boundaries(kipok, "CLR_Kipok", ds_name, X_tr)

            for method in KIPOK_METHODS:
                key = f"CLR_Kipok/K={K}/{method}"
                results[key] = calc_metrics(y_te, kipok.predict(X_te, method=method, K_neighbors=KNN_K))
                cv_results[key] = cross_validate(
                    lambda Xtr, ytr, _K=K: CLR_Kipok(K=_K, num_tries=KIPOK_TRIES, max_iter=KIPOK_ITER, random_state=RANDOM_STATE).fit(Xtr, ytr),
                    lambda m, Xte, _met=method: m.predict(Xte, method=_met, K_neighbors=KNN_K), X_tr, y_tr, random_state=RANDOM_STATE
                )

        # ── CLR_VND ───────────────────────────────────────────────────────────
        for K in K_VALUES:
            for alpha in ALPHA_VALUES:
                print(f"  [{ds_name}] CLR_VND K={K}  α={alpha} …", flush=True)
                vnd = CLR_VND_TreeIterative(K=K, l_max=1, alpha=alpha, strategy="first", random_state=RANDOM_STATE).fit(X_tr, y_tr)
                results[(K, alpha)] = calc_metrics(y_te, vnd.predict(X_te))
                cv_results[(K, alpha)] = cross_validate(
                    lambda Xtr, ytr, _K=K, _a=alpha: CLR_VND_TreeIterative(K=_K, l_max=1, alpha=_a, strategy="first", random_state=RANDOM_STATE).fit(Xtr, ytr),
                    lambda m, Xte: m.predict(Xte), X_tr, y_tr, random_state=RANDOM_STATE
                )

        # ── Aggregate Summary ──────────────────────────────────────────────────
        summary_rows.append((ds_name, 'MLR', *results['MLR'], *cv_results.get('MLR', (np.nan,)*4)))
        for K in K_VALUES:
            for method in KIPOK_METHODS:
                key = f"CLR_Kipok/K={K}/{method}"
                if key in results:
                    summary_rows.append((ds_name, f"Kipok K={K}/{method}", *results[key], *cv_results.get(key, (np.nan,)*4)))
        for K in K_VALUES:
            vnd_candidates = [(alpha, *results[(K, alpha)]) for alpha in ALPHA_VALUES if (K, alpha) in results]
            if vnd_candidates:
                best = min(vnd_candidates, key=lambda r: r[1])
                summary_rows.append((ds_name, f"CLR_VND K={K} α={best[0]:.2f}", best[1], best[2], *cv_results.get((K, best[0]), (np.nan,)*4)))

    print_grand_summary(summary_rows)