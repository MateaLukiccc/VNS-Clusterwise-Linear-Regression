import warnings
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from clr.models import CLR_Kipok, CLR_VND_TreeIterative
from clr.utils.checkpoint import CheckpointManager
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
CHECKPOINT_FILE = "results_checkpoint.json"

if __name__ == "__main__":
    summary_rows = []
    ckpt = CheckpointManager(CHECKPOINT_FILE)

    for ds_name, sigmas in SIGMAS.items():
        X, y = generate_dataset(sigmas)
        X_cv, X_te, y_cv, y_te = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)
        cv_results, final_results = {}, {}

        def run(tag, fit_fn, predict_fn):
            if ckpt.is_done(ds_name, tag):
                cv = ckpt.aggregate_cv(ds_name, tag)
                final = ckpt.get_final(ds_name, tag)
                print(f"  [skip] {ds_name} / {tag} (fully cached)", flush=True)
                return cv, final
            
            cv = cross_validate(fit_fn, predict_fn, X_cv, y_cv, random_state=RANDOM_STATE, ckpt=ckpt, ds_name=ds_name, tag=tag)
            final_model = fit_fn(X_cv, y_cv)
            rmse_f, r2_f = calc_metrics(y_te, predict_fn(final_model, X_te))
            ckpt.save_final(ds_name, tag, rmse_f, r2_f)
            return cv, (rmse_f, r2_f)

        # MLR
        print(f"\n[{ds_name}] MLR …", flush=True)
        cv_results['MLR'], final_results['MLR'] = run(
            'MLR', lambda Xtr, ytr: LinearRegression().fit(Xtr, ytr), lambda m, Xte: m.predict(Xte)
        )

        # Kipok
        for K in K_VALUES:
            print(f"\n[{ds_name}] CLR_Kipok K={K} …", flush=True)
            if K == 3:
                kipok_plot = CLR_Kipok(K=K, num_tries=KIPOK_TRIES, max_iter=KIPOK_ITER, random_state=RANDOM_STATE).fit(X_cv, y_cv)
                plot_cluster_boundaries(kipok_plot, "CLR_Kipok", ds_name, X_cv)
            for method in KIPOK_METHODS:
                tag = f"CLR_Kipok/K={K}/{method}"
                cv_results[tag], final_results[tag] = run(
                    tag,
                    lambda Xtr, ytr, _K=K: CLR_Kipok(K=_K, num_tries=KIPOK_TRIES, max_iter=KIPOK_ITER, random_state=RANDOM_STATE).fit(Xtr, ytr),
                    lambda m, Xte, _met=method: m.predict(Xte, method=_met, K_neighbors=KNN_K),
                )

        # VND
        for K in K_VALUES:
            for alpha in ALPHA_VALUES:
                tag = f"VND/K={K}/alpha={alpha:.2f}"
                print(f"  [{ds_name}] CLR_VND K={K}  α={alpha} …", flush=True)
                try:
                    cv_results[(K, alpha)], final_results[(K, alpha)] = run(
                        tag,
                        lambda Xtr, ytr, _K=K, _a=alpha: CLR_VND_TreeIterative(K=_K, l_max=1, alpha=_a, strategy="first", random_state=RANDOM_STATE).fit(Xtr, ytr),
                        lambda m, Xte: m.predict(Xte),
                    )
                except Exception as e:
                    print(f"    FAILED: {e}")

        # Aggregate Summary
        summary_rows.append((ds_name, 'MLR', *final_results['MLR'], *cv_results.get('MLR', (np.nan,)*4)))
        for K in K_VALUES:
            for method in KIPOK_METHODS:
                key = f"CLR_Kipok/K={K}/{method}"
                if key in final_results:
                    summary_rows.append((ds_name, f"Kipok K={K}/{method}", *final_results[key], *cv_results.get(key, (np.nan,)*4)))
        for K in K_VALUES:
            vnd_candidates = [(alpha, final_results[(K, alpha)]) for alpha in ALPHA_VALUES if (K, alpha) in final_results]
            if vnd_candidates:
                best_alpha, best_final = min(vnd_candidates, key=lambda r: r[1][0])
                summary_rows.append((ds_name, f"CLR_VND K={K} α={best_alpha:.2f}", *best_final, *cv_results.get((K, best_alpha), (np.nan,)*4)))

    print_grand_summary(summary_rows)
    print(f"Boundary plots saved to: {os.path.abspath('boundary_plots')}/")