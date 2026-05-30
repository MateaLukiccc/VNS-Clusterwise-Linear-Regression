import warnings
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression

#from alg import CLR_VND
from alg_tree import CLR_VND
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
N_FOLDS       = 5
PLOT_DIR      = "boundary_plots"

MU = [
    np.array([-4.0, -4.0]),
    np.array([ 0.0,  0.0]),
    np.array([ 4.0, -4.0]),
]

SIGMAS = {
    # "DS1": [
    #     np.array([[1.0,  0.0], [0.0,  1.0]]),
    #     np.array([[1.0,  0.0], [0.0,  1.0]]),
    #     np.array([[1.0,  0.0], [0.0,  1.0]]),
    # ],
    # "DS2": [
    #     np.array([[1.0,  0.0], [0.0, 15.0]]),
    #     np.array([[1.0,  0.0], [0.0, 15.0]]),
    #     np.array([[1.0,  0.0], [0.0, 15.0]]),
    # ],
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
    # "DS1": "Baseline — spherical equal-variance clusters",
    # "DS2": "X2 has 15x higher variance (global scale difference)",
    "DS3": "Different relevant variable per class (local relevance)",
    "DS4": "Same non-zero cross-covariance for all classes (correlated)",
    "DS5": "Different covariance per class + cross-covariance (hardest)",
}

# Consistent colour palette for K=3 boundary plots
_CLUSTER_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
_CLUSTER_BG     = ["#cfe2f3", "#fde8c8", "#ceecd3", "#fad4d4"]


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


# ── Cross-validation ──────────────────────────────────────────────────────────
def cross_validate(fit_fn, predict_fn, X, y, n_splits=N_FOLDS):
    """
    fit_fn(X_tr, y_tr)  -> fitted model
    predict_fn(m, X_te) -> prediction array

    X and y must be the TRAINING split only — the held-out test set is
    never passed here, so CV folds cannot leak test information.

    Returns (mean_rmse, std_rmse, mean_r2, std_r2).
    Folds that raise exceptions are skipped with a warning.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rmses, r2s = [], []
    for fold_idx, (tr, te) in enumerate(kf.split(X)):
        try:
            m     = fit_fn(X[tr], y[tr])
            preds = predict_fn(m, X[te])
            rm, r2 = metrics(y[te], preds)
            rmses.append(rm)
            r2s.append(r2)
        except Exception as e:
            print(f"      [CV fold {fold_idx}] skipped — {e}")
    if not rmses:
        return np.nan, np.nan, np.nan, np.nan
    return (float(np.mean(rmses)), float(np.std(rmses, ddof=0)),
            float(np.mean(r2s)),   float(np.std(r2s,   ddof=0)))


# ── Cluster boundary plots ────────────────────────────────────────────────────
def _cluster_label_fn(model):
    """Return a callable X_grid -> integer cluster labels for a fitted model."""
    if isinstance(model, CLR_Kipok):
        return lambda Xg: model.clf_.predict(Xg).astype(int)
    if isinstance(model, CLR_VND):
        if model.Theta_ is not None:
            return lambda Xg: model.Theta_.predict(Xg).astype(int)
        dominant = int(np.argmax(np.bincount(model.a_, minlength=model.K)))
        return lambda Xg: np.full(Xg.shape[0], dominant, dtype=int)
    raise TypeError(f"Unsupported model type: {type(model)}")


def plot_cluster_boundaries(model, label, ds_name, X_train, save_dir=PLOT_DIR):
    K        = model.K
    labels   = model.a_
    pred_fn  = _cluster_label_fn(model)

    pad  = 0.15
    x0   = X_train[:, 0]
    x1   = X_train[:, 1]
    x0_lo, x0_hi = x0.min() - pad, x0.max() + pad
    x1_lo, x1_hi = x1.min() - pad, x1.max() + pad

    step = max(x0_hi - x0_lo, x1_hi - x1_lo) / 350.0
    xx, yy = np.meshgrid(
        np.arange(x0_lo, x0_hi, step),
        np.arange(x1_lo, x1_hi, step),
    )
    Z = pred_fn(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    bg_cmap  = ListedColormap(_CLUSTER_BG[:K])
    dot_cols = _CLUSTER_COLORS[:K]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pcolormesh(xx, yy, Z, cmap=bg_cmap, alpha=0.55,
                  shading="auto", vmin=0, vmax=K - 1)

    for k in range(K):
        idx = labels == k
        ax.scatter(
            X_train[idx, 0], X_train[idx, 1],
            color=dot_cols[k], s=18, edgecolors="k",
            linewidths=0.25, alpha=0.85, label=f"Cluster {k}",
        )

    ax.set_xlim(x0_lo, x0_hi)
    ax.set_ylim(x1_lo, x1_hi)
    ax.set_title(f"{ds_name}  —  {label}\nK=3 cluster boundaries",
                 fontsize=11, pad=8)
    ax.set_xlabel("X₁", fontsize=10)
    ax.set_ylabel("X₂", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.8)
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    safe = label.replace("/", "_").replace(" ", "_")
    fpath = os.path.join(save_dir, f"{ds_name}_{safe}.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] → {fpath}", flush=True)


# ── Pretty printing ───────────────────────────────────────────────────────────
def _cv_str(cv_results, key):
    if key not in cv_results or any(np.isnan(cv_results[key])):
        return f"{'—':>14}", f"{'—':>14}"
    mr, sr, mr2, sr2 = cv_results[key]
    return f"{mr:.4f}±{sr:.4f}", f"{mr2:.4f}±{sr2:.4f}"


def print_comparison(results: dict, cv_results: dict, ds_name: str) -> None:
    W = 106
    print(f"\n{'═'*W}")
    print(f"  {ds_name}: {DESCRIPTIONS[ds_name]}")
    print(f"{'═'*W}")
    hdr = (f"  {'Method':<38}  {'Test RMSE':>9}  {'Test R²':>8}  "
           f"{'CV RMSE (mean±σ)':>16}  {'CV R² (mean±σ)':>16}")
    print(hdr)
    print(f"  {'-'*98}")

    rm, r2 = results['MLR']
    cr, cr2 = _cv_str(cv_results, 'MLR')
    print(f"  {'MLR (baseline)':<38}  {rm:>9.4f}  {r2:>8.4f}  {cr:>16}  {cr2:>16}")

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


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    summary_rows = []
    os.makedirs(PLOT_DIR, exist_ok=True)

    for ds_name, sigmas in SIGMAS.items():
        X, y = generate_dataset(sigmas)

        # ── Hold out 20 % as a true, never-touched test set ───────────────────
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE
        )

        results    = {}
        cv_results = {}

        # ── MLR baseline ──────────────────────────────────────────────────────
        mlr = LinearRegression().fit(X_tr, y_tr)
        results['MLR'] = metrics(y_te, mlr.predict(X_te))

        print(f"\n[{ds_name}] CV: MLR …", flush=True)
        # CV on training data only
        cv_results['MLR'] = cross_validate(
            lambda Xtr, ytr: LinearRegression().fit(Xtr, ytr),
            lambda m, Xte:   m.predict(Xte),
            X_tr, y_tr,   # <-- training split only
        )

        # ── CLR_Kipok ─────────────────────────────────────────────────────────
        for K in K_VALUES:
            print(f"\n[{ds_name}] Fitting CLR_Kipok "
                  f"(K={K}, {KIPOK_TRIES} restart, {KIPOK_ITER} iters)…",
                  flush=True)
            kipok = CLR_Kipok(
                K=K, num_tries=KIPOK_TRIES, max_iter=KIPOK_ITER,
                random_state=RANDOM_STATE,
            ).fit(X_tr, y_tr)
            print(f"         cluster sizes: {kipok.cluster_sizes()}", flush=True)

            if K == 3:
                plot_cluster_boundaries(kipok, "CLR_Kipok", ds_name, X_tr)

            for method in KIPOK_METHODS:
                key   = f"CLR_Kipok/K={K}/{method}"
                preds = kipok.predict(X_te, method=method, K_neighbors=KNN_K)
                results[key] = metrics(y_te, preds)
                rm, r2 = results[key]
                print(f"         {method:<16} RMSE={rm:.4f}  R²={r2:.4f}",
                      flush=True)

                print(f"           CV ({method}) …", flush=True)
                cv_results[key] = cross_validate(
                    lambda Xtr, ytr, _K=K: CLR_Kipok(
                        K=_K, num_tries=KIPOK_TRIES, max_iter=KIPOK_ITER,
                        random_state=RANDOM_STATE,
                    ).fit(Xtr, ytr),
                    lambda m, Xte, _met=method: m.predict(
                        Xte, method=_met, K_neighbors=KNN_K),
                    X_tr, y_tr,   # <-- training split only
                )

        # ── CLR_VND ───────────────────────────────────────────────────────────
        total = len(K_VALUES) * len(ALPHA_VALUES)
        done  = 0
        best_vnd_k3_model = None
        best_vnd_k3_rmse  = float("inf")
        best_vnd_k3_alpha = None

        for K in K_VALUES:
            for alpha in ALPHA_VALUES:
                done += 1
                print(f"  [{done:>2}/{total}] [{ds_name}] CLR_VND K={K}  α={alpha} …",
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
                        X_tr, y_tr,   # <-- training split only
                    )

                    if K == 3:
                        rm_test = results[(K, alpha)][0]
                        if rm_test < best_vnd_k3_rmse:
                            best_vnd_k3_rmse  = rm_test
                            best_vnd_k3_model = vnd
                            best_vnd_k3_alpha = alpha

                except Exception as e:
                    print(f"    FAILED: {e}")

        if best_vnd_k3_model is not None:
            plot_cluster_boundaries(
                best_vnd_k3_model,
                f"CLR_VND_a{best_vnd_k3_alpha:.2f}",
                ds_name,
                X_tr,
            )

        print_comparison(results, cv_results, ds_name)

        # ── Collect for grand summary ──────────────────────────────────────────
        summary_rows.append((ds_name, 'MLR', *results['MLR'],
                             *cv_results.get('MLR', (np.nan,)*4)))

        for K in K_VALUES:
            for method in KIPOK_METHODS:
                key = f"CLR_Kipok/K={K}/{method}"
                if key in results:
                    summary_rows.append(
                        (ds_name, f"Kipok K={K}/{method}", *results[key],
                         *cv_results.get(key, (np.nan,)*4))
                    )

        for K in K_VALUES:
            vnd_candidates = [
                (alpha, *results[(K, alpha)])
                for alpha in ALPHA_VALUES
                if (K, alpha) in results
            ]
            if vnd_candidates:
                best = min(vnd_candidates, key=lambda r: r[1])
                best_alpha = best[0]
                key = (K, best_alpha)
                summary_rows.append(
                    (ds_name, f"CLR_VND K={K} α={best_alpha:.2f}",
                     best[1], best[2],
                     *cv_results.get(key, (np.nan,)*4))
                )

    # ── Grand summary ─────────────────────────────────────────────────────────
    W = 110
    print(f"\n\n{'═'*W}")
    print("  GRAND SUMMARY  —  all methods per dataset")
    print(f"{'═'*W}")
    hdr = (f"  {'Dataset':<6}  {'Method':<36}  {'RMSE_test':>9}  {'R²_test':>8}  "
           f"{'CV RMSE mean':>12}  {'CV RMSE σ':>9}  {'CV R² mean':>10}  {'CV R² σ':>8}")
    print(hdr)
    print(f"  {'-'*104}")

    prev_ds = None
    for row in summary_rows:
        ds, method = row[0], row[1]
        rm_te, r2_te = row[2], row[3]
        mr, sr, mr2, sr2 = row[4], row[5], row[6], row[7]
        if ds != prev_ds and prev_ds is not None:
            print()
        prev_ds = ds
        cv_rm  = f"{mr:.4f}" if not np.isnan(mr)  else "—"
        cv_sn  = f"{sr:.4f}" if not np.isnan(sr)  else "—"
        cv_r2  = f"{mr2:.4f}" if not np.isnan(mr2) else "—"
        cv_s2  = f"{sr2:.4f}" if not np.isnan(sr2) else "—"
        print(f"  {ds:<6}  {method:<36}  {rm_te:>9.4f}  {r2_te:>8.4f}  "
              f"{cv_rm:>12}  {cv_sn:>9}  {cv_r2:>10}  {cv_s2:>8}")

    print(f"{'═'*W}\n")
    print(f"Boundary plots saved to: {os.path.abspath(PLOT_DIR)}/")