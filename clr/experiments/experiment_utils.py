import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from clr.models import CLR_Kipok, CLR_VND_TreeIterative

# ── Synthetic Data Config ─────────────────────────────────────────────────────
MU = [
    np.array([-4.0, -4.0]),
    np.array([ 0.0,  0.0]),
    np.array([ 4.0, -4.0]),
]

SIGMAS = {
    "DS1": [np.array([[1.0, 0.0], [0.0, 1.0]]) for _ in range(3)],
    "DS2": [np.array([[1.0, 0.0], [0.0, 15.0]]) for _ in range(3)],
    "DS3": [
        np.array([[0.01, 0.0], [0.0, 15.0]]),
        np.array([[1.0,  0.0], [0.0, 1.0]]),
        np.array([[15.0, 0.0], [0.0, 0.01]]),
    ],
    "DS4": [np.array([[4.30, -8.27], [-8.27, 15.89]]) for _ in range(3)],
    "DS5": [
        np.array([[4.30, -8.27], [-8.27, 15.89]]),
        np.array([[1.0,  -1.0],  [-1.0,   1.0]]),
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

_CLUSTER_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
_CLUSTER_BG     = ["#cfe2f3", "#fde8c8", "#ceecd3", "#fad4d4"]

def generate_dataset(sigmas, n_per_class=100, seed=42):
    rng  = np.random.default_rng(seed)
    coef = [
        np.array([ 1.0,  1.0, -1.0]),
        np.array([ 1.0, -1.0,  1.0]),
        np.array([-1.0,  1.0,  1.0]),
    ]
    X_parts, y_parts = [], []
    for k in range(3):
        Xk = rng.multivariate_normal(MU[k], sigmas[k], size=n_per_class)
        yk = (coef[k][0] + coef[k][1]*Xk[:, 0] + coef[k][2]*Xk[:, 1] + rng.randn(n_per_class))
        X_parts.append(Xk)
        y_parts.append(yk)
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    X -= X.min(axis=0)
    mx = X.max(axis=0); mx[mx == 0] = 1.0
    X  = X / mx * 2.0 - 1.0
    return X, y

def calc_metrics(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)), r2_score(y_true, y_pred)

def cross_validate(fit_fn, predict_fn, X, y, n_splits=5, random_state=42, ckpt=None, ds_name=None, tag=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rmses, r2s = [], []
    
    for fold_idx, (tr, te) in enumerate(kf.split(X)):
        if ckpt and ds_name and tag and ckpt.has_fold(ds_name, tag, fold_idx):
            rm, r2 = ckpt.get_fold(ds_name, tag, fold_idx)
            print(f"      [CV fold {fold_idx}] cached  RMSE={rm:.4f}  R²={r2:.4f}", flush=True)
        else:
            try:
                m = fit_fn(X[tr], y[tr])
                preds = predict_fn(m, X[te])
                rm, r2 = calc_metrics(y[te], preds)
                if ckpt and ds_name and tag:
                    ckpt.save_fold(ds_name, tag, fold_idx, rm, r2)
                print(f"      [CV fold {fold_idx}] done    RMSE={rm:.4f}  R²={r2:.4f}", flush=True)
            except Exception as e:
                print(f"      [CV fold {fold_idx}] skipped — {e}", flush=True)
                continue
        rmses.append(rm); r2s.append(r2)
        
    if not rmses:
        return np.nan, np.nan, np.nan, np.nan
    return (float(np.mean(rmses)), float(np.std(rmses, ddof=0)),
            float(np.mean(r2s)),   float(np.std(r2s,   ddof=0)))

def _cluster_label_fn(model):
    if isinstance(model, CLR_Kipok):
        return lambda Xg: model.classifier_.predict(Xg).astype(int)
    if isinstance(model, CLR_VND_TreeIterative):
        if model.classifier_ is not None:
            return lambda Xg: model.classifier_.predict(Xg).astype(int)
        dominant = int(np.argmax(np.bincount(model.assignments_, minlength=model.K)))
        return lambda Xg: np.full(Xg.shape[0], dominant, dtype=int)
    raise TypeError(f"Unsupported model type: {type(model)}")

def plot_cluster_boundaries(model, label, ds_name, X_train, save_dir="boundary_plots"):
    K, labels, pred_fn = model.K, model.assignments_, _cluster_label_fn(model)
    pad = 0.15
    x0_lo, x0_hi = X_train[:, 0].min() - pad, X_train[:, 0].max() + pad
    x1_lo, x1_hi = X_train[:, 1].min() - pad, X_train[:, 1].max() + pad
    step = max(x0_hi - x0_lo, x1_hi - x1_lo) / 350.0
    xx, yy = np.meshgrid(np.arange(x0_lo, x0_hi, step), np.arange(x1_lo, x1_hi, step))
    Z = pred_fn(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pcolormesh(xx, yy, Z, cmap=ListedColormap(_CLUSTER_BG[:K]), alpha=0.55, shading="auto", vmin=0, vmax=K - 1)
    for k in range(K):
        idx = labels == k
        ax.scatter(X_train[idx, 0], X_train[idx, 1], color=_CLUSTER_COLORS[k], s=18, edgecolors="k", linewidths=0.25, alpha=0.85, label=f"Cluster {k}")
    
    ax.set_xlim(x0_lo, x0_hi); ax.set_ylim(x1_lo, x1_hi)
    ax.set_title(f"{ds_name}  —  {label}\nK=3 cluster boundaries", fontsize=11, pad=8)
    ax.set_xlabel("X₁", fontsize=10); ax.set_ylabel("X₂", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.8)
    fig.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    fpath = os.path.join(save_dir, f"{ds_name}_{label.replace('/', '_').replace(' ', '_')}.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] → {fpath}", flush=True)

def print_grand_summary(summary_rows):
    W = 114
    print(f"\n\n{'═'*W}\n  GRAND SUMMARY  —  all methods per dataset\n{'═'*W}")
    hdr = (f"  {'Dataset':<6}  {'Method':<36}  {'Final RMSE':>10}  {'Final R²':>8}  "
           f"{'CV RMSE mean':>12}  {'CV RMSE σ':>9}  {'CV R² mean':>10}  {'CV R² σ':>8}")
    print(hdr); print(f"  {'-'*108}")
    prev_ds = None
    for row in summary_rows:
        ds, method, rm_te, r2_te = row[0], row[1], row[2], row[3]
        mr, sr, mr2, sr2 = row[4], row[5], row[6], row[7]
        if ds != prev_ds and prev_ds is not None: print()
        prev_ds = ds
        cv_rm = f"{mr:.4f}" if not np.isnan(mr) else "—"
        cv_sn = f"{sr:.4f}" if not np.isnan(sr) else "—"
        cv_r2 = f"{mr2:.4f}" if not np.isnan(mr2) else "—"
        cv_s2 = f"{sr2:.4f}" if not np.isnan(sr2) else "—"
        print(f"  {ds:<6}  {method:<36}  {rm_te:>10.4f}  {r2_te:>8.4f}  {cv_rm:>12}  {cv_sn:>9}  {cv_r2:>10}  {cv_s2:>8}")
    print(f"{'═'*W}\n")