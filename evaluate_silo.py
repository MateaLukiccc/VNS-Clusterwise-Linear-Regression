import os
import sys
import threading
import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ── Tee logging ───────────────────────────────────────────────
class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data); s.flush()
    def flush(self):
        for s in self._streams: s.flush()

def _setup_logging():
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir  = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"silo_per_station_{ts}.log")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    print(f"Logging to : {log_path}")
    print(f"Run started: {datetime.datetime.now().isoformat()}")
    return log_file, log_path

from alg import CLR_VND
from clr_kipok_wrapper import CLR_Kipok
# from clr_pred_methods import (
#     predict_largest_cluster,
#     predict_simple_weighting,
#     predict_knn,
#     predict_local_weighting,
#     predict_distance,
#     predict_rmse_local,
#     predict_cluster_centers,
# )

# ── Config ────────────────────────────────────────────────────
RANDOM_STATE  = 42
K_NEIGHBORS   = 5
K_VALUES      = [2, 3, 4]
LMAX_VALUES   = [1, 2]
ALPHA_VALUES  = [1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5,
                 0.45, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]

# CLR_Kipok uses original paper settings (1 restart, 5 iters)
KIPOK_TRIES   = 1
KIPOK_ITER    = 5
KIPOK_METHODS = ["clrp", "largest", "weighted", "centroid",
                 "distance", "knn", "knn_weighted", "rmse_local"]

SILO_DIR      = os.path.join(os.path.dirname(__file__), "silo_data")
FEATURE_COLS  = ["TMax", "TMin", "Evap", "VP", "Rad"]
TARGET_COL    = "rainfall_mm"
N_TEST        = 120


# ── Metrics ───────────────────────────────────────────────────
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def mase(y_true, y_pred):
    naive_mae = np.mean(np.abs(np.diff(y_true)))
    return np.nan if naive_mae == 0 else mae(y_true, y_pred) / naive_mae

def ce(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

def all_metrics(y_true, y_pred):
    return rmse(y_true, y_pred), mae(y_true, y_pred), \
           mase(y_true, y_pred), ce(y_true, y_pred)

def clip_negative(y_pred):
    return np.maximum(y_pred, 0.0)


# ── Data loading ──────────────────────────────────────────────
def load_stations(silo_dir=SILO_DIR):
    stations = {}
    for fname in sorted(os.listdir(silo_dir)):
        if not fname.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(silo_dir, fname))
        df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
        X_all = df[FEATURE_COLS].values.astype(float)
        y_all = df[TARGET_COL].values.astype(float)

        X_train_raw = X_all[:-N_TEST]
        X_test_raw  = X_all[-N_TEST:]
        y_train     = y_all[:-N_TEST]
        y_test      = y_all[-N_TEST:]

        X_min = X_train_raw.min(axis=0, keepdims=True)
        X_tr  = X_train_raw - X_min
        X_te  = X_test_raw  - X_min
        mx    = X_tr.max(axis=0, keepdims=True); mx[mx == 0] = 1.0
        X_tr  = X_tr / mx * 2.0 - 1.0
        X_te  = X_te / mx * 2.0 - 1.0

        stations[fname.replace(".csv", "")] = (X_tr, y_train, X_te, y_test)
    return stations


# ── Printing helpers ──────────────────────────────────────────
METRIC_HDR = f"{'RMSE':>8} {'MAE':>8} {'MASE':>6} {'CE':>7}"

def fmt_metrics(r, m, ms, c):
    return f"{r:>8.3f} {m:>8.3f} {ms:>6.3f} {c:>7.3f}"

def _print_method_block(title, rows, baseline_row):
    """Print one sorted block of method rows under a header."""
    bname, *bm = baseline_row
    print(f"\n    ── {title} ──")
    print(f"    {'Method':<30} | {METRIC_HDR}")
    print(f"    {'-'*60}")
    print(f"    {'MLR baseline':<30} | {fmt_metrics(*bm)}")
    print(f"    {'-'*60}")
    for rank, (name, *m) in enumerate(sorted(rows, key=lambda r: r[1]), 1):
        marker = " ←best" if rank == 1 else ""
        print(f"    {rank}. {name:<28} | {fmt_metrics(*m)}{marker}")


def print_summary_table(results_by_station, sort_by="rmse"):
    sort_idx = {"rmse": 0, "mae": 1, "mase": 2, "ce": 3}[sort_by]
    reverse  = sort_by == "ce"
    pick     = max if reverse else min

    print(f"\n{'='*100}")
    print(f"  SUMMARY — best method per configuration   sort={sort_by.upper()}")
    print(f"{'='*100}")

    # ── CLR_Kipok block ───────────────────────────────────────
    print(f"\n  {'─'*96}")
    print(f"  CLR_Kipok (alt-min, {KIPOK_TRIES} restart, {KIPOK_ITER} iters)")
    print(f"  {'─'*96}")
    hdr = (f"  {'Station':<18} {'K':<4} {'Best method':<28} | {METRIC_HDR}")
    print(hdr); print(f"  {'-'*90}")

    for station, data in results_by_station.items():
        kipok_rows_by_K = data["kipok"]    # K -> [(name, rmse, mae, mase, ce)]
        first = True
        for K in K_VALUES:
            rows = kipok_rows_by_K.get(K, [])
            if not rows:
                continue
            best_name, *best_m = pick(rows, key=lambda r: r[1 + sort_idx])
            label = station if first else ""
            first = False
            print(f"  {label:<18} {K:<4} {best_name:<28} | {fmt_metrics(*best_m)}")
        print()

    # ── CLR_VND block ─────────────────────────────────────────
    print(f"\n  {'─'*96}")
    print(f"  CLR_VND (VND search)")
    print(f"  {'─'*96}")
    hdr = (f"  {'Station':<18} {'lmax':<6} {'K':<4} {'alpha':<7} "
           f"{'Best method':<28} | {METRIC_HDR}")
    print(hdr); print(f"  {'-'*90}")

    for station, data in results_by_station.items():
        vnd_rows_by_combo = data["vnd"]   # (K, alpha, l_max) -> rows
        first = True
        for (K, alpha, l_max), rows in sorted(vnd_rows_by_combo.items()):
            if not rows:
                continue
            best_name, *best_m = pick(rows, key=lambda r: r[1 + sort_idx])
            label = station if first else ""
            first = False
            print(f"  {label:<18} {l_max:<6} {K:<4} {alpha:<7} "
                  f"{best_name:<28} | {fmt_metrics(*best_m)}")
        print()


# ── Main ──────────────────────────────────────────────────────
def _main():
    log_file, log_path = _setup_logging()
    stations = load_stations()
    if not stations:
        print(f"ERROR: no CSV files found in {SILO_DIR}"); return

    print(f"Stations      : {list(stations.keys())}")
    print(f"Test set      : last {N_TEST} records (chronological)")
    print(f"Features      : {FEATURE_COLS}")
    print(f"Kipok config  : K={K_VALUES}, tries={KIPOK_TRIES}, iter={KIPOK_ITER}")
    print(f"VND config    : K={K_VALUES}, lmax={LMAX_VALUES}, "
          f"{len(ALPHA_VALUES)} alpha values")

    results_by_station = {}   # station -> {"kipok": ..., "vnd": ..., "baseline": ...}

    for station, (X_tr, y_tr, X_te, y_te) in stations.items():
        print(f"\n{'#'*72}")
        print(f"  {station}  —  train={len(y_tr)}  test={len(y_te)}")
        print(f"{'#'*72}")

        # MLR baseline
        mlr_pred     = clip_negative(LinearRegression().fit(X_tr, y_tr).predict(X_te))
        baseline_row = ("MLR baseline", *all_metrics(y_te, mlr_pred))

        kipok_rows_by_K = {}   # K -> [(name, rmse, mae, mase, ce)]
        vnd_rows_by_combo = {} # (K, alpha, l_max) -> [(name, ...)]

        # ── CLR_Kipok: one fit per K ──────────────────────────
        print(f"\n  ┌─ CLR_Kipok (standard alt-min) {'─'*38}┐")
        for K in K_VALUES:
            print(f"\n  Fitting CLR_Kipok K={K} ...", flush=True)
            try:
                kipok = CLR_Kipok(
                    K=K,
                    num_tries=KIPOK_TRIES,
                    max_iter=KIPOK_ITER,
                    random_state=RANDOM_STATE,
                ).fit(X_tr, y_tr)
                print(f"    cluster sizes: {kipok.cluster_sizes()}")
            except Exception as e:
                print(f"    FAILED: {e}"); continue

            rows = []
            for method in KIPOK_METHODS:
                raw  = kipok.predict(X_te, method=method, K_neighbors=K_NEIGHBORS)
                pred = clip_negative(raw)
                rows.append((f"kipok/{method}", *all_metrics(y_te, pred)))

            kipok_rows_by_K[K] = rows
            _print_method_block(f"CLR_Kipok K={K}", rows, baseline_row)
        print(f"  └{'─'*68}┘")

        # ── CLR_VND: (l_max, K, alpha) grid ──────────────────
        total = len(LMAX_VALUES) * len(K_VALUES) * len(ALPHA_VALUES)
        done  = 0
        print(f"\n  ┌─ CLR_VND (VND search) {'─'*46}┐")
        for l_max in LMAX_VALUES:
            for K in K_VALUES:
                for alpha in ALPHA_VALUES:
                    done += 1
                    print(f"\n  [{done:>3}/{total}] lmax={l_max} K={K} α={alpha} "
                          f"— fitting ...", flush=True)
                    try:
                        model = CLR_VND(
                            K=K, l_max=l_max, alpha=alpha,
                            strategy="first", random_state=RANDOM_STATE,
                        ).fit(X_tr, y_tr)
                    except Exception as e:
                        print(f"    FAILED: {e}"); continue

                    rows = [
                        # ("1. Largest cluster",
                        #  predict_largest_cluster(model, X_te)),
                        # ("2. Simple weighting",
                        #  predict_simple_weighting(model, X_te)),
                        # (f"3. KNN majority k={K_NEIGHBORS}",
                        #  predict_knn(model, X_tr, X_te, K_NEIGHBORS)),
                        # (f"4. KNN local wt k={K_NEIGHBORS}",
                        #  predict_local_weighting(model, X_tr, X_te, K_NEIGHBORS)),
                        # ("5. Distance inv-sq",
                        #  predict_distance(model, X_tr, X_te)),
                        # (f"6. RMSE-local k={K_NEIGHBORS}",
                        #  predict_rmse_local(model, X_tr, y_tr, X_te, K_NEIGHBORS)),
                        # ("7. Cluster centers",
                        #  predict_cluster_centers(model, X_tr, X_te)),
                        ("★ Logistic clf",
                         model.predict(X_te)),
                    ]
                    clipped_rows = [
                        (name, *all_metrics(y_te, clip_negative(pred)))
                        for name, pred in rows
                    ]
                    vnd_rows_by_combo[(K, alpha, l_max)] = clipped_rows

                    _print_method_block(
                        f"CLR_VND lmax={l_max} K={K} α={alpha}",
                        clipped_rows, baseline_row,
                    )
        print(f"  └{'─'*68}┘")

        results_by_station[station] = {
            "baseline": baseline_row,
            "kipok":    kipok_rows_by_K,
            "vnd":      vnd_rows_by_combo,
        }

    # global summary
    for sort_by in ["rmse", "mae", "ce"]:
        print_summary_table(results_by_station, sort_by=sort_by)

    print(f"\nRun finished: {datetime.datetime.now().isoformat()}")
    print(f"Log saved to: {log_path}")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file.close()


if __name__ == "__main__":
    sys.setrecursionlimit(15000)
    threading.stack_size(1 << 27)
    t = threading.Thread(target=_main)
    t.start()
    t.join()
    