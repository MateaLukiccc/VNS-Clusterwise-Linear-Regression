"""
Per-station evaluation of CLR_VND prediction methods on SILO weather data.
Matches the methodology of Bagirov, Mahmood & Barton (2017):
  - One model trained per station
  - 1392 train / 120 test split (last 120 records = test)
  - Negative predictions clipped to 0
  - Metrics: RMSE, MAE, MASE, CE  (same as paper)
  - Also loops over K, alpha, l_max as extensions

Target : rainfall_mm
Features: TMax, TMin, Evap, VP, Rad
"""

import os
import sys
import threading
import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ── Tee: write to stdout AND log file simultaneously ──────────
class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


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
from clr_pred_methods import (
    predict_largest_cluster,
    predict_simple_weighting,
    predict_knn,
    predict_local_weighting,
    predict_distance,
    predict_rmse_local,
    predict_cluster_centers,
)

# ── Config ────────────────────────────────────────────────────
RANDOM_STATE = 42
K_NEIGHBORS  = 5
K_VALUES     = [2, 3, 4]
LMAX_VALUES  = [1, 2]
ALPHA_VALUES = [1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5,
                0.45, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]

SILO_DIR     = os.path.join(os.path.dirname(__file__), "silo_data")
FEATURE_COLS = ["TMax", "TMin", "Evap", "VP", "Rad"]
TARGET_COL   = "rainfall_mm"

# Paper uses last 120 records as test set
N_TEST = 120


# ── Metrics (matching the paper exactly) ──────────────────────
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


def mase(y_true, y_pred):
    """Mean Absolute Scaled Error — scaled by naive in-sample MAE."""
    naive_mae = np.mean(np.abs(np.diff(y_true)))
    if naive_mae == 0:
        return np.nan
    return mae(y_true, y_pred) / naive_mae


def ce(y_true, y_pred):
    """Nash-Sutcliffe Coefficient of Efficiency."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def all_metrics(y_true, y_pred):
    """Returns (RMSE, MAE, MASE, CE) tuple."""
    return rmse(y_true, y_pred), mae(y_true, y_pred), \
           mase(y_true, y_pred), ce(y_true, y_pred)


def clip_negative(y_pred):
    """Paper: all negative predictions replaced by 0."""
    return np.maximum(y_pred, 0.0)


# ── Data loading ──────────────────────────────────────────────
def load_stations(silo_dir=SILO_DIR):
    """
    Returns a dict: station_name -> (X, y) where the split is
    chronological: first N-120 rows = train, last 120 = test.
    Normalisation is fitted on train only, applied to both.
    """
    stations = {}
    for fname in sorted(os.listdir(silo_dir)):
        if not fname.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(silo_dir, fname))
        df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

        # chronological order — paper uses last 120 as test
        X_all = df[FEATURE_COLS].values.astype(float)
        y_all = df[TARGET_COL].values.astype(float)

        X_train_raw = X_all[:-N_TEST]
        X_test_raw  = X_all[-N_TEST:]
        y_train     = y_all[:-N_TEST]
        y_test      = y_all[-N_TEST:]

        # normalise using train statistics only
        X_min  = X_train_raw.min(axis=0, keepdims=True)
        X_train_shifted = X_train_raw - X_min
        X_test_shifted  = X_test_raw  - X_min
        mx = X_train_shifted.max(axis=0, keepdims=True)
        mx[mx == 0] = 1.0
        X_train = X_train_shifted / mx * 2.0 - 1.0
        X_test  = X_test_shifted  / mx * 2.0 - 1.0

        name = fname.replace(".csv", "")
        stations[name] = (X_train, y_train, X_test, y_test)
    return stations


# ── Printing helpers ──────────────────────────────────────────
METRIC_HDR = f"{'RMSE':>8} {'MAE':>8} {'MASE':>6} {'CE':>7}"
METRIC_SEP = "-" * (8 + 8 + 6 + 7 + 3 * 2)


def fmt_metrics(r, m, ms, c):
    return f"{r:>8.3f} {m:>8.3f} {ms:>6.3f} {c:>7.3f}"


def log_iteration(station, l_max, K, alpha, rows, baseline_row):
    """Print full results for one (station, lmax, K, alpha) combo immediately."""
    hdr = (f"    {'Method':<28} | {'RMSE':>8} {'MAE':>8} {'MASE':>6} {'CE':>7}")
    sep = f"    {'-' * (len(hdr) - 4)}"
    bname, *bm = baseline_row

    print(f"\n  ┌─ {station}  lmax={l_max}  K={K}  α={alpha} "
          f"{'─' * max(0, 60 - len(station))}┐")
    print(hdr)
    print(sep)
    print(f"    {'MLR baseline':<28} | {fmt_metrics(*bm)}   ← paper baseline")
    print(sep)

    rows_sorted = sorted(rows, key=lambda r: r[1])   # sort by RMSE
    for rank, (name, *m) in enumerate(rows_sorted, 1):
        marker = " ←best" if rank == 1 else ""
        print(f"    {rank}. {name:<26} | {fmt_metrics(*m)}{marker}")
    print(f"  └{'─' * 68}┘")


def print_station_table(station, all_rows, baseline_row, sort_by="rmse"):
    sort_idx = {"rmse": 0, "mae": 1, "mase": 2, "ce": 3}[sort_by]
    reverse  = sort_by == "ce"

    print(f"\n  {'─'*68}")
    print(f"  Station: {station}   (sort by {sort_by.upper()})")
    print(f"  {'─'*68}")
    hdr = f"  {'Rank':<5} {'lmax':<6} {'K':<4} {'alpha':<7} {'Method':<28} | {METRIC_HDR}"
    print(hdr)
    print(f"  {'-'*len(hdr)}")

    # baseline
    bname, *bm = baseline_row
    print(f"  {'—':<5} {'1':<6} {'1':<4} {'—':<7} {bname:<28} | {fmt_metrics(*bm)}")

    # collect all rows for this station
    flat = []
    for (K, alpha, l_max), rows in all_rows.items():
        for name, *m in rows:
            flat.append((l_max, K, alpha, name, *m))

    flat.sort(key=lambda r: r[4 + sort_idx], reverse=reverse)

    for rank, (l_max, K, alpha, name, *m) in enumerate(flat[:20], 1):
        print(f"  {rank:<5} {l_max:<6} {K:<4} {alpha:<7} {name:<28} | {fmt_metrics(*m)}")


def print_summary_table(results_by_station, sort_by="rmse"):
    """Best method per (station, l_max, K, alpha)."""
    sort_idx = {"rmse": 0, "mae": 1, "mase": 2, "ce": 3}[sort_by]
    pick     = min if sort_by != "ce" else max
    key_fn   = (lambda r: r[sort_idx]) if sort_by != "ce" \
               else (lambda r: r[sort_idx])

    print(f"\n{'='*90}")
    print(f"  SUMMARY — best method per (station, lmax, K, α)   sort={sort_by.upper()}")
    print(f"{'='*90}")
    hdr = (f"  {'Station':<18} {'lmax':<6} {'K':<4} {'alpha':<7} "
           f"{'Best method':<28} | {METRIC_HDR}")
    print(hdr)
    print(f"  {'-'*len(hdr)}")

    for station, (baseline_row, all_rows) in results_by_station.items():
        first = True
        for (K, alpha, l_max), rows in sorted(all_rows.items()):
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
        print(f"ERROR: no CSV files found in {SILO_DIR}")
        return

    print(f"Stations loaded: {list(stations.keys())}")
    print(f"Test set: last {N_TEST} records per station (chronological, matching paper)")
    print(f"Features: {FEATURE_COLS}")

    results_by_station = {}   # station -> (baseline_row, all_rows)

    total_combos = len(LMAX_VALUES) * len(K_VALUES) * len(ALPHA_VALUES)

    for station, (X_train, y_train, X_test, y_test) in stations.items():
        print(f"\n{'#'*72}")
        print(f"  {station}  —  train={len(y_train)}  test={len(y_test)}")
        print(f"{'#'*72}")

        # MLR baseline (paper comparison)
        mlr       = LinearRegression().fit(X_train, y_train)
        mlr_pred  = clip_negative(mlr.predict(X_test))
        baseline_row = ("MLR baseline", *all_metrics(y_test, mlr_pred))

        all_rows = {}   # (K, alpha, l_max) -> [(name, rmse, mae, mase, ce), ...]
        done = 0

        for l_max in LMAX_VALUES:
            for K in K_VALUES:
                for alpha in ALPHA_VALUES:
                    done += 1
                    print(f"\n  [{done:>2}/{total_combos}] station={station} "
                          f"lmax={l_max} K={K} α={alpha} — fitting ...",
                          flush=True)
                    try:
                        model = CLR_VND(K=K, l_max=l_max, alpha=alpha,
                                        strategy="first",
                                        random_state=RANDOM_STATE)
                        model.fit(X_train, y_train)
                    except Exception as e:
                        print(f"FAILED: {e}")
                        continue

                    methods = [
                        ("1. Largest cluster",
                         predict_largest_cluster(model, X_test)),
                        ("2. Simple weighting",
                         predict_simple_weighting(model, X_test)),
                        (f"3. KNN majority (k={K_NEIGHBORS})",
                         predict_knn(model, X_train, X_test, K_NEIGHBORS)),
                        (f"4. KNN local wt (k={K_NEIGHBORS})",
                         predict_local_weighting(model, X_train, X_test,
                                                 K_NEIGHBORS)),
                        ("5. Distance inv-sq",
                         predict_distance(model, X_train, X_test)),
                        (f"6. RMSE-local (k={K_NEIGHBORS})",
                         predict_rmse_local(model, X_train, y_train,
                                            X_test, K_NEIGHBORS)),
                        ("7. Cluster centers",
                         predict_cluster_centers(model, X_train, X_test)),
                        ("★ Logistic clf",
                         model.predict(X_test)),
                    ]

                    rows = []
                    for name, raw_pred in methods:
                        pred = clip_negative(raw_pred)   # paper: clip negatives
                        rows.append((name, *all_metrics(y_test, pred)))

                    all_rows[(K, alpha, l_max)] = rows

                    # ── log results immediately after this combo ──
                    log_iteration(station, l_max, K, alpha, rows, baseline_row)

        results_by_station[station] = (baseline_row, all_rows)
        print_station_table(station, all_rows, baseline_row, sort_by="rmse")

    # global summary tables
    for sort_by in ["rmse", "mae", "ce"]:
        print_summary_table(results_by_station, sort_by=sort_by)

    print(f"\nRun finished: {datetime.datetime.now().isoformat()}")
    print(f"Log saved to: {log_path}")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file.close()


if __name__ == "__main__":
    sys.setrecursionlimit(15000)
    threading.stack_size(1 << 27)   # 128 MB stack for deep VND recursion
    t = threading.Thread(target=_main)
    t.start()
    t.join()