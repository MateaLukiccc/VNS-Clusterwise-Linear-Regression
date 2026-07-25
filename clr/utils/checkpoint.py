import json
import pathlib
import numpy as np


class CheckpointManager:
    """
    Persist results incrementally so crashes don't lose progress.

    Storage layout (all keys are strings):
      "{ds}||{tag}||fold_{i}"  → {"rmse": float, "r2": float}   (per CV fold)
      "{ds}||{tag}||final"     → {"rmse": float, "r2": float}   (held-out test)

    A tag is considered fully done only when its "final" key exists.
    Individual fold keys survive crashes, so a resumed run skips
    completed folds and only reruns the ones that were lost.

    Atomic writes via temp-file rename.
    """

    def __init__(self, path="results_checkpoint.json"):
        self.path = pathlib.Path(path)
        self.data = self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path) as f:
                    raw = json.load(f)
                print(f"[checkpoint] Loaded {len(raw)} entries from {self.path}")
                return raw
            except Exception as e:
                print(f"[checkpoint] Could not load ({e}), starting fresh")
        return {}

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
        tmp.replace(self.path)

    # ── low-level ─────────────────────────────────────────────────────────────
    def _fold_key(self, ds_name, tag, fold_idx):
        return f"{ds_name}||{tag}||fold_{fold_idx}"

    def _final_key(self, ds_name, tag):
        return f"{ds_name}||{tag}||final"

    # ── fold-level API ────────────────────────────────────────────────────────
    def has_fold(self, ds_name, tag, fold_idx):
        return self._fold_key(ds_name, tag, fold_idx) in self.data

    def save_fold(self, ds_name, tag, fold_idx, rmse, r2):
        self.data[self._fold_key(ds_name, tag, fold_idx)] = {
            "rmse": float(rmse), "r2": float(r2)
        }
        self._save()

    def get_fold(self, ds_name, tag, fold_idx):
        entry = self.data[self._fold_key(ds_name, tag, fold_idx)]
        return entry["rmse"], entry["r2"]

    # ── final-level API ───────────────────────────────────────────────────────
    def is_done(self, ds_name, tag):
        """True only when the final held-out result has been saved."""
        return self._final_key(ds_name, tag) in self.data

    def save_final(self, ds_name, tag, rmse, r2):
        self.data[self._final_key(ds_name, tag)] = {
            "rmse": float(rmse), "r2": float(r2)
        }
        self._save()

    def get_final(self, ds_name, tag):
        entry = self.data[self._final_key(ds_name, tag)]
        return entry["rmse"], entry["r2"]

    # ── aggregate CV stats from saved folds ───────────────────────────────────
    def aggregate_cv(self, ds_name, tag, n_splits=5):
        """Re-derive (mean_rmse, std_rmse, mean_r2, std_r2) from stored folds."""
        rmses, r2s = [], []
        for i in range(n_splits):
            if self.has_fold(ds_name, tag, i):
                rm, r2 = self.get_fold(ds_name, tag, i)
                if not (np.isnan(rm) or np.isnan(r2)):
                    rmses.append(rm)
                    r2s.append(r2)
        if not rmses:
            return np.nan, np.nan, np.nan, np.nan
        return (float(np.mean(rmses)), float(np.std(rmses, ddof=0)),
                float(np.mean(r2s)),   float(np.std(r2s,   ddof=0)))
