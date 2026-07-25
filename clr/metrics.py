import numpy as np

def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true)))


def mase(y_true, y_pred, y_train):
    """Mean Absolute Scaled Error (Hyndman & Koehler, 2006).

    Denominator is the in-sample MAE of the naive 1-step-ahead forecast
    on the TRAINING series — so y_train must be passed separately.
    """
    y_true  = np.asarray(y_true,  dtype=float)
    y_pred  = np.asarray(y_pred,  dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) < 2:
        return float("nan")
    denom = np.mean(np.abs(np.diff(y_train)))
    if denom == 0.0:
        return float("inf")
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def ce(y_true, y_pred):
    """Nash-Sutcliffe Coefficient of Efficiency.  CE=1 perfect, CE=0 == predicting the mean,
    CE<0 worse than predicting the mean."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    den = np.sum((y_true - np.mean(y_true)) ** 2)
    if den == 0.0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / den)


def all_metrics(y_true, y_pred, y_train):
    """Returns the four-metric dict for a single (predictor, test set)."""
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE":  mae(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train),
        "CE":   ce(y_true, y_pred),
    }
