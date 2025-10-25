"""
Regression metrics utilities.

This module implements common regression metrics such as mean
absolute error, root mean squared error, coefficient of determination
(R²) and normalized metrics.  It also provides functions to compute
mean absolute percentage error (MAPE), symmetric MAPE (sMAPE) and
normalized RMSE.  A ``summarize`` helper aggregates a variety of
metrics for a pair of series.

These metrics are widely used to evaluate the performance of
regression models.  When using early stopping with XGBoost, the
evaluation metric is monitored on a validation set, and training
stops when the metric fails to improve【151138559352955†L176-L186】.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the mean absolute error (MAE).

    Parameters
    ----------
    y_true, y_pred : array‑like
        Ground truth and predicted values.  These will be cast to
        floats.

    Returns
    -------
    float
        The mean absolute difference between ``y_true`` and ``y_pred``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the root mean squared error (RMSE).

    Parameters
    ----------
    y_true, y_pred : array‑like
        Ground truth and predicted values.  These will be cast to
        floats.

    Returns
    -------
    float
        The square root of the mean squared error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the coefficient of determination (R²).

    R² measures the proportion of variance in ``y_true`` that is
    explained by ``y_pred``.  A value of 1.0 indicates perfect
    prediction; values less than 0 indicate that the model performs
    worse than predicting the mean.

    Parameters
    ----------
    y_true, y_pred : array‑like
        Ground truth and predicted values.

    Returns
    -------
    float
        The R² score.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Compute mean absolute percentage error (MAPE).

    Percentage error is computed relative to the true value.  An
    epsilon term is added to the denominator to avoid division by
    zero.

    Parameters
    ----------
    y_true, y_pred : array‑like
        Ground truth and predicted values.
    eps : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    float
        MAPE expressed in percent.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Compute symmetric mean absolute percentage error (sMAPE).

    The symmetric MAPE divides the absolute error by the average of
    absolute true and predicted values.  This avoids asymmetry when
    true values are close to zero.

    Parameters
    ----------
    y_true, y_pred : array‑like
        Ground truth and predicted values.
    eps : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    float
        sMAPE expressed in percent.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(num / np.maximum(den, eps)) * 100)


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute normalized RMSE as percentages.

    Normalisation is performed by dividing the RMSE by either the
    range (max–min) or the standard deviation of the true values.

    Parameters
    ----------
    y_true, y_pred : array‑like
        Ground truth and predicted values.

    Returns
    -------
    dict
        Dictionary with keys ``'by_range_%'`` and ``'by_std_%'``.
    """
    rmse = root_mean_squared_error(y_true, y_pred)
    y_true = np.asarray(y_true, dtype=float)
    return {
        "by_range_%": float(rmse / (y_true.max() - y_true.min()) * 100),
        "by_std_%": float(rmse / (y_true.std(ddof=0) + 1e-9) * 100),
    }


def summarize(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    tag: str = "model",
    daytime_mask: np.ndarray | None = None,
) -> pd.Series:
    """Aggregate several regression metrics into a pandas Series.

    This helper computes MAE, RMSE, R², MAPE, sMAPE and normalised
    RMSE for the provided predictions.  If a boolean mask is given to
    ``daytime_mask``, metrics are also computed on the subset of
    observations where the mask is True (e.g. daytime hours).

    Parameters
    ----------
    y_true, y_pred : array‑like or pd.Series
        Ground truth and predicted values.  These are converted to
        numpy arrays internally.
    tag : str, optional
        Label for the resulting series (used as the index name).
    daytime_mask : array‑like of bool, optional
        Mask indicating a subset of observations (e.g. daytime only)
        for which additional metrics should be computed.

    Returns
    -------
    pd.Series
        Series of aggregated metrics with the provided tag as its name.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    res: Dict[str, Any] = {}
    res["MAE"] = mean_absolute_error(yt, yp)
    res["RMSE"] = root_mean_squared_error(yt, yp)
    res["R2"] = r2_score(yt, yp)
    res["MAPE_%"] = mape(yt, yp)
    res["sMAPE_%"] = smape(yt, yp)
    res.update({f"NRMSE_{k}": v for k, v in nrmse(yt, yp).items()})
    # If a mask is provided, compute metrics on the masked subset
    if daytime_mask is not None:
        mask = np.asarray(daytime_mask, dtype=bool)
        yt_d = yt[mask]
        yp_d = yp[mask]
        res["DAY_MAE"] = mean_absolute_error(yt_d, yp_d)
        res["DAY_MAPE_%"] = mape(yt_d, yp_d)
        res["DAY_sMAPE_%"] = smape(yt_d, yp_d)
    return pd.Series(res, name=tag)