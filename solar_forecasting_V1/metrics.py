"""
Metrics for evaluating forecasting models.

This module collects simple helper functions to compute common error
metrics used in regression tasks: mean absolute error, root mean squared
error and the coefficient of determination (R²). While these metrics are
readily available in scikit‑learn, providing wrappers here simplifies
the dependency footprint when external libraries may not be available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_predictions(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute MAE, RMSE and R² for predicted values.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        A dictionary with keys ``"mae"``, ``"rmse"`` and ``"r2"``.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}