"""
Model training utilities for solar forecasting.

This module encapsulates common routines for splitting time series
datasets chronologically and training regression models. In particular,
functions for working with gradient boosting (XGBoost) are provided.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chronologically split features and target into training and test sets.

    Time series forecasting requires that the model be trained on data
    preceding the evaluation period. This function sorts the input
    samples by index and then splits them such that the first
    ``(1 - test_fraction)`` portion is used for training and the remaining
    portion for testing.

    Args:
        X: Feature matrix with a time index.
        y: Target values aligned with ``X``.
        test_fraction: Proportion of the data to reserve for testing. Must
            be between 0 and 1.

    Returns:
        A tuple ``(X_train, X_test, y_train, y_test)``.

    Raises:
        ValueError: If ``test_fraction`` is not in the interval (0, 1).
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    # Sort by index to preserve chronological order
    X_sorted = X.sort_index()
    y_sorted = y.loc[X_sorted.index]
    split_idx = int(len(X_sorted) * (1 - test_fraction))
    if split_idx <= 0 or split_idx >= len(X_sorted):
        raise ValueError(
            "Split index is out of bounds; adjust test_fraction or provide more data"
        )
    X_train = X_sorted.iloc[:split_idx]
    X_test = X_sorted.iloc[split_idx:]
    y_train = y_sorted.iloc[:split_idx]
    y_test = y_sorted.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label: str | None = None,
    params: Dict[str, Any] | None = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
    seed: int = 42,
) -> Tuple[xgb.Booster, np.ndarray, Dict[str, float]]:
    """Train an XGBoost regressor and evaluate it on a test set.

    This function wraps the ``xgboost.train`` API and returns the trained
    ``Booster`` object, the predictions on the test set and a dictionary
    of evaluation metrics (MAE, RMSE, R²). It uses early stopping to
    mitigate overfitting.

    Args:
        X_train: Training feature matrix.
        y_train: Training target values.
        X_test: Test feature matrix.
        y_test: Test target values.
        label: Optional name for logging purposes.
        params: Dictionary of XGBoost hyperparameters. Reasonable
            defaults are provided if None.
        num_boost_round: Maximum number of boosting rounds.
        early_stopping_rounds: Number of rounds with no improvement after
            which training will be stopped early.
        seed: Random seed for reproducibility.

    Returns:
        A tuple ``(booster, y_pred, metrics)`` where ``booster`` is the
        trained model, ``y_pred`` is a NumPy array of predictions on
        ``X_test`` and ``metrics`` is a dictionary containing mean
        absolute error, root mean squared error and R² score.

    Raises:
        ValueError: If training or test sets are empty.
    """
    if X_train.empty or X_test.empty:
        raise ValueError("Training and test sets must not be empty")

    # Default parameters
    default_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": seed,
    }
    # Merge user parameters if provided
    if params:
        default_params.update(params)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    watchlist = [(dtrain, "train"), (dtest, "eval")]

    booster = xgb.train(
        default_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=watchlist,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    # Generate predictions
    y_pred = booster.predict(dtest)
    # Evaluate metrics
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    if label:
        logger.info(
            "%s - MAE: %.3f | RMSE: %.3f | R2: %.3f", label, mae, rmse, r2
        )
    return booster, y_pred, metrics