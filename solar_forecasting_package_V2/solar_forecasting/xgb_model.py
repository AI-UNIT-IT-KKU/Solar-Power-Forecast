"""
Training and inference utilities for gradient boosted tree models.

This module wraps the native XGBoost interface to train regression
models on tabular data.  Functions are provided for single model
training with early stopping as well as a simple grid search over
hyperparameters.  The implementation uses XGBoost's native
``Booster`` object and its early stopping mechanism, which halts
training when the validation error fails to improve for a number of
rounds【151138559352955†L176-L190】.

Example
-------
::

    from solar_forecasting.xgb_model import train_xgb_single, grid_search_xgb

    booster, pred, metrics = train_xgb_single(
        X_train, y_train, X_val, y_val,
        label="My XGB Model"
    )
    print(metrics["rmse"])

    # Perform a simple grid search
    best_booster, best_params, results_df = grid_search_xgb(
        X_train, y_train, X_val, y_val,
        param_grid={"eta": [0.03, 0.05], "max_depth": [4, 6]},
        fixed_params={"subsample": 0.9}
    )

Notes
-----
This module depends on the ``xgboost`` package; be sure it is
installed in your environment.  Early stopping will add ``best_score``
and ``best_iteration`` attributes to the returned ``Booster`` when it
triggers【151138559352955†L186-L190】.
"""

from __future__ import annotations

import itertools
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from .metrics import mean_absolute_error, root_mean_squared_error, r2_score


def train_xgb_single(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    label: str = "XGB",
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
    seed: int = 42,
    base_params: Optional[Dict[str, Any]] = None,
) -> Tuple[xgb.Booster, np.ndarray, Dict[str, float]]:
    """Train a single XGBoost model with early stopping.

    Parameters
    ----------
    X_tr, y_tr : np.ndarray
        Training features and targets.  Shape should be (n_samples, n_features).
    X_val, y_val : np.ndarray
        Validation features and targets used for early stopping.
    label : str, optional
        Label printed when reporting metrics.
    num_boost_round : int, optional
        Maximum number of boosting iterations.  Default is 2000.
    early_stopping_rounds : int, optional
        Number of rounds without improvement before stopping early.
    seed : int, optional
        Random seed for reproducibility.
    base_params : dict, optional
        Dictionary of hyperparameters to override the defaults.

    Returns
    -------
    booster : xgboost.Booster
        Trained model.  The ``best_iteration`` attribute indicates the
        optimal boosting round.
    pred : np.ndarray
        Predictions on the validation set using the best iteration.
    metrics : dict
        Dictionary of evaluation metrics (MSE, RMSE, MAE, R² and best_iteration).
    """
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_val, label=y_val)

    # Default hyperparameters.  These can be overridden via base_params.
    params: Dict[str, Any] = {
        "objective": "reg:squarederror",  # squared error => MSE loss
        "eval_metric": "rmse",           # monitor RMSE on validation set
        "eta": 0.03,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "seed": seed,
    }
    if base_params:
        params.update(base_params)

    evals = [(dtrain, "train"), (dvalid, "valid")]
    booster: xgb.Booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    # Predictions using the best iteration
    pred = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))

    # Compute evaluation metrics
    mse = float(np.mean((y_val - pred) ** 2))
    rmse = root_mean_squared_error(y_val, pred)
    mae = mean_absolute_error(y_val, pred)
    r2 = r2_score(y_val, pred)
    metrics: Dict[str, float] = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "best_iteration": booster.best_iteration,
    }
    print(
        f"\n--- {label} ---\n"
        f"Best Iteration : {booster.best_iteration}\n"
        f"MSE            : {mse:.6f}\n"
        f"RMSE           : {rmse:.6f}\n"
        f"MAE            : {mae:.6f}\n"
        f"R2             : {r2:.6f}"
    )
    return booster, pred, metrics


def grid_search_xgb(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    label: str = "XGB",
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
    seed: int = 42,
    param_grid: Optional[Dict[str, list[Any]]] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
) -> Tuple[xgb.Booster, Dict[str, Any], pd.DataFrame]:
    """Perform a simple grid search over XGBoost hyperparameters.

    Parameters
    ----------
    X_tr, y_tr, X_val, y_val : np.ndarray
        Training and validation data.
    label : str, optional
        Prefix for printed trial labels.
    num_boost_round, early_stopping_rounds, seed : see ``train_xgb_single``.
    param_grid : dict, optional
        Dictionary mapping hyperparameter names to lists of values to
        search over.  If ``None``, a reasonable default grid is used.
    fixed_params : dict, optional
        Parameters to keep constant across all trials.  These override
        entries in ``param_grid``.

    Returns
    -------
    best_booster : xgboost.Booster
        The model achieving the lowest mean squared error on the
        validation set.
    best_params : dict
        Hyperparameter settings for the best model.
    results_df : pd.DataFrame
        A dataframe summarising each trial with its metrics and
        hyperparameters.
    """
    if param_grid is None:
        param_grid = {
            "eta": [0.03, 0.05],
            "max_depth": [4, 6, 8],
            "subsample": [0.7, 0.9],
            "colsample_bytree": [0.7, 0.9],
            "reg_lambda": [0.0, 1.0],
        }
    if fixed_params is None:
        fixed_params = {}
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    results = []
    best = {"mse": np.inf, "booster": None, "params": None, "metrics": None}
    print(f"Total trials: {len(combos)}")
    for i, values in enumerate(combos, start=1):
        trial_params = dict(zip(keys, values))
        trial_params.update(fixed_params)
        booster, pred, metrics = train_xgb_single(
            X_tr, y_tr, X_val, y_val,
            label=f"{label} | trial {i}",
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            seed=seed,
            base_params=trial_params,
        )
        short = {k: trial_params[k] for k in keys}
        print(
            f"[trial {i:03d}] params={short} | MSE={metrics['mse']:.6f} "
            f"RMSE={metrics['rmse']:.6f} MAE={metrics['mae']:.6f} "
            f"R2={metrics['r2']:.6f} best_it={metrics['best_iteration']}"
        )
        row = {"trial": i, **trial_params, **metrics}
        results.append(row)
        if metrics["mse"] < best["mse"]:
            best = {
                "mse": metrics["mse"],
                "booster": booster,
                "params": trial_params,
                "metrics": metrics,
            }
    results_df = pd.DataFrame(results).sort_values("mse").reset_index(drop=True)
    print("\n=== Top 5 by MSE ===")
    display_cols = ["trial", *keys, "mse", "rmse", "mae", "r2", "best_iteration"]
    print(results_df[display_cols].head(5).to_string(index=False))
    return best["booster"], best["params"], results_df


def predict(booster: xgb.Booster, X: np.ndarray) -> np.ndarray:
    """Generate predictions from a fitted XGBoost booster.

    Parameters
    ----------
    booster : xgboost.Booster
        A trained model returned by ``train_xgb_single`` or
        ``grid_search_xgb``.
    X : np.ndarray
        Feature matrix for which to generate predictions.

    Returns
    -------
    np.ndarray
        Predicted values using the booster up to its ``best_iteration``.
    """
    dmat = xgb.DMatrix(X)
    best_it = getattr(booster, "best_iteration", None)
    if best_it is not None:
        return booster.predict(dmat, iteration_range=(0, best_it + 1))
    return booster.predict(dmat)