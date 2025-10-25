"""
Data loading and preprocessing utilities for solar power forecasting.

This module contains functions for cleaning column names, detecting
outliers, splitting a time‑ordered dataset into training, validation
and test segments and scaling features for use with machine learning
models.  The functions are written to be independent and reusable.

References
==========
The chronological split implemented here follows a common practice
for time‑series data where the last portion of the sequence is held
out for validation/testing.  The Python Packaging guide recommends
organizing code into modules rather than notebooks to encourage
reusability【479238146967212†L124-L133】.  For regression with LSTM models, a
sliding window of past observations is used to predict future values
【588888984768813†L75-L90】.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names by removing duplicate suffixes and
    collapsing whitespace.

    Many sensor data sets include columns whose names contain
    whitespace, newline characters or duplicate suffixes such as
    ``".1"``, ``".2"``.  This helper standardises column names in a
    simple, repeatable way so that downstream code can refer to the
    names consistently.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe whose columns need normalisation.

    Returns
    -------
    pd.DataFrame
        A new dataframe with cleaned column names.
    """
    cleaned = df.copy()
    # Remove ``.1`` / ``.2`` style suffixes and collapse whitespace.
    cleaned.columns = (
        cleaned.columns
        .str.replace(r"\.\d+$", "", regex=True)  # remove trailing .1, .2, ...
        .str.replace("\n", " ")  # replace line breaks with space
        .str.replace(r"\s+", " ", regex=True)  # collapse multiple spaces
        .str.strip()
    )
    return cleaned


def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """Flag observations that fall outside the interquartile range.

    The interquartile range (IQR) method identifies outliers as
    observations that lie below ``Q1 - factor * IQR`` or above
    ``Q3 + factor * IQR``.  This implementation returns a boolean
    mask indicating which values are considered outliers.  The
    function handles empty or constant series gracefully by returning
    all False.

    Parameters
    ----------
    series : pd.Series
        A numeric series to analyse.
    factor : float, optional
        Multiplier on the IQR used to define the outlier threshold.

    Returns
    -------
    pd.Series
        A boolean series aligned to the input index where True
        indicates an outlier.
    """
    valid = series.dropna()
    if valid.empty:
        return pd.Series(False, index=series.index)
    q1, q3 = valid.quantile([0.25, 0.75])
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=series.index)
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    mask = (series < lower) | (series > upper)
    return mask.reindex(series.index, fill_value=False)


@dataclass
class SplitData:
    """Container for split and scaled datasets.

    Attributes
    ----------
    X_train, X_val, X_test : np.ndarray
        Arrays of features for each split.
    y_train, y_val, y_test : np.ndarray
        Target arrays for each split.
    x_scaler, y_scaler : StandardScaler
        Fitted scalers used to standardise features and targets.
    """

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    x_scaler: StandardScaler
    y_scaler: StandardScaler


def time_series_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split feature and target data chronologically into train/validation/test.

    The split preserves ordering: training data comes from the earliest
    portion of the series, validation follows training and test is
    carved from the most recent observations.  This approach is
    commonly used for time‑series modelling【588888984768813†L75-L92】.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix where rows are sorted by time.
    y : pd.Series
        Corresponding target values sorted by time.
    test_fraction : float, optional
        Fraction of samples to reserve for the test set (default
        0.20).  Must be between 0 and 1.
    val_fraction : float, optional
        Fraction of the remaining data (after removing test) to use
        for validation (default 0.10).

    Returns
    -------
    Tuple containing six elements: ``(X_train, X_val, X_test, y_train,
    y_val, y_test)``.

    Raises
    ------
    ValueError
        If the fractions are not within (0, 1).
    """
    if not (0 < test_fraction < 1) or not (0 < val_fraction < 1):
        raise ValueError("test_fraction and val_fraction must be between 0 and 1")
    n = len(X)
    test_cut = int(n * (1.0 - test_fraction))
    trainval = X.iloc[:test_cut], y.iloc[:test_cut]
    test = X.iloc[test_cut:], y.iloc[test_cut:]
    n_trainval = len(trainval[0])
    val_cut = int(n_trainval * (1.0 - val_fraction))
    X_train = trainval[0].iloc[:val_cut].copy()
    X_val = trainval[0].iloc[val_cut:].copy()
    X_test = test[0].copy()
    y_train = trainval[1].iloc[:val_cut].copy()
    y_val = trainval[1].iloc[val_cut:].copy()
    y_test = test[1].copy()
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> SplitData:
    """Standardise features and targets using ``StandardScaler``.

    Scaling is performed by fitting separate ``StandardScaler``
    instances on the training data and then applying the learned
    parameters to validation and test sets.  This prevents information
    leakage from the validation/test sets into the training process.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Feature dataframes for each split.
    y_train, y_val, y_test : pd.Series
        Target series for each split.

    Returns
    -------
    SplitData
        A dataclass containing scaled arrays and the fitted scalers.
    """
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train.values.reshape(-1, 1))
    X_train_scaled = x_scaler.transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)
    X_test_scaled = x_scaler.transform(X_test)
    y_train_scaled = y_scaler.transform(y_train.values.reshape(-1, 1)).ravel()
    y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
    return SplitData(
        X_train=X_train_scaled,
        X_val=X_val_scaled,
        X_test=X_test_scaled,
        y_train=y_train_scaled,
        y_val=y_val_scaled,
        y_test=y_test_scaled,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )