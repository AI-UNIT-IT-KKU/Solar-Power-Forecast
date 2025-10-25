"""
Data loading and preprocessing utilities for the solar forecasting project.

This module groups together functions for resampling time‑series data,
normalising column names, detecting outliers and handling missing values
across multiple sensor files. Each function is implemented independently
with clear type hints, docstrings and meaningful variable names. Errors
are handled gracefully by raising exceptions with informative messages.
"""

from __future__ import annotations

import logging
from typing import Iterable, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def resample_data(
    df: pd.DataFrame,
    freq: str = "1min",
    time_column: str | None = None,
) -> pd.DataFrame:
    """Resample a time‑indexed DataFrame to a specified frequency.

    If ``time_column`` is provided, it will be converted to a datetime
    index before resampling. Otherwise the existing index must already
    be a ``DatetimeIndex``. Numeric columns are aggregated using the
    mean; non‑numeric columns will be forward‑filled.

    Args:
        df: The input DataFrame containing a datetime index or a
            separate time column. This DataFrame is **not** modified in
            place.
        freq: The new sampling frequency expressed in pandas offset
            alias notation (e.g. ``"1min"`` for one minute). Defaults to
            one minute.
        time_column: Optional name of the column containing timestamps.
            If provided, this column will be parsed into a datetime
            index and dropped from the returned DataFrame.

    Returns:
        A new DataFrame resampled to the desired frequency.

    Raises:
        ValueError: If the index cannot be converted to datetime or
            ``time_column`` does not exist in ``df``.
    """
    # Work on a copy to avoid mutating the caller's DataFrame
    df_copy = df.copy()

    # Ensure we have a datetime index
    if time_column is not None:
        if time_column not in df_copy.columns:
            raise ValueError(
                f"time_column '{time_column}' not found in DataFrame columns"
            )
        try:
            df_copy[time_column] = pd.to_datetime(df_copy[time_column])
        except Exception as exc:
            raise ValueError(
                f"Failed to convert column '{time_column}' to datetime"
            ) from exc
        df_copy = df_copy.set_index(time_column)
    else:
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            # Attempt to coerce index to datetime
            try:
                df_copy.index = pd.to_datetime(df_copy.index)
            except Exception as exc:
                raise ValueError(
                    "DataFrame index is not a DatetimeIndex and cannot be coerced"
                ) from exc

    # Determine aggregation for numeric vs. non‑numeric columns
    num_cols = df_copy.select_dtypes(include=[np.number]).columns
    other_cols = df_copy.columns.difference(num_cols)

    # Aggregate numeric columns by mean
    df_num = df_copy[num_cols].resample(freq).mean()

    # Forward fill non‑numeric columns
    df_other = df_copy[other_cols].resample(freq).ffill()

    # Combine and return new DataFrame with sorted columns
    result = pd.concat([df_num, df_other], axis=1).sort_index()
    return result


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names by removing trailing numeric suffixes.

    Some data files may include duplicate column names distinguished by
    suffixes such as ``.1``, ``.2`` etc. This function strips those
    suffixes while preserving the base name, ensuring consistency across
    multiple files.

    Args:
        df: Input DataFrame whose columns will be normalised. The
            original DataFrame is not modified.

    Returns:
        A new DataFrame with cleaned column names.
    """
    df_copy = df.copy()
    # Use regex to remove a dot followed by digits at the end of a name
    df_copy.columns = (
        df_copy.columns.str.replace(r"\.\d+$", "", regex=True)
    )
    return df_copy


def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """Detect outlier values in a numeric series using the IQR method.

    An outlier is defined as a value lying below
    ``Q1 - 1.5 * IQR`` or above ``Q3 + 1.5 * IQR`` where ``Q1`` and
    ``Q3`` are the 25th and 75th percentiles respectively.

    Args:
        series: A numeric pandas Series.

    Returns:
        A boolean Series of the same length where ``True`` indicates
        that the corresponding value is an outlier.

    Raises:
        TypeError: If ``series`` is not numeric.
    """
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("detect_outliers_iqr expects a numeric Series")

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (series < lower_bound) | (series > upper_bound)


def replace_outliers_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace outlier values in all numeric columns with ``NaN``.

    Args:
        df: Input DataFrame. It will not be modified in place.

    Returns:
        A new DataFrame where numeric column values identified as
        outliers have been replaced by ``NaN``.
    """
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=[np.number]).columns:
        try:
            mask = detect_outliers_iqr(df_copy[col])
            df_copy.loc[mask, col] = np.nan
        except TypeError:
            # Skip non‑numeric columns silently
            continue
    return df_copy


def fill_missing_cross(dfs: Iterable[pd.DataFrame]) -> List[pd.DataFrame]:
    """Cross‑fill missing numeric values across aligned DataFrames.

    When multiple sensor files monitor the same physical system, it is
    often beneficial to fill missing values in one file using available
    observations from other files. This function assumes that all
    provided DataFrames share the same index and column names. For each
    numeric column and timestamp, the missing value in a given
    DataFrame will be replaced by the mean of the non‑missing values
    across all other DataFrames at that timestamp.

    Args:
        dfs: An iterable of DataFrames to fill. Each DataFrame should
            have the same shape, index and columns. Non‑numeric columns
            are ignored.

    Returns:
        A list of new DataFrames with cross‑filled missing values. The
        ordering of the returned list matches the input ordering.

    Raises:
        ValueError: If no DataFrames are provided or if their shapes
            and columns do not match.
    """
    dfs_list = list(dfs)
    if not dfs_list:
        raise ValueError("fill_missing_cross requires at least one DataFrame")

    # Validate shapes and columns
    first_shape = dfs_list[0].shape
    first_cols = dfs_list[0].columns
    for df in dfs_list[1:]:
        if df.shape != first_shape or not df.columns.equals(first_cols):
            raise ValueError(
                "All DataFrames must have the same shape and columns to cross‑fill"
            )

    num_cols = first_cols[first_cols.map(lambda c: pd.api.types.is_numeric_dtype(dfs_list[0][c]))]
    result_dfs = []

    for i, df in enumerate(dfs_list):
        df_copy = df.copy()
        for col in num_cols:
            # Identify missing values in this DataFrame for the given column
            missing_mask = df_copy[col].isna()
            if not missing_mask.any():
                continue
            # Stack values from other dataframes at the same timestamps
            other_values = [d[col] for j, d in enumerate(dfs_list) if j != i]
            # Compute row‑wise mean across other values, ignoring NaNs
            mean_values = pd.concat(other_values, axis=1).mean(axis=1)
            # Fill missing with mean
            df_copy.loc[missing_mask, col] = mean_values[missing_mask]
        result_dfs.append(df_copy)
    return result_dfs