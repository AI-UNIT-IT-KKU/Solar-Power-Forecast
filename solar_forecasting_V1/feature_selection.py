"""
Feature scoring and selection utilities for solar forecasting.

This module provides functions to compute the statistical relationship
between candidate input features and a target variable using Pearson
correlation and mutual information. These measures can guide the
construction of parsimonious and informative models.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

logger = logging.getLogger(__name__)


def compute_pearson_correlation(df: pd.DataFrame, target: str) -> pd.Series:
    """Compute absolute Pearson correlation between numeric features and a target.

    Args:
        df: DataFrame containing the target and candidate predictors.
        target: Name of the target column in ``df``.

    Returns:
        A ``pd.Series`` indexed by feature name containing the absolute
        correlation coefficient between each feature and the target.

    Raises:
        KeyError: If the target column does not exist.
        ValueError: If less than two numeric features are available.
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame")
    # Select numeric columns excluding target
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        raise ValueError("DataFrame must contain at least one numeric feature and a numeric target")
    features = numeric_df.columns.drop(target)
    correlations = numeric_df[features].corrwith(numeric_df[target]).abs()
    return correlations


def compute_mutual_information(df: pd.DataFrame, target: str, n_neighbors: int = 3) -> pd.Series:
    """Compute mutual information between numeric features and a numeric target.

    Mutual information is a non‑linear measure of dependence that can
    capture arbitrary relationships between variables. A higher mutual
    information score indicates a stronger dependency on the target.

    Args:
        df: DataFrame containing the target and candidate predictors. All
            features must be numeric.
        target: Name of the target column in ``df``.
        n_neighbors: Number of nearest neighbours to use for density
            estimation. This is passed directly to
            :func:`sklearn.feature_selection.mutual_info_regression`.

    Returns:
        A ``pd.Series`` indexed by feature name containing the mutual
        information scores.

    Raises:
        KeyError: If the target column does not exist.
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame")
    numeric_df = df.select_dtypes(include=[np.number])
    features = numeric_df.columns.drop(target)
    X = numeric_df[features]
    y = numeric_df[target]
    # Compute mutual information scores; handle potential numerical issues
    try:
        mi = mutual_info_regression(X, y, n_neighbors=n_neighbors, random_state=42)
    except Exception as exc:
        raise RuntimeError("Failed to compute mutual information scores") from exc
    return pd.Series(mi, index=features)


def select_top_features(
    corr_scores: pd.Series,
    mi_scores: pd.Series,
    n_corr: int = 10,
    n_mi: int = 10,
) -> Tuple[List[str], List[str]]:
    """Select top features based on correlation and mutual information.

    Given series of correlation and mutual information scores, this
    function returns two lists containing the names of the top
    ``n_corr`` and ``n_mi`` features, respectively. Ties are broken
    arbitrarily by pandas sorting.

    Args:
        corr_scores: A ``pd.Series`` of absolute correlation scores.
        mi_scores: A ``pd.Series`` of mutual information scores.
        n_corr: Number of top correlation features to return.
        n_mi: Number of top mutual information features to return.

    Returns:
        A tuple ``(features_corr, features_mi)`` where each element is a
        list of column names ordered by descending score.

    Raises:
        ValueError: If any of the score series is empty.
    """
    if corr_scores.empty:
        raise ValueError("Correlation score series is empty")
    if mi_scores.empty:
        raise ValueError("Mutual information score series is empty")
    # Sort in descending order
    top_corr = corr_scores.sort_values(ascending=False).head(n_corr).index.tolist()
    top_mi = mi_scores.sort_values(ascending=False).head(n_mi).index.tolist()
    return top_corr, top_mi