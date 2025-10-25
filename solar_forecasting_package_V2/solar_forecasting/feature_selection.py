"""
Feature scoring and selection utilities.

This module provides functions to calculate feature importance metrics
including Pearson and Spearman correlations and mutual information for
regression tasks.  The functions return pandas objects to allow easy
ranking and filtering.  Use these scores to select a subset of
informative predictors for models such as XGBoost or LSTM.

References
==========
Pearson correlation measures linear association between variables,
while mutual information can capture non‑linear relationships.  In
practice, sliding windows of past observations are often used for
time‑series prediction【588888984768813†L75-L90】.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression


def correlation_scores(X: pd.DataFrame, y: pd.Series, method: str = "pearson") -> pd.Series:
    """Compute correlation between each feature and the target.

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe of numeric features.
    y : pd.Series
        Target variable.
    method : {'pearson', 'spearman'}, optional
        Correlation metric to compute.  ``'pearson'`` measures linear
        association, whereas ``'spearman'`` measures rank‑based
        association.  Default is ``'pearson'``.

    Returns
    -------
    pd.Series
        A series of correlations indexed by feature name.

    Raises
    ------
    ValueError
        If an unsupported method is provided.
    """
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be either 'pearson' or 'spearman'")
    if method == "pearson":
        return X.corrwith(y, method="pearson")
    return X.corrwith(y, method="spearman")


def mutual_info_scores(X: pd.DataFrame, y: pd.Series, random_state: int | None = None) -> pd.Series:
    """Estimate mutual information (MI) between each feature and target.

    Mutual information measures the dependency between two variables
    without assuming a particular functional form.  A higher MI value
    indicates that knowing the feature reduces uncertainty about the
    target.

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe of numeric features.  Missing values are filled
        internally with zeros before computation.
    y : pd.Series
        Target variable.  Missing values are filled with zeros.
    random_state : int, optional
        Seed for the underlying estimator used to approximate MI.

    Returns
    -------
    pd.Series
        Estimated mutual information for each feature.
    """
    # Fill NaNs to avoid exceptions in mutual_info_regression
    X_filled = X.fillna(0.0)
    y_filled = y.fillna(0.0)
    mi = mutual_info_regression(X_filled, y_filled, random_state=random_state)
    return pd.Series(mi, index=X.columns)


def rank_features(
    X: pd.DataFrame,
    y: pd.Series,
    methods: tuple[str, ...] = ("pearson", "spearman", "mi"),
    random_state: int | None = None,
) -> pd.DataFrame:
    """Calculate and combine multiple feature importance scores.

    This convenience function calculates Pearson and Spearman
    correlations as well as mutual information.  It returns a
    dataframe with one row per feature and columns for each score.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    methods : tuple of str, optional
        Scores to compute.  Valid values are ``'pearson'``,
        ``'spearman'`` and ``'mi'``.  Default is all three.
    random_state : int, optional
        Random seed used for mutual information estimation.

    Returns
    -------
    pd.DataFrame
        A dataframe indexed by feature name containing columns for
        each requested score and additional helper columns ``pearson_abs``
        (absolute Pearson correlation) and ``mi_norm`` (MI normalised
        by its maximum).
    """
    allowed = {"pearson", "spearman", "mi"}
    invalid = set(methods) - allowed
    if invalid:
        raise ValueError(f"Unsupported methods: {invalid}")
    scores = {}
    if "pearson" in methods:
        p = correlation_scores(X, y, method="pearson")
        scores["pearson"] = p
    if "spearman" in methods:
        s = correlation_scores(X, y, method="spearman")
        scores["spearman"] = s
    if "mi" in methods:
        m = mutual_info_scores(X, y, random_state=random_state)
        scores["mi"] = m
    df_scores = pd.concat(scores, axis=1)
    if "pearson" in df_scores.columns:
        df_scores["pearson_abs"] = df_scores["pearson"].abs()
    if "mi" in df_scores.columns:
        max_mi = df_scores["mi"].max() or 1.0
        df_scores["mi_norm"] = df_scores["mi"] / max_mi
    return df_scores