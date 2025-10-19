"""
Top-level package for the solar_forecasting project.

This package provides utilities for loading, cleaning and processing
time‑series data recorded from solar power systems, selecting informative
features and training forecasting models. All functions exposed here
follow the clean code guidelines: they are written as self‑contained
functions with meaningful names, clear docstrings and robust error
handling. Refer to the individual modules for detailed documentation.
"""

from .data_processing import (
    resample_data,
    normalize_columns,
    detect_outliers_iqr,
    replace_outliers_with_nan,
    fill_missing_cross,
)
from .feature_selection import (
    compute_pearson_correlation,
    compute_mutual_information,
    select_top_features,
)
from .modeling import (
    chronological_split,
    train_xgboost,
)
from .metrics import evaluate_predictions

__all__ = [
    "resample_data",
    "normalize_columns",
    "detect_outliers_iqr",
    "replace_outliers_with_nan",
    "fill_missing_cross",
    "compute_pearson_correlation",
    "compute_mutual_information",
    "select_top_features",
    "chronological_split",
    "train_xgboost",
    "evaluate_predictions",
]