"""
The **solar_forecasting** package exposes helper functions and models for
preprocessing solar‑power datasets, performing feature engineering,
training predictive models (both gradient‑boosted trees and neural
networks) and computing common regression metrics.

This package was extracted and refactored from several Jupyter
notebooks. All functionality is organized into small, well‑documented
functions to encourage reusability and maintainability.  See
``usage_example.py`` for a demonstration of how the various pieces
fit together.
"""

from . import preprocessing
from . import feature_selection
from . import metrics
from . import xgb_model
from . import lstm_pytorch

__all__ = [
    "preprocessing",
    "feature_selection",
    "metrics",
    "xgb_model",
    "lstm_pytorch",
]