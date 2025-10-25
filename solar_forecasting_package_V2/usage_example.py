"""
Example script demonstrating how to use the ``solar_forecasting`` package.

This script outlines a simple workflow: load a CSV file containing
time‑ordered data, clean up column names, split the data
chronologically, scale the features, compute feature scores, train
an XGBoost model with a small hyperparameter grid search and train a
vanilla LSTM model using PyTorch.  The purpose is illustrative;
adapt the code to your own datasets and experiment with different
models and parameters.

The workflow draws on best practices for packaging and code reuse
suggested in the Python Packaging Guide【479238146967212†L124-L133】 and uses a
sliding window to prepare sequences for the LSTM【588888984768813†L75-L90】.

Usage
-----
To run this example, ensure that all required dependencies (pandas,
numpy, scikit‑learn, xgboost and torch) are installed.  Then
execute::

    python usage_example.py --csv your_data.csv --target active_power

The script expects a CSV file path passed via ``--csv`` and the name
of the target column via ``--target``.  See the argument parser
definition below for other optional parameters.
"""

from __future__ import annotations

import argparse
import pandas as pd
import numpy as np

from solar_forecasting import preprocessing, feature_selection, xgb_model, lstm_pytorch
from solar_forecasting.metrics import summarize

import torch.nn as nn
from torch.utils.data import DataLoader


def main() -> None:
    """Parse arguments and run the end‑to‑end example."""
    parser = argparse.ArgumentParser(description="Solar forecasting example")
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--target", type=str, required=True, help="Name of target column")
    parser.add_argument(
        "--test-fraction", type=float, default=0.2,
        help="Fraction of samples to reserve for testing (default 0.2)",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.1,
        help="Fraction of remaining data reserved for validation (default 0.1)",
    )
    parser.add_argument(
        "--window", type=int, default=60,
        help="Window size for sequence models (default 60)",
    )
    args = parser.parse_args()
    # Load and clean CSV
    df = pd.read_csv(args.csv)
    df = preprocessing.normalize_columns(df)
    if args.target not in df.columns:
        raise KeyError(f"Target column '{args.target}' not found in the data")
    # Separate features and target
    X = df.drop(columns=[args.target])
    y = df[args.target]
    # Perform chronological split
    X_train_df, X_val_df, X_test_df, y_train_s, y_val_s, y_test_s = preprocessing.time_series_split(
        X,
        y,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
    )
    # Scale features and target
    split = preprocessing.scale_splits(
        X_train_df, X_val_df, X_test_df, y_train_s, y_val_s, y_test_s
    )
    # Compute feature scores on training data
    scores = feature_selection.rank_features(
        X_train_df.select_dtypes(include="number"), y_train_s
    )
    print("Top 10 features by absolute Pearson correlation:")
    print(
        scores.sort_values("pearson_abs", ascending=False).head(10)
    )
    # Train a small XGBoost model
    booster, best_params, results_df = xgb_model.grid_search_xgb(
        split.X_train,
        split.y_train,
        split.X_val,
        split.y_val,
        param_grid={"eta": [0.03, 0.05], "max_depth": [4, 6]},
        fixed_params={"subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0},
    )
    # Evaluate on test set using best iteration
    y_test_pred = xgb_model.predict(booster, split.X_test)
    print("\nXGBoost Test Metrics:")
    print(summarize(split.y_test, y_test_pred, tag="XGB_test"))
    # Prepare sequences for LSTM
    X_seq_tr, y_seq_tr = lstm_pytorch.make_sequences(
        split.X_train, split.y_train, window=args.window
    )
    X_seq_val, y_seq_val = lstm_pytorch.make_sequences(
        split.X_val, split.y_val, window=args.window
    )
    X_seq_te, y_seq_te = lstm_pytorch.make_sequences(
        split.X_test, split.y_test, window=args.window
    )
    # Train a vanilla LSTM
    model, history = lstm_pytorch.train_one_window(
        "vanilla",
        X_seq_tr,
        y_seq_tr,
        X_seq_val,
        y_seq_val,
        window=args.window,
        epochs=50,
        patience=5,
        verbose=True,
    )
    # Evaluate on test sequences
    test_loader = DataLoader(
        lstm_pytorch.SeqDataset(X_seq_te, y_seq_te),
        batch_size=256,
        shuffle=False,
    )
    _, y_te_true, y_te_pred = lstm_pytorch.run_epoch(
        model, test_loader, nn.MSELoss(), optimizer=None
    )
    print("\nLSTM Test Metrics:")
    print(summarize(y_te_true, y_te_pred, tag="LSTM_test"))


if __name__ == "__main__":
    main()