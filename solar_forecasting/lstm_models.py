"""
Deep learning models for solar forecasting using recurrent neural networks.

This module defines a set of utility functions to construct different
architectures of Long Short‑Term Memory (LSTM) networks, including
vanilla, stacked, bidirectional and convolutional variants. The
functions are designed to import TensorFlow/Keras lazily so that this
module can be imported even when the deep learning stack is not
installed; an informative exception will be raised when model builders
are called without a TensorFlow installation.
"""

from __future__ import annotations

from typing import Any, Dict


def _import_keras():
    """Internal helper to import Keras from TensorFlow.

    Raises:
        ImportError: If TensorFlow/Keras is not installed.
    """
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras  # type: ignore
        return keras
    except Exception as exc:
        raise ImportError(
            "TensorFlow and Keras must be installed to build LSTM models"
        ) from exc


def build_vanilla_lstm(input_shape: tuple[int, int], units: int = 64) -> Any:
    """Construct a simple single‑layer LSTM model.

    Args:
        input_shape: Shape of the input sequences as ``(timesteps, features)``.
        units: Number of LSTM units in the hidden layer.

    Returns:
        A compiled Keras model.

    Raises:
        ImportError: If TensorFlow/Keras is not installed.
    """
    keras = _import_keras()
    model = keras.models.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.LSTM(units),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_stacked_lstm(
    input_shape: tuple[int, int],
    units: int = 128,
    num_layers: int = 2,
) -> Any:
    """Construct a stacked LSTM with a configurable number of layers.

    Args:
        input_shape: Shape of the input sequences as ``(timesteps, features)``.
        units: Number of LSTM units in each layer.
        num_layers: How many LSTM layers to stack. Must be at least 1.

    Returns:
        A compiled Keras model.

    Raises:
        ValueError: If ``num_layers`` is less than 1.
        ImportError: If TensorFlow/Keras is not installed.
    """
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")
    keras = _import_keras()
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    for i in range(num_layers):
        return_sequences = i < num_layers - 1
        model.add(keras.layers.LSTM(units, return_sequences=return_sequences))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def build_bidirectional_lstm(
    input_shape: tuple[int, int],
    units: int = 128,
) -> Any:
    """Construct a bidirectional LSTM model.

    Args:
        input_shape: Shape of the input sequences as ``(timesteps, features)``.
        units: Number of LSTM units in each directional layer.

    Returns:
        A compiled Keras model.

    Raises:
        ImportError: If TensorFlow/Keras is not installed.
    """
    keras = _import_keras()
    model = keras.models.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Bidirectional(keras.layers.LSTM(units)),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_cnn_lstm(
    input_shape: tuple[int, int],
    conv_filters: int = 64,
    kernel_size: int = 3,
    lstm_units: int = 128,
) -> Any:
    """Construct a hybrid CNN‑LSTM model.

    The model applies a 1D convolution to capture local temporal patterns
    before feeding the representations into an LSTM layer.

    Args:
        input_shape: Shape of the input sequences as ``(timesteps, features)``.
        conv_filters: Number of convolutional filters.
        kernel_size: Width of the convolution kernel.
        lstm_units: Number of units in the LSTM layer.

    Returns:
        A compiled Keras model.

    Raises:
        ImportError: If TensorFlow/Keras is not installed.
    """
    keras = _import_keras()
    model = keras.models.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv1D(conv_filters, kernel_size, padding="causal", activation="relu"),
        keras.layers.LSTM(lstm_units),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model