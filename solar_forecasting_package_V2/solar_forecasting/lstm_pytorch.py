"""
PyTorch LSTM models and training helpers for time‑series regression.

This module contains multiple LSTM architectures (vanilla, stacked,
bidirectional and CNN+LSTM) along with utilities for preparing
sequence data, training models with early stopping and evaluating
performance.  The implementations draw on concepts of sliding windows
for time‑series prediction【588888984768813†L75-L90】.  Early stopping monitors
validation loss and halts training when no improvement is observed
for a specified number of epochs.

Example
-------
::

    from solar_forecasting.lstm_pytorch import (
        SeqDataset, VanillaLSTMReg, train_one_window, make_sequences
    )
    # prepare windowed sequences
    X_seq, y_seq = make_sequences(X_scaled, y_scaled, window=60)
    # create dataloaders
    train_ds = SeqDataset(X_seq[:-100], y_seq[:-100])
    val_ds = SeqDataset(X_seq[-100:], y_seq[-100:])
    # train a vanilla LSTM
    model, history = train_one_window(
        "vanilla", X_seq[:-100], y_seq[:-100], X_seq[-100:], y_seq[-100:],
        window=60
    )

Notes
-----
* Models are defined as subclasses of ``torch.nn.Module`` and accept
  three‑dimensional input tensors of shape ``(batch, window, n_features)``.
* ``train_one_window`` will automatically select the appropriate
  architecture based on the ``kind`` parameter.
* Gradients are clipped to mitigate exploding gradients.
"""

from __future__ import annotations

import logging
from typing import Tuple, Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .metrics import mean_absolute_error, root_mean_squared_error, r2_score

logger = logging.getLogger(__name__)

# Detect the best available device (GPU/CPU)
DEVICE: str = "cuda" if torch.cuda.is_available() else (
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
)


class SeqDataset(Dataset):
    """Dataset for feeding sequence data to PyTorch models.

    Converts numpy arrays into PyTorch tensors and exposes them via
    the Dataset interface.  Each item is a tuple ``(X[i], y[i])``.

    Parameters
    ----------
    X : np.ndarray
        Array of input sequences of shape ``(n_samples, window, n_features)``.
    y : np.ndarray
        Array of targets of shape ``(n_samples,)`` or ``(n_samples, 1)``.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(np.asarray(X)).float()
        # Flatten y to 1D if it has an extra dimension
        y_arr = np.asarray(y).reshape(-1)
        self.y = torch.from_numpy(y_arr).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class VanillaLSTMReg(nn.Module):
    """Single‑layer LSTM for regression with dropout.

    Parameters
    ----------
    n_features : int
        Number of input features per timestep.
    hidden : int, optional
        Number of hidden units in the LSTM layer.  Default is 64.
    dropout : float, optional
        Dropout probability applied to the hidden state before the final
        linear layer.  Default is 0.1.
    """

    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)            # out: (B, W, H)
        last = out[:, -1, :]             # take last timestep: (B, H)
        last = self.dropout(last)
        return self.fc(last).squeeze(-1)


class StackedLSTMReg(nn.Module):
    """Two‑layer LSTM for regression with dropout between layers.

    Parameters
    ----------
    n_features : int
        Number of input features per timestep.
    hidden1 : int, optional
        Hidden units in the first LSTM layer.  Default is 128.
    hidden2 : int, optional
        Hidden units in the second LSTM layer.  Default is 64.
    dropout1 : float, optional
        Dropout applied after the first LSTM layer.  Default is 0.01.
    dropout2 : float, optional
        Dropout applied after the second LSTM layer.  Default is 0.2.
    """

    def __init__(
        self,
        n_features: int,
        hidden1: int = 128,
        hidden2: int = 64,
        dropout1: float = 0.01,
        dropout2: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden1,
            num_layers=1,
            batch_first=True,
        )
        self.do1 = nn.Dropout(dropout1)
        self.lstm2 = nn.LSTM(
            input_size=hidden1,
            hidden_size=hidden2,
            num_layers=1,
            batch_first=True,
        )
        self.do2 = nn.Dropout(dropout2)
        self.fc = nn.Linear(hidden2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, _ = self.lstm1(x)          # (B, W, H1)
        out1 = self.do1(out1)
        out2, _ = self.lstm2(out1)       # (B, W, H2)
        out2 = self.do2(out2)
        last = out2[:, -1, :]            # (B, H2)
        return self.fc(last).squeeze(-1)


class BiLSTMReg(nn.Module):
    """Two‑layer bidirectional LSTM for regression.

    Parameters
    ----------
    n_features : int
        Number of input features per timestep.
    hidden1 : int, optional
        Hidden units in the first LSTM layer per direction.  Default is 128.
    hidden2 : int, optional
        Hidden units in the second LSTM layer per direction.  Default is 64.
    dropout : float, optional
        Dropout applied between layers.  Default is 0.2.
    """

    def __init__(
        self,
        n_features: int,
        hidden1: int = 128,
        hidden2: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.bi1 = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.do1 = nn.Dropout(dropout)
        self.bi2 = nn.LSTM(
            input_size=hidden1 * 2,
            hidden_size=hidden2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.do2 = nn.Dropout(dropout)
        # Final feature dimension doubles due to bidirectionality
        self.fc = nn.Linear(hidden2 * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, _ = self.bi1(x)            # (B, W, H1*2)
        out1 = self.do1(out1)
        out2, _ = self.bi2(out1)         # (B, W, H2*2)
        out2 = self.do2(out2)
        last = out2[:, -1, :]            # (B, H2*2)
        return self.fc(last).squeeze(-1)


class CNNLSTMReg(nn.Module):
    """1D Convolution followed by LSTM for regression.

    Parameters
    ----------
    n_features : int
        Number of input features per timestep (treated as channels).
    conv_channels : int, optional
        Number of convolutional filters.  Default is 64.
    kernel_size : int, optional
        Length of the 1D convolution kernels.  Default is 3.
    pool : int, optional
        Max pooling size.  Default is 2.
    lstm_hidden : int, optional
        Hidden units in the LSTM layer.  Default is 64.
    dropout : float, optional
        Dropout probability after the convolution and LSTM layers.
    """

    def __init__(
        self,
        n_features: int,
        conv_channels: int = 64,
        kernel_size: int = 3,
        pool: int = 2,
        lstm_hidden: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # In conv1d, channels correspond to features and length to window
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=pad,
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=pool)
        self.conv2 = nn.Conv1d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=pad,
        )
        self.relu2 = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.do = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, F) => (B, F, W)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        # transpose back: (B, C, L) => (B, L, C)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        out = self.do(out)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)


def make_sequences(X: np.ndarray, y: np.ndarray, window: int, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Construct sequences of fixed length for time‑series regression.

    Uses a sliding window approach: for each position ``i``, a slice
    ``X[i : i + window]`` is taken as the predictor and the target is
    ``y[i + window + horizon - 1]``【588888984768813†L75-L90】.  The resulting arrays are
    returned as float32 to conserve memory.

    Parameters
    ----------
    X : np.ndarray
        2D array of shape ``(n_samples, n_features)``.
    y : np.ndarray
        1D or 2D array of targets.  If 2D, the last dimension must be 1.
    window : int
        Number of timesteps in each input sequence.
    horizon : int, optional
        Number of steps ahead to predict.  Default is 1.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple ``(xs, ys)`` where ``xs`` has shape
        ``(n_sequences, window, n_features)`` and ``ys`` has shape
        ``(n_sequences,)``.
    """
    xs: List[np.ndarray] = []
    ys: List[float] = []
    y_flat = np.asarray(y).reshape(-1)
    for i in range(len(X) - window - horizon + 1):
        xs.append(X[i : i + window])
        ys.append(y_flat[i + window + horizon - 1])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Run a single training or evaluation epoch.

    Parameters
    ----------
    model : nn.Module
        The neural network to train or evaluate.
    loader : DataLoader
        DataLoader yielding batches of ``(X, y)`` pairs.
    criterion : nn.Module
        Loss function (e.g. ``nn.MSELoss()``).
    optimizer : torch.optim.Optimizer, optional
        If provided, the model is trained; otherwise, it is evaluated.

    Returns
    -------
    tuple
        A triple ``(loss, y_true, y_pred)`` where ``loss`` is the
        average loss across batches and ``y_true``/``y_pred`` are
        concatenated numpy arrays of true and predicted values.
    """
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    losses: List[float] = []
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    with torch.set_grad_enabled(train_mode):
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            yhat = model(xb)
            loss = criterion(yhat, yb)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                # clip gradients to avoid exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            losses.append(loss.item())
            preds.append(yhat.detach().cpu().numpy())
            trues.append(yb.detach().cpu().numpy())
    if not preds:
        return float("nan"), np.array([]), np.array([])
    preds_arr = np.concatenate(preds).reshape(-1)
    trues_arr = np.concatenate(trues).reshape(-1)
    return float(np.mean(losses)), trues_arr, preds_arr


def train_one_window(
    kind: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    window: int,
    batch: int = 256,
    epochs: int = 200,
    patience: int = 10,
    verbose: bool = False,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train a model for a specific window length with early stopping.

    Parameters
    ----------
    kind : {'vanilla', 'stacked', 'bi', 'cnn-lstm'}
        Specifies which architecture to train.
    X_tr, y_tr, X_val, y_val : np.ndarray
        Training and validation sequences and targets.
    window : int
        Length of the input sequences.
    batch : int, optional
        Batch size for training.  Default is 256.
    epochs : int, optional
        Maximum number of epochs.  Default is 200.
    patience : int, optional
        Number of epochs to wait for improvement in validation loss
        before early stopping.  Default is 10.
    verbose : bool, optional
        If True, print progress messages each epoch.

    Returns
    -------
    model : nn.Module
        Trained model restored to the best state observed during
        training.
    history : dict
        Dictionary of training history containing keys ``loss``,
        ``val_loss``, ``mae`` and ``val_mae`` (scaled domain).
    """
    n_features = X_tr.shape[-1]
    kind_lower = kind.lower()
    if kind_lower == "vanilla":
        model: nn.Module = VanillaLSTMReg(n_features)
    elif kind_lower == "stacked":
        model = StackedLSTMReg(n_features)
    elif kind_lower in ("bi", "bidirectional"):
        model = BiLSTMReg(n_features)
    elif kind_lower in ("cnn_lstm", "cnn-lstm"):
        model = CNNLSTMReg(n_features)
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    model = model.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5, verbose=verbose
    )
    # Dataloaders
    train_loader = DataLoader(SeqDataset(X_tr, y_tr), batch_size=batch, shuffle=False)
    val_loader = DataLoader(SeqDataset(X_val, y_val), batch_size=batch, shuffle=False)
    history: Dict[str, List[float]] = {
        "loss": [],
        "val_loss": [],
        "mae": [],
        "val_mae": [],
    }
    best_val = float("inf")
    best_state: Dict[str, Any] | None = None
    wait = 0
    for epoch in range(1, epochs + 1):
        tr_loss, yt_tr, yp_tr = run_epoch(model, train_loader, criterion, optimizer=optimizer)
        val_loss, yt_val, yp_val = run_epoch(model, val_loader, criterion, optimizer=None)
        # scaled domain MAE for reporting
        tr_mae = mean_absolute_error(yt_tr, yp_tr) if yt_tr.size > 0 else float("nan")
        val_mae = mean_absolute_error(yt_val, yp_val) if yt_val.size > 0 else float("nan")
        history["loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["mae"].append(tr_mae)
        history["val_mae"].append(val_mae)
        scheduler.step(val_loss)
        if verbose:
            print(
                f"Epoch {epoch:03d} | loss {tr_loss:.4f} val_loss {val_loss:.4f} "
                f"| MAE(sc) {tr_mae:.4f} val_MAE(sc) {val_mae:.4f}"
            )
        # Early stopping check
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print("Early stopping")
                break
    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history