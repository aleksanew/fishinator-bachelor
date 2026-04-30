"""
PyTorch models for fishing vessel detection from AIS sequences.

Three architectures:
  1. CNN1D       – 1-D convolutional network over the time dimension
  2. ForwardRNN  – Unidirectional GRU
  3. BiRNN       – Bidirectional GRU (forward + backward)

All models:
  - Input:  (batch, seq_len, n_features)
  - Output: (batch, 1)  logit (use BCEWithLogitsLoss)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FishingSequenceDataset(Dataset):
    """Wraps (N, T, F) numpy arrays into a PyTorch Dataset."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Parameters
        ----------
        sequences : np.ndarray  (N, T, F)  float32
        labels    : np.ndarray  (N,)       int / float
        """
        assert len(sequences) == len(labels), "sequences and labels must have same length"
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(labels,    dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# 1.  1-D CNN
# ---------------------------------------------------------------------------

class CNN1D(nn.Module):
    """
    Three stacked Conv1D blocks with residual-style skip connections,
    followed by global average pooling and an MLP head.

    Architecture (with default params):
      Conv1d(F, 64, k=3) → BN → ReLU → Dropout
      Conv1d(64, 128, k=3) → BN → ReLU → Dropout
      Conv1d(128, 64, k=3) → BN → ReLU
      GlobalAvgPool → Linear(64→32) → ReLU → Linear(32→1)
    """

    def __init__(self, n_features: int, seq_len: int,
                 channels: tuple = (64, 128, 64),
                 kernel_size: int = 3,
                 dropout: float = 0.3):
        super().__init__()

        layers = []
        in_ch = n_features
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch

        self.conv_blocks = nn.Sequential(*layers)

        # Global average pool → output is (batch, in_ch)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(in_ch, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, T, F) → permute to (batch, F, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv_blocks(x)       # (batch, C_last, T)
        x = self.gap(x).squeeze(-1)   # (batch, C_last)
        return self.head(x).squeeze(-1)  # (batch,)


# ---------------------------------------------------------------------------
# 2.  Forward (unidirectional) GRU RNN
# ---------------------------------------------------------------------------

class ForwardRNN(nn.Module):
    """
    Stacked unidirectional GRU.

    The final hidden state of the last layer is passed through an MLP head.
    """

    def __init__(self, n_features: int, seq_len: int = 60,
                 hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.gru = nn.GRU(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
            bidirectional = False,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, T, F)
        _, h_n = self.gru(x)          # h_n: (num_layers, batch, hidden)
        last_h = h_n[-1]              # (batch, hidden)
        return self.head(last_h).squeeze(-1)  # (batch,)


# ---------------------------------------------------------------------------
# 3.  Bidirectional (forward + backward) GRU
# ---------------------------------------------------------------------------

class BiRNN(nn.Module):
    """
    Stacked bidirectional GRU.

    Forward and backward final hidden states are concatenated before the head.
    """

    def __init__(self, n_features: int, seq_len: int = 60,
                 hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.gru = nn.GRU(
            input_size    = n_features,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
        )

        # Concatenated forward + backward → 2 * hidden_size
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch, T, F)
        _, h_n = self.gru(x)
        # h_n: (num_layers * 2, batch, hidden)
        # Last layer: forward = h_n[-2], backward = h_n[-1]
        last_h = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (batch, 2*hidden)
        return self.head(last_h).squeeze(-1)             # (batch,)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "cnn":         CNN1D,
    "forward_rnn": ForwardRNN,
    "bi_rnn":      BiRNN,
}


def build_model(name: str, n_features: int, seq_len: int, **kwargs) -> nn.Module:
    """
    Build a model by name.

    Parameters
    ----------
    name       : one of 'cnn', 'forward_rnn', 'bi_rnn'
    n_features : number of input features per time-step
    seq_len    : sequence length (only used for CNN)
    **kwargs   : passed to the model constructor
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[name]
    return cls(n_features=n_features, seq_len=seq_len, **kwargs)
