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
    Three parallel Conv1D branches (k=3, 7, 11) capture fishing patterns at
    multiple temporal scales. Outputs are fused with a k=1 conv, then both
    global average and max pooling are concatenated before the MLP head.

    Architecture:
      branch_3:  Conv1d(F→32, k=3,  pad=1) → BN → ReLU
      branch_7:  Conv1d(F→32, k=7,  pad=3) → BN → ReLU
      branch_11: Conv1d(F→32, k=11, pad=5) → BN → ReLU
      cat → (batch, 96, T)
      fusion: Conv1d(96→64, k=1) → BN → ReLU → Dropout
      GAP + GMP → cat → (batch, 128)
      head: Linear(128→64) → ReLU → Dropout → Linear(64→1)
    """

    def __init__(self, n_features: int, seq_len: int, dropout: float = 0.3):
        super().__init__()

        def _branch(k):
            return nn.Sequential(
                nn.Conv1d(n_features, 32, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
            )

        self.branch_3  = _branch(3)
        self.branch_7  = _branch(7)
        self.branch_11 = _branch(11)

        # k=1 conv to compress 3×32=96 channels down to 64 before pooling
        self.fusion = nn.Sequential(
            nn.Conv1d(96, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)  # average across time → (batch, 64)
        self.gmp = nn.AdaptiveMaxPool1d(1)  # peak across time   → (batch, 64)

        self.head = nn.Sequential(
            nn.Linear(128, 64),  # 128 = 64 (GAP) + 64 (GMP)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch, T, F) → (batch, F, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = torch.cat([self.branch_3(x), self.branch_7(x), self.branch_11(x)], dim=1)
        x = self.fusion(x)                                         # (batch, 64, T)
        x = torch.cat([self.gap(x).squeeze(-1),
                        self.gmp(x).squeeze(-1)], dim=-1)          # (batch, 128)
        return self.head(x).squeeze(-1)                            # (batch,)


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

        # Learned query vector: scores each timestep's relevance for fishing detection
        self.attn_query = nn.Parameter(torch.randn(hidden_size))

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, T, F)
        out, _ = self.gru(x)                                        # (batch, T, hidden)
        scores  = out @ self.attn_query                             # (batch, T)
        scores  = scores / (self.gru.hidden_size ** 0.5)            # scale
        weights = torch.softmax(scores, dim=1)                      # (batch, T)
        context = (weights.unsqueeze(-1) * out).sum(dim=1)          # (batch, hidden)
        return self.head(context).squeeze(-1)                       # (batch,)


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

        # Each timestep in `out` already encodes both past and future context
        # (forward + backward GRU), making it a rich signal for attention to score.
        self.attn_query = nn.Parameter(torch.randn(2 * hidden_size))

        # Concatenated forward + backward → 2 * hidden_size
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch, T, F)
        out, _ = self.gru(x)                                        # (batch, T, 2*hidden)
        scores  = out @ self.attn_query                             # (batch, T)
        scores  = scores / ((2 * self.gru.hidden_size) ** 0.5)     # scale
        weights = torch.softmax(scores, dim=1)                      # (batch, T)
        context = (weights.unsqueeze(-1) * out).sum(dim=1)          # (batch, 2*hidden)
        return self.head(context).squeeze(-1)                       # (batch,)


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
