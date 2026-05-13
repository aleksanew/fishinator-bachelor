"""
models_nn.py

"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import average_precision_score


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FishingDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(labels,    dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# CNN1D
# ---------------------------------------------------------------------------

class CNN1D(nn.Module):

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

        self.fusion = nn.Sequential(
            nn.Conv1d(96, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)   # (B, F, T) for Conv1d
        x = torch.cat([self.branch_3(x), self.branch_7(x), self.branch_11(x)], dim=1)
        x = self.fusion(x)
        x = torch.cat([self.gap(x).squeeze(-1), self.gmp(x).squeeze(-1)], dim=-1)
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------------------------
# ForwardRNN
# ---------------------------------------------------------------------------

class ForwardRNN(nn.Module):
    def __init__(self, n_features: int, seq_len: int = 60,
                 hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.head(h_n[-1]).squeeze(-1)


# ---------------------------------------------------------------------------
# BiRNN
# ---------------------------------------------------------------------------

class BiRNN(nn.Module):
    def __init__(self, n_features: int, seq_len: int = 60,
                 hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        _, h_n = self.gru(x)
        last_h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return self.head(last_h).squeeze(-1)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

NN_MODELS = {
    "cnn":         CNN1D,
    "forward_rnn": ForwardRNN,
    "bi_rnn":      BiRNN,
}


def build_nn_model(name: str, n_features: int, seq_len: int, **kwargs) -> nn.Module:
    if name not in NN_MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(NN_MODELS)}")
    return NN_MODELS[name](n_features=n_features, seq_len=seq_len, **kwargs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_nn(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    model = model.to(device)

    dataset = FishingDataset(X_train, y_train)

    # Weighted sampler for class imbalance
    class_counts = np.bincount(y_train.astype(int))
    sample_weights = (1.0 / (class_counts + 1e-6))[y_train.astype(int)]
    sampler = WeightedRandomSampler(
        torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
        dtype=torch.float32, device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"loss": []}
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        history["loss"].append(avg_loss)
        if verbose and (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:>3d}/{epochs}  loss={avg_loss:.4f}")

    return history


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_nn(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int = 128,
    device: str = "cpu",
) -> np.ndarray:
    """Return predicted probabilities (N,)."""
    model.eval().to(device)
    dataset = FishingDataset(X, np.zeros(len(X)))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs   = []
    for X_batch, _ in loader:
        logits = model(X_batch.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)
