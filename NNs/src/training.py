"""
Training loop and evaluation utilities for fishing vessel classification.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

from src.models import FishingSequenceDataset


# ---------------------------------------------------------------------------
# Weighted sampler to handle class imbalance
# ---------------------------------------------------------------------------

def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Return a WeightedRandomSampler that up-samples the minority class."""
    class_counts = np.bincount(labels.astype(int))
    weights_per_class = 1.0 / (class_counts + 1e-6)
    sample_weights = weights_per_class[labels.astype(int)]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    X_train: np.ndarray,   # (N, T, F)
    y_train: np.ndarray,   # (N,)
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """
    Train a model using BCEWithLogitsLoss with class-balanced sampling.

    Returns a dict with training history (loss per epoch).
    """
    model = model.to(device)

    dataset = FishingSequenceDataset(X_train, y_train)
    sampler = make_weighted_sampler(y_train)
    loader  = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # Positive class weight for additional stability
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"loss": []}

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        history["loss"].append(avg_loss)

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:>3d}/{epochs}  loss={avg_loss:.4f}")

    return history


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray,
            batch_size: int = 128, device: str = "cpu") -> tuple:
    """
    Run inference.

    Returns
    -------
    y_prob : np.ndarray  (N,)  predicted probabilities
    y_pred : np.ndarray  (N,)  binary predictions at threshold 0.5
    """
    model.eval()
    model = model.to(device)

    dataset = FishingSequenceDataset(X, np.zeros(len(X)))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        logits  = model(X_batch)
        probs   = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    y_prob = np.concatenate(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_prob, y_pred


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             y_prob: np.ndarray, model_name: str = "") -> dict:
    """Compute and print a full suite of classification metrics."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)

    prefix = f"[{model_name}] " if model_name else ""
    print(f"\n{prefix}Evaluation Results")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    return {
        "model":     model_name,
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "roc_auc":   auc,
        "confusion_matrix": cm.tolist(),
    }
