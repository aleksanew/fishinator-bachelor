"""
Training loop and evaluation utilities for fishing vessel classification.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
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
    val_fraction: float = 0.0,  # >0 enables validation, early stopping, threshold tuning
    patience: int = 5,
) -> dict:
    """
    Train a model using BCEWithLogitsLoss with class-balanced sampling.

    When val_fraction > 0, a stratified validation split is held out from
    X_train. Val loss is tracked each epoch, early stopping fires after
    `patience` epochs without improvement, and the best decision threshold
    (maximising val F1) is stored in history["best_threshold"].

    Returns a dict with training history.
    """
    # --- Optional validation split (sequence-level; vessel-level split is upstream) ---
    if val_fraction > 0.0:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=val_fraction, stratify=y_train, random_state=42,
        )
    else:
        X_tr, y_tr = X_train, y_train
        X_val, y_val = None, None

    model = model.to(device)

    dataset = FishingSequenceDataset(X_tr, y_tr)
    sampler = make_weighted_sampler(y_tr)
    loader  = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    if X_val is not None:
        val_loader = DataLoader(
            FishingSequenceDataset(X_val, y_val),
            batch_size=batch_size, shuffle=False,
        )

    # Positive class weight for additional stability
    pos_weight = torch.tensor(
        [(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"loss": [], "val_loss": [], "best_threshold": 0.5}

    best_val_loss    = float("inf")
    patience_counter = 0
    best_state_dict  = None

    for epoch in range(epochs):
        model.train()
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

        # --- Validation ---
        if X_val is not None:
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    val_loss_sum += criterion(model(X_b.to(device)), y_b.to(device)).item()
            val_loss = val_loss_sum / len(val_loader)
            history["val_loss"].append(val_loss)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:>3d}/{epochs}  loss={avg_loss:.4f}  val_loss={val_loss:.4f}")

            # --- Early stopping ---
            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                best_state_dict  = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break
        else:
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:>3d}/{epochs}  loss={avg_loss:.4f}")

    # Restore best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # --- Threshold selection on val set ---
    if X_val is not None:
        y_val_prob, _ = predict(model, X_val, batch_size=batch_size, device=device)
        best_thr, best_f1 = 0.5, -1.0
        for thr in np.arange(0.1, 0.91, 0.02):
            score = f1_score(y_val, (y_val_prob >= thr).astype(int), zero_division=0)
            if score > best_f1:
                best_f1, best_thr = score, float(thr)
        history["best_threshold"] = best_thr
        if verbose:
            print(f"  Best val threshold: {best_thr:.2f}  (val F1={best_f1:.4f})")

    return history


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray,
            batch_size: int = 128, device: str = "cpu",
            threshold: float = 0.5) -> tuple:
    """
    Run inference.

    Returns
    -------
    y_prob : np.ndarray  (N,)  predicted probabilities
    y_pred : np.ndarray  (N,)  binary predictions at `threshold`
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
    y_pred = (y_prob >= threshold).astype(int)
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
