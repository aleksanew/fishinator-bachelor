"""
models_classical.py
Classical ML model definitions: Logistic Regression, Random Forest,
Gradient Boosting. All use the improved hyperparameters and class
imbalance handling discussed in the thesis.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np

RANDOM_STATE = 42


def make_logreg():
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=0.1,
        solver="lbfgs",
    )


def make_rf():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_gb():
    return GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
    )


def fit_classical(model, X_train, y_train):
    """Fit a classical model, applying sample weights for GB."""
    if isinstance(model, GradientBoostingClassifier):
        weights = compute_sample_weight("balanced", y_train)
        model.fit(X_train, y_train, sample_weight=weights)
    else:
        model.fit(X_train, y_train)
    return model


def baseline_predict(X_df):
    """Rule-based baseline: slow speed AND high turning rate."""
    return (
        (X_df["mean_speed"] < 5) &
        (X_df["turning_rate"] > 10)
    ).astype(int).values


CLASSICAL_MODELS = {
    "logreg": make_logreg,
    "rf":     make_rf,
    "gb":     make_gb,
}