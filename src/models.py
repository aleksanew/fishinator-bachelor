from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np

def train_logreg(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

def train_rf(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def train_gb(X, y):
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y)
    return model

def baseline_model(X, y):
    return (X["mean_speed"] < 5).astype(int)