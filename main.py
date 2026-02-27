from src.models import train_logistic_regression, train_random_forest
from src.baseline import rule_based_baseline
from src.evaluation import evaluate_model, pr_auc_score
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Load processed dataset
data = pd.read_csv("data/processed/segments_with_labels.csv")

features = ["mean_speed", "speed_std", "turning_rate"]

X = data[features]
y = data["label"]
groups = data["MMSI"]

# Vessel-level split
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

train = data.iloc[train_idx]
test = data.iloc[test_idx]

# Baseline
baseline_preds = rule_based_baseline(test)
evaluate_model(test["label"], baseline_preds, model_name="Baseline")

# Logistic Regression
model = train_logistic_regression(train, features)
probs = model.predict_proba(test[features])[:, 1]
for t in [0.3, 0.5, 0.7, 0.8, 0.9]:
    preds = (probs > t).astype(int)
    print(f"\nThreshold: {t}")
    evaluate_model(test["label"], preds, model_name="Logistic Regression")

print("PR-AUC:", pr_auc_score(test["label"], probs))

# Random Forest
rf_model = train_random_forest(train, features)
rf_probs = rf_model.predict_proba(test[features])[:, 1]

for t in [0.3, 0.5, 0.7, 0.8, 0.9]:
    rf_preds = (rf_probs > t).astype(int)
    print(f"\nRandom Forest Threshold: {t}")
    evaluate_model(test["label"], rf_preds, model_name="Random Forest")

print("Random Forest PR-AUC:", pr_auc_score(test["label"], rf_probs))
