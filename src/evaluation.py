from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc

def pr_auc_score(y_true, probs):
    precision, recall, _ = precision_recall_curve(y_true, probs)
    return auc(recall, precision)

def evaluate_model(y_true, y_pred, model_name="Model"):
    print(f"\n{model_name} performance:")
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1 score:", f1_score(y_true, y_pred, zero_division=0))