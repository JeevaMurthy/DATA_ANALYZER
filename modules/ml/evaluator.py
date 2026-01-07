import numpy as np
import pandas as pd

# ---------------- CLASSIFICATION METRICS ----------------
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ---------------- REGRESSION METRICS ----------------
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------- UNSUPERVISED METRICS ----------------
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)


# ==================================================
# 🔍 CLASSIFICATION EVALUATION
# ==================================================
def evaluate_classification(y_true, y_pred, y_prob=None):
    """
    Evaluates classification models.
    """
    results = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Confusion Matrix": confusion_matrix(y_true, y_pred)
    }

    if y_prob is not None and len(np.unique(y_true)) == 2:
        results["ROC-AUC"] = roc_auc_score(y_true, y_prob[:, 1])

    results["Classification Report"] = classification_report(
        y_true, y_pred, output_dict=True
    )

    return results


# ==================================================
# 🔍 REGRESSION EVALUATION
# ==================================================
def evaluate_regression(y_true, y_pred):
    """
    Evaluates regression models.
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R² Score": r2_score(y_true, y_pred)
    }


# ==================================================
# 🔍 UNSUPERVISED EVALUATION
# ==================================================
def evaluate_unsupervised(X, labels):
    """
    Evaluates clustering / unsupervised models.
    """
    unique_labels = len(set(labels))

    if unique_labels <= 1:
        return {"Warning": "Clustering produced a single cluster"}

    return {
        "Silhouette Score": silhouette_score(X, labels),
        "Calinski-Harabasz Score": calinski_harabasz_score(X, labels),
        "Davies-Bouldin Score": davies_bouldin_score(X, labels),
        "Clusters Found": unique_labels
    }


# ==================================================
# 📊 MODEL COMPARISON TABLE
# ==================================================
def build_comparison_table(results, problem_type):
    """
    Converts evaluation results into a DataFrame.
    """
    records = []

    for model_name, metrics in results.items():
        row = {"Model": model_name}

        if problem_type in ["classification", "regression"]:
            for k, v in metrics.items():
                if not isinstance(v, (dict, list, np.ndarray)):
                    row[k] = round(v, 4)

        elif problem_type == "unsupervised":
            row.update(metrics)

        records.append(row)

    return pd.DataFrame(records)
