import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.decomposition import PCA

sns.set_style("whitegrid")

# ==================================================
# 🅰️ CLASSIFICATION VISUALIZATIONS (1–10)
# ==================================================

def confusion_matrix_plot(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig


def normalized_confusion_matrix_plot(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Normalized Confusion Matrix")
    return fig


def roc_curve_plot(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.legend()
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    return fig


def precision_recall_plot(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])

    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set_title("Precision–Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    return fig


def class_probability_distribution(y_prob):
    fig, ax = plt.subplots()
    ax.hist(y_prob[:, 1], bins=30)
    ax.set_title("Class Probability Distribution")
    ax.set_xlabel("Predicted Probability")
    return fig


def misclassification_plot(y_true, y_pred):
    errors = y_true != y_pred
    fig, ax = plt.subplots()
    ax.plot(errors.astype(int))
    ax.set_title("Misclassification Plot")
    ax.set_ylabel("Error (1 = Incorrect)")
    return fig


# ==================================================
# 🅱️ REGRESSION VISUALIZATIONS (11–18)
# ==================================================

def actual_vs_predicted(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred)
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            linestyle="--")
    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    return fig


def residual_plot(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals)
    ax.axhline(0, linestyle="--")
    ax.set_title("Residual Plot")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    return fig


def residual_distribution(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title("Residual Distribution")
    return fig


def residual_qq_plot(y_true, y_pred):
    from scipy import stats
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    stats.probplot(residuals, plot=ax)
    ax.set_title("Residual Q-Q Plot")
    return fig


# ==================================================
# 🅲 FEATURE IMPORTANCE & INTERPRETABILITY (19–24)
# ==================================================

def feature_importance_plot(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=importances, y=feature_names, ax=ax)
    ax.set_title("Feature Importance")
    return fig


def permutation_importance_plot(importances, feature_names):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=importances, y=feature_names, ax=ax)
    ax.set_title("Permutation Importance")
    return fig


def partial_dependence_placeholder():
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "PDP requires sklearn.inspection", ha="center")
    ax.set_title("Partial Dependence Plot (Conceptual)")
    return fig


def ice_plot_placeholder():
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "ICE Plot (Conceptual)", ha="center")
    ax.set_title("ICE Plot")
    return fig


# ==================================================
# 🅳 DIMENSIONALITY REDUCTION & CLUSTERING (25–30)
# ==================================================

def pca_explained_variance(X):
    pca = PCA()
    pca.fit(X)

    fig, ax = plt.subplots()
    ax.plot(np.cumsum(pca.explained_variance_ratio_))
    ax.set_title("PCA Explained Variance")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance")
    return fig


def pca_scatter(X, labels=None):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    ax.set_title("PCA Scatter Plot")
    return fig


def cluster_assignment_plot(X, labels):
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=labels)
    ax.set_title("Cluster Assignment Plot")
    return fig


def elbow_method_plot(X, k_range=range(1, 11)):
    from sklearn.cluster import KMeans
    inertias = []

    for k in k_range:
        km = KMeans(n_clusters=k)
        km.fit(X)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots()
    ax.plot(list(k_range), inertias)
    ax.set_title("Elbow Method")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    return fig


# ==================================================
# 🅴 TRAINING PROCESS VISUALS (31–35)
# ==================================================

def learning_curve_placeholder():
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "Learning Curve (Requires CV)", ha="center")
    ax.set_title("Learning Curve")
    return fig


def model_comparison_bar(df, metric):
    fig, ax = plt.subplots()
    sns.barplot(x="Model", y=metric, data=df, ax=ax)
    ax.set_title(f"Model Comparison – {metric}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    return fig
