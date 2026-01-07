import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

# ==================================================
# STEP 5: BINNING / DISCRETIZATION
# ==================================================
# Goal: Convert continuous numeric data into
# categorical bins for pattern analysis & reporting
# ==================================================


# --------------------------------------------------
# 1️⃣ EQUAL-WIDTH BINNING
# --------------------------------------------------
def equal_width_binning(df, col, bins, labels=None):
    """
    Splits data into bins of equal width.
    """
    df = df.copy()
    df[col] = pd.cut(df[col], bins=bins, labels=labels)
    return df


# --------------------------------------------------
# 2️⃣ EQUAL-FREQUENCY (QUANTILE) BINNING
# --------------------------------------------------
def quantile_binning(df, col, q, labels=None):
    """
    Splits data so each bin has equal number of records.
    """
    df = df.copy()
    df[col] = pd.qcut(df[col], q=q, labels=labels, duplicates="drop")
    return df


# --------------------------------------------------
# 3️⃣ CUSTOM BINNING
# --------------------------------------------------
def custom_binning(df, col, bin_edges, labels):
    """
    User-defined bin ranges.
    Example:
    bin_edges = [0, 18, 35, 60, 100]
    labels = ["Child", "Young", "Adult", "Senior"]
    """
    df = df.copy()
    df[col] = pd.cut(df[col], bins=bin_edges, labels=labels)
    return df


# --------------------------------------------------
# 4️⃣ K-MEANS BINNING
# --------------------------------------------------
def kmeans_binning(df, col, k):
    """
    Uses K-Means clustering to create bins.
    """
    df = df.copy()
    values = df[[col]].dropna()

    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(values)

    df.loc[values.index, f"{col}_kmeans_bin"] = clusters
    return df


# --------------------------------------------------
# 5️⃣ DECISION TREE BINNING (SUPERVISED)
# --------------------------------------------------
def decision_tree_binning(df, feature_col, target_col, max_depth=3):
    """
    Supervised discretization using decision tree splits.
    """
    df = df.copy()
    valid = df[[feature_col, target_col]].dropna()

    X = valid[[feature_col]]
    y = valid[target_col]

    tree = DecisionTreeClassifier(max_depth=max_depth)
    tree.fit(X, y)

    bins = tree.apply(X)
    df.loc[valid.index, f"{feature_col}_tree_bin"] = bins

    return df


# --------------------------------------------------
# 🔍 UTILITY FUNCTIONS
# --------------------------------------------------
def is_numeric_column(df, col):
    return pd.api.types.is_numeric_dtype(df[col])


def suggest_bins(df, col):
    """
    Suggests number of bins using Sturges' rule.
    """
    n = df[col].dropna().shape[0]
    return int(np.ceil(np.log2(n) + 1))
