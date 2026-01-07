import pandas as pd
import numpy as np

# ==================================================
# EDA – CATEGORICAL DATA ANALYSIS
# ==================================================
# Covers:
# - Frequency counts
# - Category dominance
# - Rare category detection
# - Category imbalance
# - Cross-tabulation
# ==================================================


# --------------------------------------------------
# 1️⃣ FREQUENCY COUNTS
# --------------------------------------------------
def frequency_table(df, col):
    """
    Returns frequency count table for a categorical column.
    """
    freq = df[col].value_counts(dropna=False)
    return pd.DataFrame({
        col: freq.index.astype(str),
        "Count": freq.values
    })


# --------------------------------------------------
# 2️⃣ CATEGORY DOMINANCE ANALYSIS
# --------------------------------------------------
def category_dominance(df, col):
    """
    Returns dominance ratio of most frequent category.
    """
    freq = df[col].value_counts(dropna=False)
    dominance_ratio = freq.iloc[0] / freq.sum()

    return {
        "Most Frequent Category": str(freq.index[0]),
        "Dominance Ratio": round(dominance_ratio, 4)
    }


# --------------------------------------------------
# 3️⃣ RARE CATEGORY DETECTION
# --------------------------------------------------
def rare_categories(df, col, threshold=0.05):
    """
    Detects rare categories based on percentage threshold.
    threshold = minimum % share (default 5%)
    """
    freq = df[col].value_counts(normalize=True, dropna=False)
    rare = freq[freq < threshold]

    return pd.DataFrame({
        "Category": rare.index.astype(str),
        "Percentage": (rare.values * 100).round(2)
    })


# --------------------------------------------------
# 4️⃣ CATEGORY IMBALANCE CHECK
# --------------------------------------------------
def category_imbalance(df, col):
    """
    Returns imbalance ratio (max/min count).
    """
    freq = df[col].value_counts(dropna=False)
    imbalance_ratio = freq.max() / freq.min()

    return {
        "Max Count": int(freq.max()),
        "Min Count": int(freq.min()),
        "Imbalance Ratio": round(imbalance_ratio, 4)
    }


# --------------------------------------------------
# 5️⃣ CROSS-TABULATION
# --------------------------------------------------
def cross_tabulation(df, col1, col2, normalize=False):
    """
    Returns cross-tabulation between two categorical columns.
    """
    return pd.crosstab(
        df[col1],
        df[col2],
        normalize="index" if normalize else False
    )


# --------------------------------------------------
# 🔍 UTILITY
# --------------------------------------------------
def categorical_columns(df):
    """Returns list of categorical columns"""
    return df.select_dtypes(exclude=np.number).columns.tolist()
