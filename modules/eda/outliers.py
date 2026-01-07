import numpy as np
import pandas as pd
from scipy.stats import zscore

# ==================================================
# EDA – OUTLIER & ANOMALY ANALYSIS
# ==================================================
# Covers:
# - Boxplot-based outliers
# - IQR-based detection
# - Z-score detection
# - Extreme value analysis
# - Mean vs Median comparison
# - Outlier count per column
# ==================================================


# --------------------------------------------------
# 1️⃣ BOXPLOT-BASED OUTLIER DETECTION
# --------------------------------------------------
def boxplot_outliers(df, col):
    """
    Detects outliers using boxplot fences.
    """
    data = df[col].dropna()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    outliers = df[(df[col] < lower_fence) | (df[col] > upper_fence)]
    return outliers, lower_fence, upper_fence


# --------------------------------------------------
# 2️⃣ IQR-BASED OUTLIER DETECTION
# --------------------------------------------------
def iqr_outliers(df, col):
    """
    Detects outliers using IQR method.
    """
    data = df[col].dropna()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    outliers = df[
        (df[col] < (q1 - 1.5 * iqr)) |
        (df[col] > (q3 + 1.5 * iqr))
    ]

    return outliers


# --------------------------------------------------
# 3️⃣ Z-SCORE OUTLIER DETECTION
# --------------------------------------------------
def zscore_outliers(df, col, threshold=3):
    """
    Detects outliers using Z-score method.
    """
    data = df[col].dropna()
    z_scores = np.abs(zscore(data))

    outlier_indices = data.index[z_scores > threshold]
    return df.loc[outlier_indices]


# --------------------------------------------------
# 4️⃣ EXTREME VALUE ANALYSIS
# --------------------------------------------------
def extreme_values(df, col):
    """
    Returns extreme min and max values.
    """
    return {
        "Min Value": df[col].min(),
        "Max Value": df[col].max()
    }


# --------------------------------------------------
# 5️⃣ MEAN VS MEDIAN COMPARISON
# --------------------------------------------------
def mean_vs_median(df, col):
    """
    Compares mean and median to indicate skew/outliers.
    """
    mean_val = df[col].mean()
    median_val = df[col].median()

    return {
        "Mean": round(mean_val, 4),
        "Median": round(median_val, 4),
        "Difference": round(mean_val - median_val, 4)
    }


# --------------------------------------------------
# 6️⃣ OUTLIER COUNT
# --------------------------------------------------
def outlier_count(df, col, method="iqr"):
    """
    Returns count of outliers based on selected method.
    method: 'iqr' or 'zscore'
    """
    if method == "iqr":
        outliers = iqr_outliers(df, col)
    elif method == "zscore":
        outliers = zscore_outliers(df, col)
    else:
        raise ValueError("Unsupported method")

    return len(outliers)


# --------------------------------------------------
# 🔍 UTILITY
# --------------------------------------------------
def is_numeric_column(df, col):
    """Checks if column is numeric"""
    return pd.api.types.is_numeric_dtype(df[col])
