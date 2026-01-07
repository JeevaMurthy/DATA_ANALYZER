import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ==================================================
# EDA – CORRELATION & RELATIONSHIP ANALYSIS
# ==================================================
# Covers:
# - Pearson, Spearman, Kendall correlation
# - Correlation matrix & heatmap data
# - Highly correlated feature detection
# - Multicollinearity (VIF)
# - Pairwise scatter relationships
# ==================================================


# --------------------------------------------------
# 1️⃣ CORRELATION MATRIX
# --------------------------------------------------
def correlation_matrix(df, method="pearson"):
    """
    Returns correlation matrix.
    method: 'pearson', 'spearman', 'kendall'
    """
    numeric_df = df.select_dtypes(include=np.number)
    return numeric_df.corr(method=method)


# --------------------------------------------------
# 2️⃣ HIGHLY CORRELATED FEATURES
# --------------------------------------------------
def highly_correlated_features(df, threshold=0.9, method="pearson"):
    """
    Detects pairs of highly correlated features.
    """
    corr_matrix = correlation_matrix(df, method).abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    high_corr = []

    for col in upper_triangle.columns:
        for row in upper_triangle.index:
            corr_value = upper_triangle.loc[row, col]
            if corr_value > threshold:
                high_corr.append({
                    "Feature 1": row,
                    "Feature 2": col,
                    "Correlation": round(corr_value, 4)
                })

    return pd.DataFrame(high_corr)


# --------------------------------------------------
# 3️⃣ MULTICOLLINEARITY CHECK (VIF)
# --------------------------------------------------
def vif_table(df):
    """
    Computes Variance Inflation Factor (VIF)
    for numeric features.
    """
    numeric_df = df.select_dtypes(include=np.number).dropna()

    vif_data = []
    for i in range(numeric_df.shape[1]):
        vif_data.append({
            "Feature": numeric_df.columns[i],
            "VIF": round(
                variance_inflation_factor(numeric_df.values, i), 4
            )
        })

    return pd.DataFrame(vif_data)


# --------------------------------------------------
# 4️⃣ PAIRWISE RELATIONSHIP DATA
# --------------------------------------------------
def scatter_data(df, x_col, y_col):
    """
    Returns data for scatter plot analysis.
    """
    return df[[x_col, y_col]].dropna()


# --------------------------------------------------
# 5️⃣ CORRELATION RANKING
# --------------------------------------------------
def correlation_with_target(df, target_col, method="pearson"):
    """
    Ranks features by correlation with target variable.
    """
    numeric_df = df.select_dtypes(include=np.number)

    if target_col not in numeric_df.columns:
        raise ValueError("Target column must be numeric")

    corr = numeric_df.corr(method=method)[target_col]
    return corr.drop(target_col).sort_values(ascending=False)


# --------------------------------------------------
# 🔍 UTILITY
# --------------------------------------------------
def numeric_columns(df):
    """Returns numeric columns"""
    return df.select_dtypes(include=np.number).columns.tolist()
