import pandas as pd
import numpy as np

# ==================================================
# EDA – TARGET VARIABLE ANALYSIS
# ==================================================
# Covers:
# - Target distribution
# - Class imbalance detection
# - Target vs feature comparison
# - Mean target per category
# - Target variance
# - Target correlation ranking
# ==================================================


# --------------------------------------------------
# 1️⃣ TARGET DISTRIBUTION
# --------------------------------------------------
def target_distribution(df, target_col):
    """
    Returns distribution of target variable.
    Useful for both classification & regression.
    """
    return df[target_col].value_counts(dropna=False)


# --------------------------------------------------
# 2️⃣ CLASS IMBALANCE DETECTION (CLASSIFICATION)
# --------------------------------------------------
def class_imbalance(df, target_col):
    """
    Returns class imbalance ratio.
    """
    counts = df[target_col].value_counts(dropna=False)
    imbalance_ratio = counts.max() / counts.min()

    return {
        "Class Counts": counts.to_dict(),
        "Imbalance Ratio": round(imbalance_ratio, 4)
    }


# --------------------------------------------------
# 3️⃣ TARGET VARIANCE (REGRESSION)
# --------------------------------------------------
def target_variance(df, target_col):
    """
    Returns variance of target variable.
    """
    return round(df[target_col].var(), 4)


# --------------------------------------------------
# 4️⃣ TARGET vs FEATURE COMPARISON (NUMERIC FEATURE)
# --------------------------------------------------
def target_vs_numeric_feature(df, target_col, feature_col):
    """
    Returns correlation between target and numeric feature.
    """
    if not pd.api.types.is_numeric_dtype(df[feature_col]):
        raise ValueError("Feature column must be numeric")

    return round(df[[target_col, feature_col]].corr().iloc[0, 1], 4)


# --------------------------------------------------
# 5️⃣ MEAN TARGET PER CATEGORY
# --------------------------------------------------
def mean_target_per_category(df, target_col, category_col):
    """
    Returns mean target value for each category.
    """
    return df.groupby(category_col)[target_col].mean().round(4)


# --------------------------------------------------
# 6️⃣ TARGET CORRELATION RANKING
# --------------------------------------------------
def target_correlation_ranking(df, target_col, method="pearson"):
    """
    Ranks numeric features by correlation with target.
    """
    numeric_df = df.select_dtypes(include=np.number)

    if target_col not in numeric_df.columns:
        raise ValueError("Target column must be numeric")

    corr = numeric_df.corr(method=method)[target_col]
    return corr.drop(target_col).sort_values(ascending=False)


# --------------------------------------------------
# 🔍 UTILITY
# --------------------------------------------------
def is_classification_target(df, target_col):
    """
    Determines if target is likely classification.
    """
    unique_values = df[target_col].nunique()
    return unique_values <= 20
