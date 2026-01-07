import pandas as pd
import numpy as np

# ==================================================
# EDA – MISSING DATA & QUALITY ANALYSIS
# ==================================================
# Covers:
# - Missing count per column
# - Missing percentage
# - Missing heatmap data
# - Missing vs non-missing comparison
# - Column completeness score
# ==================================================


# --------------------------------------------------
# 1️⃣ MISSING VALUE COUNT PER COLUMN
# --------------------------------------------------
def missing_count(df):
    """
    Returns missing value count for each column.
    """
    return df.isnull().sum()


# --------------------------------------------------
# 2️⃣ MISSING VALUE PERCENTAGE
# --------------------------------------------------
def missing_percentage(df):
    """
    Returns missing value percentage for each column.
    """
    return (df.isnull().mean() * 100).round(2)


# --------------------------------------------------
# 3️⃣ MISSING SUMMARY TABLE
# --------------------------------------------------
def missing_summary(df):
    """
    Returns a summary DataFrame with count and percentage.
    """
    summary = pd.DataFrame({
        "Missing Count": df.isnull().sum(),
        "Missing %": (df.isnull().mean() * 100).round(2)
    })

    return summary[summary["Missing Count"] > 0]


# --------------------------------------------------
# 4️⃣ MISSING HEATMAP DATA
# --------------------------------------------------
def missing_heatmap_data(df):
    """
    Returns boolean DataFrame for missing heatmap.
    True = Missing, False = Present
    """
    return df.isnull()


# --------------------------------------------------
# 5️⃣ MISSING vs NON-MISSING COMPARISON
# --------------------------------------------------
def missing_vs_non_missing(df, target_col):
    """
    Compares missing vs non-missing impact
    on a target column (mean comparison).
    """
    comparison = {}

    for col in df.columns:
        if col != target_col:
            missing_group = df[df[col].isnull()][target_col].mean()
            non_missing_group = df[df[col].notnull()][target_col].mean()

            comparison[col] = {
                "Target Mean (Missing)": missing_group,
                "Target Mean (Not Missing)": non_missing_group
            }

    return pd.DataFrame(comparison).T


# --------------------------------------------------
# 6️⃣ COLUMN COMPLETENESS SCORE
# --------------------------------------------------
def completeness_score(df):
    """
    Returns completeness score for each column.
    Completeness = % of non-missing values.
    """
    score = 100 - (df.isnull().mean() * 100)
    return score.round(2)


# --------------------------------------------------
# 🔍 OVERALL DATA QUALITY SCORE
# --------------------------------------------------
def overall_data_quality_score(df):
    """
    Computes overall data quality score
    based on completeness of all columns.
    """
    return round(completeness_score(df).mean(), 2)
