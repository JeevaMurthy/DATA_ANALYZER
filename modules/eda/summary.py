import pandas as pd
import numpy as np

# ==================================================
# EDA – SUMMARY & REPORTING
# ==================================================
# Covers:
# - Column-wise insights
# - Key pattern identification
# - Data quality score
# - EDA summary table
# - Auto-generated EDA report text
# ==================================================


# --------------------------------------------------
# 1️⃣ COLUMN-WISE INSIGHTS
# --------------------------------------------------
def column_insights(df):
    """
    Generates column-wise insights.
    """
    insights = []

    for col in df.columns:
        col_info = {
            "Column": col,
            "Data Type": str(df[col].dtype),
            "Missing %": round(df[col].isnull().mean() * 100, 2),
            "Unique Values": df[col].nunique(dropna=True)
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({
                "Mean": round(df[col].mean(), 4),
                "Std Dev": round(df[col].std(), 4),
                "Min": df[col].min(),
                "Max": df[col].max()
            })
        else:
            col_info.update({
                "Most Frequent": df[col].mode().iloc[0] if not df[col].mode().empty else None
            })

        insights.append(col_info)

    return pd.DataFrame(insights)


# --------------------------------------------------
# 2️⃣ KEY PATTERN IDENTIFICATION
# --------------------------------------------------
def key_patterns(df):
    """
    Identifies key patterns and data issues.
    """
    patterns = []

    # Missing data pattern
    high_missing = df.isnull().mean()
    high_missing = high_missing[high_missing > 0.3]

    if not high_missing.empty:
        patterns.append(
            f"High missing values detected in columns: {list(high_missing.index)}"
        )

    # Constant columns
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        patterns.append(
            f"Constant columns detected: {constant_cols}"
        )

    # High cardinality
    cat_cols = df.select_dtypes(exclude=np.number).columns
    high_card = [
        c for c in cat_cols if df[c].nunique(dropna=True) > 20
    ]
    if high_card:
        patterns.append(
            f"High-cardinality categorical columns: {high_card}"
        )

    if not patterns:
        patterns.append("No major data quality issues detected.")

    return patterns


# --------------------------------------------------
# 3️⃣ DATA QUALITY SCORE
# --------------------------------------------------
def data_quality_score(df):
    """
    Computes data quality score based on:
    - Missing values
    - Constant columns
    """
    missing_penalty = df.isnull().mean().mean() * 100
    constant_penalty = len(
        [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    ) * 2

    score = 100 - missing_penalty - constant_penalty
    return round(max(score, 0), 2)


# --------------------------------------------------
# 4️⃣ EDA SUMMARY TABLE
# --------------------------------------------------
def eda_summary_table(df):
    """
    Returns a compact EDA summary table.
    """
    return pd.DataFrame({
        "Rows": [df.shape[0]],
        "Columns": [df.shape[1]],
        "Total Missing Values": [df.isnull().sum().sum()],
        "Duplicate Rows": [df.duplicated().sum()],
        "Numeric Columns": [df.select_dtypes(include=np.number).shape[1]],
        "Categorical Columns": [df.select_dtypes(exclude=np.number).shape[1]],
        "Data Quality Score": [data_quality_score(df)]
    })


# --------------------------------------------------
# 5️⃣ AUTO-GENERATED EDA REPORT TEXT
# --------------------------------------------------
def generate_eda_report(df):
    """
    Generates natural-language EDA report text.
    """
    summary = eda_summary_table(df).iloc[0]
    patterns = key_patterns(df)

    report = f"""
EDA REPORT SUMMARY
------------------
Dataset contains {summary['Rows']} rows and {summary['Columns']} columns.

• Total missing values: {summary['Total Missing Values']}
• Duplicate rows: {summary['Duplicate Rows']}
• Numeric features: {summary['Numeric Columns']}
• Categorical features: {summary['Categorical Columns']}
• Data quality score: {summary['Data Quality Score']} / 100

KEY OBSERVATIONS:
"""
    for p in patterns:
        report += f"- {p}\n"

    report += """
RECOMMENDATION:
Proceed with feature selection and modeling after addressing the highlighted issues.
"""

    return report.strip()
