import pandas as pd
import numpy as np

# ==================================================
# EDA – DATASET OVERVIEW & STRUCTURE
# ==================================================
# Covers:
# - Dataset shape
# - Column names & data types
# - Numeric vs categorical count
# - Memory usage
# - Index consistency
# - Constant column detection
# - High-cardinality detection
# ==================================================


# --------------------------------------------------
# 1️⃣ DATASET SHAPE
# --------------------------------------------------
def dataset_shape(df):
    """
    Returns number of rows and columns.
    """
    return {
        "rows": df.shape[0],
        "columns": df.shape[1]
    }


# --------------------------------------------------
# 2️⃣ COLUMN NAMES & DATA TYPES
# --------------------------------------------------
def column_dtypes(df):
    """
    Returns DataFrame with column names and dtypes.
    """
    return pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str)
    })


# --------------------------------------------------
# 3️⃣ NUMERIC VS CATEGORICAL COUNT
# --------------------------------------------------
def feature_type_count(df):
    """
    Returns count of numeric and categorical features.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    return {
        "numeric_features": len(numeric_cols),
        "categorical_features": len(categorical_cols),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols
    }


# --------------------------------------------------
# 4️⃣ MEMORY USAGE ANALYSIS
# --------------------------------------------------
def memory_usage(df):
    """
    Returns memory usage per column and total memory.
    """
    mem_usage = df.memory_usage(deep=True)
    mem_df = pd.DataFrame({
        "Column": mem_usage.index,
        "Memory (KB)": (mem_usage.values / 1024).round(2)
    })

    total_memory_mb = mem_usage.sum() / (1024 ** 2)

    return mem_df, round(total_memory_mb, 2)


# --------------------------------------------------
# 5️⃣ INDEX CONSISTENCY CHECK
# --------------------------------------------------
def index_consistency(df):
    """
    Checks if index is sequential and unique.
    """
    is_unique = df.index.is_unique
    is_monotonic = df.index.is_monotonic_increasing

    return {
        "is_unique": is_unique,
        "is_sequential": is_monotonic
    }


# --------------------------------------------------
# 6️⃣ CONSTANT COLUMN DETECTION
# --------------------------------------------------
def constant_columns(df):
    """
    Detects columns with only one unique value.
    """
    constant_cols = [
        col for col in df.columns
        if df[col].nunique(dropna=False) <= 1
    ]

    return constant_cols


# --------------------------------------------------
# 7️⃣ HIGH-CARDINALITY COLUMN DETECTION
# --------------------------------------------------
def high_cardinality_columns(df, threshold=20):
    """
    Detects categorical columns with high cardinality.
    threshold: number of unique values
    """
    high_card_cols = []

    for col in df.select_dtypes(exclude=np.number).columns:
        unique_count = df[col].nunique(dropna=True)
        if unique_count > threshold:
            high_card_cols.append({
                "Column": col,
                "Unique Values": unique_count
            })

    return pd.DataFrame(high_card_cols)


# --------------------------------------------------
# 🔍 MASTER OVERVIEW FUNCTION (OPTIONAL)
# --------------------------------------------------
def dataset_overview(df):
    """
    Returns all overview insights in one dictionary.
    Useful for summary generation.
    """
    shape = dataset_shape(df)
    feature_types = feature_type_count(df)
    mem_df, total_mem = memory_usage(df)

    return {
        "shape": shape,
        "feature_types": feature_types,
        "total_memory_mb": total_mem,
        "constant_columns": constant_columns(df),
    }
