import pandas as pd
import numpy as np

# ==================================================
# STEP 1: DATA TYPE CONVERSION
# ==================================================
# Goal: Convert columns into correct formats for
# analysis, visualization, and machine learning
# ==================================================


# --------------------------------------------------
# 1️⃣ STRING → NUMERIC
# --------------------------------------------------
def string_to_numeric(df, col, to_type="float"):
    """
    Converts string column to numeric.
    Supports float or int casting.
    Invalid values are coerced to NaN.
    """
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")

    if to_type == "int":
        df[col] = df[col].astype("Int64")  # nullable integer

    return df


# --------------------------------------------------
# 2️⃣ STRING → DATETIME
# --------------------------------------------------
def string_to_datetime(df, col, date_format=None, utc=False):
    """
    Converts string column to datetime.
    Supports custom format and timezone.
    """
    df = df.copy()
    df[col] = pd.to_datetime(
        df[col],
        format=date_format,
        errors="coerce",
        utc=utc
    )
    return df


# --------------------------------------------------
# 3️⃣ NUMERIC → STRING
# --------------------------------------------------
def numeric_to_string(df, col):
    """
    Converts numeric column to string.
    Useful for IDs or codes.
    """
    df = df.copy()
    df[col] = df[col].astype(str)
    return df


# --------------------------------------------------
# 4️⃣ BOOLEAN CONVERSION
# --------------------------------------------------
def to_boolean(df, col):
    """
    Converts column to boolean.
    Handles Yes/No, True/False, 0/1.
    """
    df = df.copy()

    df[col] = df[col].replace(
        {
            "Yes": True, "No": False,
            "yes": True, "no": False,
            "TRUE": True, "FALSE": False,
            1: True, 0: False
        }
    )

    df[col] = df[col].astype("boolean")
    return df


# --------------------------------------------------
# 5️⃣ CATEGORY CONVERSION
# --------------------------------------------------
def to_category(df, col):
    """
    Converts object column to pandas category dtype.
    Improves memory efficiency.
    """
    df = df.copy()
    df[col] = df[col].astype("category")
    return df


# --------------------------------------------------
# 🔍 UTILITY: COLUMN TYPE INSPECTION
# --------------------------------------------------
def get_column_dtype(df, col):
    """
    Returns the dtype of a column.
    Useful for UI validation.
    """
    return df[col].dtype


def is_numeric(df, col):
    return pd.api.types.is_numeric_dtype(df[col])


def is_categorical(df, col):
    return pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == "object"


def is_datetime(df, col):
    return pd.api.types.is_datetime64_any_dtype(df[col])
