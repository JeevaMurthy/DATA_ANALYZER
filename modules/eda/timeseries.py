import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf

# ==================================================
# EDA – TIME-SERIES ANALYSIS
# ==================================================
# Covers:
# - Trend analysis
# - Seasonality hints
# - Rolling mean & variance
# - Lag analysis
# - Autocorrelation
# ==================================================


# --------------------------------------------------
# 1️⃣ ENSURE DATETIME INDEX
# --------------------------------------------------
def ensure_datetime(df, date_col):
    """
    Converts a column to datetime and sets as index.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.set_index(date_col).sort_index()
    return df


# --------------------------------------------------
# 2️⃣ TREND DATA
# --------------------------------------------------
def trend_data(df, target_col):
    """
    Returns target values over time.
    """
    return df[target_col].dropna()


# --------------------------------------------------
# 3️⃣ SEASONALITY HINT
# --------------------------------------------------
def seasonality_hint(df):
    """
    Provides conceptual seasonality indication.
    """
    if df.index.inferred_freq:
        return f"Detected frequency: {df.index.inferred_freq}"
    else:
        return "No clear seasonality detected"


# --------------------------------------------------
# 4️⃣ ROLLING MEAN
# --------------------------------------------------
def rolling_mean(df, target_col, window=7):
    """
    Returns rolling mean values.
    """
    return df[target_col].rolling(window=window).mean()


# --------------------------------------------------
# 5️⃣ ROLLING VARIANCE
# --------------------------------------------------
def rolling_variance(df, target_col, window=7):
    """
    Returns rolling variance values.
    """
    return df[target_col].rolling(window=window).var()


# --------------------------------------------------
# 6️⃣ LAG FEATURES
# --------------------------------------------------
def lag_features(df, target_col, lag=1):
    """
    Returns lagged series.
    """
    return df[target_col].shift(lag)


# --------------------------------------------------
# 7️⃣ AUTOCORRELATION
# --------------------------------------------------
def autocorrelation(df, target_col, nlags=20):
    """
    Returns autocorrelation values.
    """
    data = df[target_col].dropna()
    return acf(data, nlags=nlags)


# --------------------------------------------------
# 🔍 UTILITY
# --------------------------------------------------
def has_datetime_column(df):
    """
    Checks if dataframe contains datetime columns.
    """
    return any(pd.api.types.is_datetime64_any_dtype(df[col]) for col in df.columns)
