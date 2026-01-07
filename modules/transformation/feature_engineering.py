import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# ==================================================
# STEP 4: FEATURE ENGINEERING
# ==================================================
# Goal: Create new meaningful features from existing
# data to improve insights and ML performance
# ==================================================


# --------------------------------------------------
# 1️⃣ ARITHMETIC FEATURES
# --------------------------------------------------
def add_sum_feature(df, col1, col2, new_col):
    """Creates Sum feature: col1 + col2"""
    df = df.copy()
    df[new_col] = df[col1] + df[col2]
    return df


def add_difference_feature(df, col1, col2, new_col):
    """Creates Difference feature: col1 - col2"""
    df = df.copy()
    df[new_col] = df[col1] - df[col2]
    return df


def add_product_feature(df, col1, col2, new_col):
    """Creates Product feature: col1 * col2"""
    df = df.copy()
    df[new_col] = df[col1] * df[col2]
    return df


def add_ratio_feature(df, col1, col2, new_col):
    """Creates Ratio feature: col1 / col2"""
    df = df.copy()
    df[new_col] = df[col1] / df[col2]
    return df


# --------------------------------------------------
# 2️⃣ POLYNOMIAL & INTERACTION FEATURES
# --------------------------------------------------
def polynomial_features(df, cols, degree=2, include_bias=False):
    """
    Generates polynomial and interaction features.
    """
    df = df.copy()
    poly = PolynomialFeatures(
        degree=degree,
        include_bias=include_bias
    )

    transformed = poly.fit_transform(df[cols])
    feature_names = poly.get_feature_names_out(cols)

    poly_df = pd.DataFrame(
        transformed,
        columns=feature_names,
        index=df.index
    )

    df = df.drop(columns=cols)
    df = pd.concat([df, poly_df], axis=1)

    return df


# --------------------------------------------------
# 3️⃣ DATE / TIME FEATURE EXTRACTION
# --------------------------------------------------
def extract_date_features(df, col):
    """
    Extracts temporal features from datetime column.
    """
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")

    df[f"{col}_year"] = df[col].dt.year
    df[f"{col}_month"] = df[col].dt.month
    df[f"{col}_day"] = df[col].dt.day
    df[f"{col}_weekday"] = df[col].dt.weekday
    df[f"{col}_quarter"] = df[col].dt.quarter

    return df


# --------------------------------------------------
# 4️⃣ ROLLING & LAG FEATURES (TIME SERIES)
# --------------------------------------------------
def rolling_mean(df, col, window, new_col):
    """
    Rolling mean feature.
    """
    df = df.copy()
    df[new_col] = df[col].rolling(window=window).mean()
    return df


def rolling_std(df, col, window, new_col):
    """
    Rolling standard deviation feature.
    """
    df = df.copy()
    df[new_col] = df[col].rolling(window=window).std()
    return df


def lag_feature(df, col, lag, new_col):
    """
    Creates lag feature.
    """
    df = df.copy()
    df[new_col] = df[col].shift(lag)
    return df


# --------------------------------------------------
# 5️⃣ AGGREGATED FEATURES
# --------------------------------------------------
def aggregated_feature(df, group_col, target_col, agg_func, new_col):
    """
    Creates aggregated feature using group statistics.
    """
    df = df.copy()
    agg_map = df.groupby(group_col)[target_col].transform(agg_func)
    df[new_col] = agg_map
    return df


# --------------------------------------------------
# 6️⃣ DOMAIN-SPECIFIC FEATURES
# --------------------------------------------------
def bmi_feature(df, weight_col, height_col, new_col="BMI"):
    """
    Calculates Body Mass Index.
    BMI = weight / height^2
    """
    df = df.copy()
    df[new_col] = df[weight_col] / (df[height_col] ** 2)
    return df


def profit_feature(df, revenue_col, cost_col, new_col="Profit"):
    """
    Calculates profit feature.
    Profit = Revenue - Cost
    """
    df = df.copy()
    df[new_col] = df[revenue_col] - df[cost_col]
    return df


# --------------------------------------------------
# 🔍 UTILITY FUNCTIONS
# --------------------------------------------------
def is_datetime_column(df, col):
    return pd.api.types.is_datetime64_any_dtype(df[col])


def is_numeric_column(df, col):
    return pd.api.types.is_numeric_dtype(df[col])
