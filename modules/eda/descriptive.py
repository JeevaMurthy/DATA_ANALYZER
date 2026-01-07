import pandas as pd
import numpy as np

# ==================================================
# EDA – DESCRIPTIVE STATISTICS
# ==================================================
# Covers:
# Mean, Median, Mode
# Min, Max, Range
# Variance, Standard Deviation
# Quartiles, IQR
# Percentiles
# ==================================================


# --------------------------------------------------
# 1️⃣ BASIC STATISTICS
# --------------------------------------------------
def mean(df, cols):
    """Returns mean for selected numeric columns"""
    return df[cols].mean()


def median(df, cols):
    """Returns median for selected numeric columns"""
    return df[cols].median()


def mode(df, cols):
    """Returns mode for selected columns"""
    return df[cols].mode().iloc[0]


def minimum(df, cols):
    """Returns minimum values"""
    return df[cols].min()


def maximum(df, cols):
    """Returns maximum values"""
    return df[cols].max()


def value_range(df, cols):
    """Returns range (max - min)"""
    return df[cols].max() - df[cols].min()


# --------------------------------------------------
# 2️⃣ VARIABILITY MEASURES
# --------------------------------------------------
def variance(df, cols):
    """Returns variance"""
    return df[cols].var()


def standard_deviation(df, cols):
    """Returns standard deviation"""
    return df[cols].std()


# --------------------------------------------------
# 3️⃣ QUARTILES & IQR
# --------------------------------------------------
def quartiles(df, cols):
    """
    Returns Q1, Q2 (Median), Q3
    """
    q = df[cols].quantile([0.25, 0.5, 0.75])
    q.index = ["Q1", "Q2 (Median)", "Q3"]
    return q


def interquartile_range(df, cols):
    """Returns IQR (Q3 - Q1)"""
    q1 = df[cols].quantile(0.25)
    q3 = df[cols].quantile(0.75)
    return q3 - q1


# --------------------------------------------------
# 4️⃣ PERCENTILES
# --------------------------------------------------
def percentiles(df, cols, percent_list):
    """
    Returns selected percentiles
    Example percent_list = [10, 25, 50, 75, 90]
    """
    p = df[cols].quantile([p / 100 for p in percent_list])
    p.index = [f"P{p}" for p in percent_list]
    return p


# --------------------------------------------------
# 🔍 MASTER SUMMARY TABLE
# --------------------------------------------------
def descriptive_summary(df, cols):
    """
    Returns a single summary table with all major
    descriptive statistics.
    """
    summary = pd.DataFrame({
        "Mean": df[cols].mean(),
        "Median": df[cols].median(),
        "Min": df[cols].min(),
        "Max": df[cols].max(),
        "Range": df[cols].max() - df[cols].min(),
        "Variance": df[cols].var(),
        "Std Dev": df[cols].std(),
        "Q1": df[cols].quantile(0.25),
        "Q2 (Median)": df[cols].quantile(0.50),
        "Q3": df[cols].quantile(0.75),
        "IQR": df[cols].quantile(0.75) - df[cols].quantile(0.25)
    })

    return summary.round(4)


# --------------------------------------------------
# 🔍 UTILITY
# --------------------------------------------------
def numeric_columns(df):
    """Returns list of numeric columns"""
    return df.select_dtypes(include=np.number).columns.tolist()
