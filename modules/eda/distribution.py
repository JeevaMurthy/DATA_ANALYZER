import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, probplot

# ==================================================
# EDA – DISTRIBUTION & SPREAD ANALYSIS
# ==================================================
# Covers:
# Histogram, KDE (conceptual)
# Box & Violin plot statistics
# Skewness, Kurtosis
# Normality check (conceptual)
# CDF, Q-Q plot
# ==================================================


# --------------------------------------------------
# 1️⃣ HISTOGRAM DATA
# --------------------------------------------------
def histogram_data(df, col, bins=10):
    """
    Returns histogram counts and bin edges.
    Useful for plotting histograms.
    """
    data = df[col].dropna()
    counts, bin_edges = np.histogram(data, bins=bins)
    return counts, bin_edges


# --------------------------------------------------
# 2️⃣ DENSITY / KDE (DATA ONLY)
# --------------------------------------------------
def kde_data(df, col):
    """
    Returns data for KDE plotting.
    Actual plotting handled in UI.
    """
    return df[col].dropna()


# --------------------------------------------------
# 3️⃣ BOXPLOT STATISTICS
# --------------------------------------------------
def boxplot_stats(df, col):
    """
    Returns boxplot statistics.
    """
    data = df[col].dropna()
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    return {
        "Q1": q1,
        "Median": q2,
        "Q3": q3,
        "IQR": iqr,
        "Lower Fence": lower_fence,
        "Upper Fence": upper_fence
    }


# --------------------------------------------------
# 4️⃣ VIOLIN PLOT DATA
# --------------------------------------------------
def violin_data(df, col):
    """
    Returns data for violin plot.
    """
    return df[col].dropna()


# --------------------------------------------------
# 5️⃣ SKEWNESS
# --------------------------------------------------
def skewness(df, col):
    """
    Returns skewness of the distribution.
    """
    return skew(df[col].dropna())


# --------------------------------------------------
# 6️⃣ KURTOSIS
# --------------------------------------------------
def kurtosis_value(df, col):
    """
    Returns kurtosis (tailedness).
    """
    return kurtosis(df[col].dropna())


# --------------------------------------------------
# 7️⃣ NORMALITY CHECK (CONCEPTUAL)
# --------------------------------------------------
def normality_hint(df, col):
    """
    Returns conceptual normality assessment
    based on skewness & kurtosis.
    """
    s = skewness(df, col)
    k = kurtosis_value(df, col)

    if abs(s) < 0.5 and abs(k) < 1:
        return "Approximately Normal"
    elif abs(s) < 1:
        return "Moderately Skewed"
    else:
        return "Highly Skewed"


# --------------------------------------------------
# 8️⃣ CDF (CUMULATIVE DISTRIBUTION FUNCTION)
# --------------------------------------------------
def cdf_data(df, col):
    """
    Returns x and y values for CDF plotting.
    """
    data = np.sort(df[col].dropna())
    y = np.arange(1, len(data) + 1) / len(data)
    return data, y


# --------------------------------------------------
# 9️⃣ Q-Q PLOT DATA
# --------------------------------------------------
def qq_plot_data(df, col):
    """
    Returns theoretical and sample quantiles
    for Q-Q plot.
    """
    data = df[col].dropna()
    theoretical, sample = probplot(data, dist="norm")[:2]
    return theoretical, sample


# --------------------------------------------------
# 🔍 UTILITY
# --------------------------------------------------
def is_numeric_column(df, col):
    """Checks if column is numeric"""
    return pd.api.types.is_numeric_dtype(df[col])
