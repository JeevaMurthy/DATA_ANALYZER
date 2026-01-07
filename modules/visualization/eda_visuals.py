import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ==================================================
# EDA VISUALIZATION MODULE
# ==================================================
# Covers:
# - Distribution plots
# - Categorical plots
# - Correlation plots
# - Missing value plots
# - Outlier plots
# - Time-series plots
# ==================================================


# --------------------------------------------------
# 🅰️ DISTRIBUTION VISUALS
# --------------------------------------------------
def histogram(df, col, bins=30):
    fig, ax = plt.subplots()
    ax.hist(df[col].dropna(), bins=bins)
    ax.set_title(f"Histogram of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    return fig


def kde_plot(df, col):
    fig, ax = plt.subplots()
    sns.kdeplot(df[col].dropna(), ax=ax, fill=True)
    ax.set_title(f"KDE Plot of {col}")
    return fig


def box_plot(df, col):
    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], ax=ax)
    ax.set_title(f"Box Plot of {col}")
    return fig


def violin_plot(df, col):
    fig, ax = plt.subplots()
    sns.violinplot(x=df[col], ax=ax)
    ax.set_title(f"Violin Plot of {col}")
    return fig


def qq_plot(df, col):
    from scipy import stats
    fig, ax = plt.subplots()
    stats.probplot(df[col].dropna(), dist="norm", plot=ax)
    ax.set_title(f"Q-Q Plot of {col}")
    return fig


def cdf_plot(df, col):
    data = np.sort(df[col].dropna())
    y = np.arange(1, len(data) + 1) / len(data)

    fig, ax = plt.subplots()
    ax.plot(data, y)
    ax.set_title(f"CDF of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("CDF")
    return fig


# --------------------------------------------------
# 🅱️ CATEGORICAL VISUALS
# --------------------------------------------------
def frequency_bar(df, col):
    freq = df[col].value_counts()
    fig, ax = plt.subplots()
    freq.plot(kind="bar", ax=ax)
    ax.set_title(f"Frequency Bar Chart of {col}")
    ax.set_ylabel("Count")
    return fig


def percentage_bar(df, col):
    freq = df[col].value_counts(normalize=True) * 100
    fig, ax = plt.subplots()
    freq.plot(kind="bar", ax=ax)
    ax.set_title(f"Percentage Distribution of {col}")
    ax.set_ylabel("Percentage")
    return fig


def dominant_category_plot(df, col):
    freq = df[col].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=freq.index[:5], y=freq.values[:5], ax=ax)
    ax.set_title(f"Top Categories in {col}")
    return fig


def crosstab_heatmap(df, col1, col2):
    ct = pd.crosstab(df[col1], df[col2])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Cross-tab Heatmap: {col1} vs {col2}")
    return fig


# --------------------------------------------------
# 🅲 CORRELATION & RELATIONSHIP VISUALS
# --------------------------------------------------
def correlation_heatmap(df, method="pearson"):
    corr = df.select_dtypes(include=np.number).corr(method=method)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title(f"{method.title()} Correlation Heatmap")
    return fig


def scatter_plot(df, x_col, y_col):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
    ax.set_title(f"{x_col} vs {y_col}")
    return fig


def pair_plot(df, cols):
    return sns.pairplot(df[cols].dropna())


def hexbin_plot(df, x_col, y_col):
    fig, ax = plt.subplots()
    ax.hexbin(df[x_col], df[y_col], gridsize=30)
    ax.set_title(f"Hexbin Plot: {x_col} vs {y_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    return fig


# --------------------------------------------------
# 🅳 MISSING VALUE VISUALS
# --------------------------------------------------
def missing_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False, ax=ax)
    ax.set_title("Missing Value Heatmap")
    return fig


def missing_percentage_bar(df):
    miss_pct = df.isnull().mean() * 100
    fig, ax = plt.subplots()
    miss_pct.plot(kind="bar", ax=ax)
    ax.set_title("Missing Value Percentage per Column")
    ax.set_ylabel("Percentage")
    return fig


# --------------------------------------------------
# 🅴 OUTLIER VISUALS
# --------------------------------------------------
def boxplot_outliers(df, col):
    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], ax=ax)
    ax.set_title(f"Outlier Detection – {col}")
    return fig


def zscore_outlier_plot(df, col, threshold=3):
    from scipy.stats import zscore
    z = np.abs(zscore(df[col].dropna()))

    fig, ax = plt.subplots()
    ax.plot(z)
    ax.axhline(threshold, linestyle="--")
    ax.set_title(f"Z-score Outlier Plot – {col}")
    ax.set_ylabel("Z-score")
    return fig


# --------------------------------------------------
# 🅵 TIME-SERIES VISUALS
# --------------------------------------------------
def timeseries_line(df, date_col, target_col):
    fig, ax = plt.subplots()
    ax.plot(df[date_col], df[target_col])
    ax.set_title(f"Time-Series: {target_col}")
    ax.set_xlabel("Time")
    ax.set_ylabel(target_col)
    return fig


def rolling_mean_plot(df, date_col, target_col, window=7):
    fig, ax = plt.subplots()
    ax.plot(df[date_col], df[target_col], label="Original")
    ax.plot(
        df[date_col],
        df[target_col].rolling(window).mean(),
        label=f"Rolling Mean ({window})"
    )
    ax.legend()
    ax.set_title("Rolling Mean Plot")
    return fig


def lag_plot(df, col, lag=1):
    fig, ax = plt.subplots()
    ax.scatter(df[col][lag:], df[col][:-lag])
    ax.set_title(f"Lag Plot ({lag}) – {col}")
    ax.set_xlabel(f"{col}(t)")
    ax.set_ylabel(f"{col}(t-{lag})")
    return fig
