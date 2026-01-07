import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ==================================================
# EDA – ADVANCED / ML-READY ANALYSIS
# ==================================================
# Covers:
# - Feature importance (model-based)
# - Mutual information
# - Covariance analysis
# - Noise detection
# - Signal-to-noise estimation
# ==================================================


# --------------------------------------------------
# 1️⃣ COVARIANCE MATRIX
# --------------------------------------------------
def covariance_matrix(df):
    """
    Returns covariance matrix for numeric features.
    """
    numeric_df = df.select_dtypes(include=np.number)
    return numeric_df.cov()


# --------------------------------------------------
# 2️⃣ MUTUAL INFORMATION (AUTOMATIC TYPE DETECTION)
# --------------------------------------------------
def mutual_information(df, target_col):
    """
    Computes mutual information between features and target.
    Automatically detects classification vs regression.
    """
    numeric_df = df.select_dtypes(include=np.number).dropna()

    X = numeric_df.drop(columns=[target_col], errors="ignore")
    y = numeric_df[target_col]

    if y.nunique() <= 20:
        mi = mutual_info_classif(X, y, random_state=42)
    else:
        mi = mutual_info_regression(X, y, random_state=42)

    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


# --------------------------------------------------
# 3️⃣ FEATURE IMPORTANCE (MODEL-BASED)
# --------------------------------------------------
def feature_importance(df, target_col):
    """
    Computes feature importance using Random Forest.
    """
    numeric_df = df.select_dtypes(include=np.number).dropna()

    X = numeric_df.drop(columns=[target_col], errors="ignore")
    y = numeric_df[target_col]

    if y.nunique() <= 20:
        model = RandomForestClassifier(
            n_estimators=100, random_state=42
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100, random_state=42
        )

    model.fit(X, y)

    return pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)


# --------------------------------------------------
# 4️⃣ NOISE DETECTION (VARIANCE-BASED)
# --------------------------------------------------
def noise_detection(df, threshold=0.01):
    """
    Detects low-variance (noisy) features.
    """
    numeric_df = df.select_dtypes(include=np.number)
    variances = numeric_df.var()

    noisy_features = variances[variances < threshold]

    return pd.DataFrame({
        "Feature": noisy_features.index,
        "Variance": noisy_features.values
    })


# --------------------------------------------------
# 5️⃣ SIGNAL-TO-NOISE RATIO
# --------------------------------------------------
def signal_to_noise(df):
    """
    Computes signal-to-noise ratio for numeric features.
    SNR = Mean / Standard Deviation
    """
    numeric_df = df.select_dtypes(include=np.number)

    snr = numeric_df.mean() / numeric_df.std()

    return snr.replace([np.inf, -np.inf], np.nan).round(4)
