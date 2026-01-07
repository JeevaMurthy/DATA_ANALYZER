import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    Normalizer,
    PowerTransformer
)

# ==================================================
# STEP 3: SCALING & NORMALIZATION
# ==================================================
# Goal: Bring numeric features to comparable scales
# to improve ML performance and visualization clarity
# ==================================================


# --------------------------------------------------
# 1️⃣ MIN–MAX SCALING
# --------------------------------------------------
def minmax_scaling(df, col):
    """
    Scales values to range [0, 1].
    Suitable for bounded numeric data and NN models.
    """
    df = df.copy()
    scaler = MinMaxScaler()
    df[[col]] = scaler.fit_transform(df[[col]])
    return df


# --------------------------------------------------
# 2️⃣ STANDARDIZATION (Z-SCORE)
# --------------------------------------------------
def standard_scaling(df, col):
    """
    Centers data to mean = 0 and std = 1.
    Used in linear models, SVM, KNN.
    """
    df = df.copy()
    scaler = StandardScaler()
    df[[col]] = scaler.fit_transform(df[[col]])
    return df


# --------------------------------------------------
# 3️⃣ ROBUST SCALING
# --------------------------------------------------
def robust_scaling(df, col):
    """
    Uses median and IQR.
    Robust to outliers.
    """
    df = df.copy()
    scaler = RobustScaler()
    df[[col]] = scaler.fit_transform(df[[col]])
    return df


# --------------------------------------------------
# 4️⃣ UNIT VECTOR SCALING (NORMALIZATION)
# --------------------------------------------------
def unit_vector_scaling(df, col):
    """
    Normalizes values to unit length.
    Used in cosine similarity–based models.
    """
    df = df.copy()
    normalizer = Normalizer()
    df[[col]] = normalizer.fit_transform(df[[col]])
    return df


# --------------------------------------------------
# 5️⃣ LOG TRANSFORMATION
# --------------------------------------------------
def log_transform(df, col):
    """
    Applies log(1 + x) transformation.
    Reduces right skewness.
    """
    df = df.copy()
    df[col] = np.log1p(df[col])
    return df


# --------------------------------------------------
# 6️⃣ BOX–COX TRANSFORMATION
# --------------------------------------------------
def boxcox_transform(df, col):
    """
    Power transformation for strictly positive data.
    Makes distribution more normal.
    """
    df = df.copy()
    transformer = PowerTransformer(method="box-cox")
    df[[col]] = transformer.fit_transform(df[[col]])
    return df


# --------------------------------------------------
# 7️⃣ YEO–JOHNSON TRANSFORMATION
# --------------------------------------------------
def yeojohnson_transform(df, col):
    """
    Power transformation supporting negative values.
    """
    df = df.copy()
    transformer = PowerTransformer(method="yeo-johnson")
    df[[col]] = transformer.fit_transform(df[[col]])
    return df


# --------------------------------------------------
# 🔍 UTILITY FUNCTIONS
# --------------------------------------------------
def is_numeric_column(df, col):
    """Checks if column is numeric"""
    return pd.api.types.is_numeric_dtype(df[col])


def detect_skewness(df, col):
    """
    Returns skewness value of a column.
    Used to suggest log / power transform.
    """
    return df[col].dropna().skew()
