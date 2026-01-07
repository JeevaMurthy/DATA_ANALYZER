import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# ==================================================
# STEP 2: ENCODING CATEGORICAL DATA
# ==================================================
# Goal: Convert categorical/text data into numeric
# form suitable for ML models and analytics
# ==================================================


# --------------------------------------------------
# 1️⃣ LABEL ENCODING
# --------------------------------------------------
def label_encoding(df, col):
    """
    Encodes categorical values as integers.
    Suitable for ordered or semi-ordered data.
    """
    df = df.copy()
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    return df


# --------------------------------------------------
# 2️⃣ ONE-HOT ENCODING
# --------------------------------------------------
def one_hot_encoding(df, col, drop_first=False):
    """
    Performs one-hot encoding.
    Creates binary columns for each category.
    """
    df = df.copy()
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)
    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    return df


# --------------------------------------------------
# 3️⃣ ORDINAL ENCODING
# --------------------------------------------------
def ordinal_encoding(df, col, categories):
    """
    Encodes categories while preserving order.
    Categories must be provided in correct order.
    Example: ["Low", "Medium", "High"]
    """
    df = df.copy()
    encoder = OrdinalEncoder(categories=[categories])
    df[[col]] = encoder.fit_transform(df[[col]])
    return df


# --------------------------------------------------
# 4️⃣ BINARY ENCODING
# --------------------------------------------------
def binary_encoding(df, col):
    """
    Encodes categories using binary representation.
    Useful for high-cardinality categorical features.
    """
    df = df.copy()
    categories = df[col].astype(str).unique()
    mapping = {cat: i for i, cat in enumerate(categories)}
    df[col] = df[col].map(mapping)

    max_bits = len(bin(len(categories))) - 2
    binary_df = df[col].apply(
        lambda x: list(map(int, bin(x)[2:].zfill(max_bits)))
    )

    binary_cols = [
        f"{col}_bin_{i}" for i in range(max_bits)
    ]

    binary_df = pd.DataFrame(
        binary_df.tolist(),
        columns=binary_cols,
        index=df.index
    )

    df = pd.concat([df.drop(columns=[col]), binary_df], axis=1)
    return df


# --------------------------------------------------
# 5️⃣ FREQUENCY ENCODING
# --------------------------------------------------
def frequency_encoding(df, col):
    """
    Encodes categories based on their frequency.
    Reduces dimensionality.
    """
    df = df.copy()
    freq_map = df[col].value_counts()
    df[col] = df[col].map(freq_map)
    return df


# --------------------------------------------------
# 6️⃣ TARGET ENCODING (SUPERVISED)
# --------------------------------------------------
def target_encoding(df, col, target_col):
    """
    Encodes categories using mean of target variable.
    Only for supervised ML problems.
    """
    df = df.copy()
    target_mean = df.groupby(col)[target_col].mean()
    df[col] = df[col].map(target_mean)
    return df


# --------------------------------------------------
# 🔍 UTILITY FUNCTIONS
# --------------------------------------------------
def get_unique_categories(df, col):
    """Returns unique categories of a column"""
    return df[col].dropna().unique().tolist()


def is_high_cardinality(df, col, threshold=20):
    """
    Checks if column has high cardinality.
    Default threshold = 20 unique values.
    """
    return df[col].nunique() > threshold
