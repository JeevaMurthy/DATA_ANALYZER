import pandas as pd

# ==================================================
# STEP 6: AGGREGATION & SUMMARIZATION
# ==================================================
# Goal: Summarize, group, and reshape data for
# dashboards, reporting, and analytics
# ==================================================


# --------------------------------------------------
# 1️⃣ GROUPBY AGGREGATION
# --------------------------------------------------
def groupby_aggregate(df, group_col, target_col, agg_func):
    """
    Performs groupby aggregation.
    agg_func: mean, sum, count, min, max, std
    """
    return (
        df.groupby(group_col)[target_col]
        .agg(agg_func)
        .reset_index()
    )


# --------------------------------------------------
# 2️⃣ MULTI-AGGREGATION
# --------------------------------------------------
def multi_aggregate(df, group_col, target_col, agg_funcs):
    """
    Performs multiple aggregations at once.
    agg_funcs: list -> ["mean", "sum", "count"]
    """
    return (
        df.groupby(group_col)[target_col]
        .agg(agg_funcs)
        .reset_index()
    )


# --------------------------------------------------
# 3️⃣ PIVOT TABLE
# --------------------------------------------------
def pivot_table(df, index, columns, values, agg_func="mean"):
    """
    Creates pivot table.
    """
    return pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values=values,
        aggfunc=agg_func
    ).reset_index()


# --------------------------------------------------
# 4️⃣ CUMULATIVE AGGREGATION
# --------------------------------------------------
def cumulative_sum(df, col, new_col):
    """
    Creates cumulative sum feature.
    """
    df = df.copy()
    df[new_col] = df[col].cumsum()
    return df


def cumulative_mean(df, col, new_col):
    """
    Creates cumulative mean feature.
    """
    df = df.copy()
    df[new_col] = df[col].expanding().mean()
    return df


# --------------------------------------------------
# 5️⃣ ROLLING AGGREGATION
# --------------------------------------------------
def rolling_aggregate(df, col, window, agg_func, new_col):
    """
    Rolling window aggregation.
    agg_func: mean, sum, std, min, max
    """
    df = df.copy()
    df[new_col] = df[col].rolling(window=window).agg(agg_func)
    return df


# --------------------------------------------------
# 6️⃣ CROSS-TABULATION
# --------------------------------------------------
def cross_tabulation(df, row_col, col_col, normalize=False):
    """
    Creates cross-tabulation (frequency table).
    normalize: True -> percentage
    """
    return pd.crosstab(
        df[row_col],
        df[col_col],
        normalize=normalize
    )


# --------------------------------------------------
# 🔍 UTILITY FUNCTIONS
# --------------------------------------------------
def available_aggregations():
    """
    Returns supported aggregation functions.
    """
    return ["mean", "sum", "count", "min", "max", "std"]
