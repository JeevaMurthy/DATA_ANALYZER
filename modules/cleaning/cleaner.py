import pandas as pd

# --------------------------------------------------
# DATASET METRICS
# --------------------------------------------------
def dataset_metrics(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing": df.isnull().sum().sum()
    }


# --------------------------------------------------
# MISSING VALUE TABLE
# --------------------------------------------------
def missing_value_table(df):
    table = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isnull().sum(),
        "Missing %": (df.isnull().mean() * 100).round(2)
    })
    return table[table["Missing Count"] > 0]


# --------------------------------------------------
# COLUMN-WISE FILLING
# --------------------------------------------------
def fill_column(df, column, method, constant_value=None):
    df = df.copy()

    if method == "Mean":
        df[column] = df[column].fillna(df[column].mean())

    elif method == "Median":
        df[column] = df[column].fillna(df[column].median())

    elif method == "Mode":
        df[column] = df[column].fillna(df[column].mode()[0])

    elif method == "Constant Value":
        df[column] = df[column].fillna(constant_value)

    elif method == "Forward Fill":
        df[column] = df[column].fillna(method="ffill")

    elif method == "Backward Fill":
        df[column] = df[column].fillna(method="bfill")

    return df


# --------------------------------------------------
# ROW-LEVEL CLEANING
# --------------------------------------------------
def drop_null_rows(df):
    return df.dropna()


def drop_duplicate_rows(df):
    return df.drop_duplicates()


def reset_index(df):
    return df.reset_index(drop=True)
