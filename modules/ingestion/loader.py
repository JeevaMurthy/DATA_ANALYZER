import pandas as pd

def load_dataset(uploaded_file):
    """
    Load CSV or Excel dataset.
    """
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    elif uploaded_file.name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)

    else:
        raise ValueError("Unsupported file format")
