import streamlit as st
from modules.cleaning.cleaner import (
    dataset_metrics,
    missing_value_table,
    fill_column,
    drop_null_rows,
    drop_duplicate_rows,
    reset_index
)

# --------------------------------------------------
# LOAD THEME
# --------------------------------------------------
def load_css():
    with open("assets/theme.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Data Cleaning",
    page_icon="🧹",
    layout="wide"
)

st.title("🧹 Data Cleaning & Missing Value Handling")
st.caption("Step 2: Improve data quality before transformation")

st.markdown("---")

# --------------------------------------------------
# CHECK DATA
# --------------------------------------------------
if "df" not in st.session_state:
    st.warning("⚠️ Please upload a dataset in Page 1 first.")
    st.stop()

df = st.session_state["df"]

# --------------------------------------------------
# METRICS
# --------------------------------------------------
metrics = dataset_metrics(df)

c1, c2, c3 = st.columns(3)
c1.metric("Rows", metrics["rows"])
c2.metric("Columns", metrics["columns"])
c3.metric("Total Missing Values", metrics["missing"])

st.markdown("---")

# --------------------------------------------------
# MISSING VALUE TABLE
# --------------------------------------------------
st.markdown("### 📋 Missing Values by Column")
st.dataframe(missing_value_table(df), use_container_width=True)

st.markdown("---")

# --------------------------------------------------
# COLUMN-WISE FILLING
# --------------------------------------------------
st.markdown("### 🧩 Column-wise Missing Value Filling")

cols_with_missing = df.columns[df.isnull().any()].tolist()

if cols_with_missing:
    selected_col = st.selectbox("Select column", cols_with_missing)

    method = st.selectbox(
        "Select filling technique",
        [
            "Mean",
            "Median",
            "Mode",
            "Constant Value",
            "Forward Fill",
            "Backward Fill"
        ]
    )

    constant_val = None
    if method == "Constant Value":
        constant_val = st.text_input("Enter constant value")

    if st.button("Apply Filling"):
        df = fill_column(df, selected_col, method, constant_val)
        st.success(f"✅ Filled missing values in `{selected_col}`")

else:
    st.success("🎉 No missing values detected.")

st.markdown("---")

# --------------------------------------------------
# ROW-LEVEL CLEANING
# --------------------------------------------------
st.markdown("### 🗑️ Row-Level Cleaning")

c1, c2, c3 = st.columns(3)
drop_null = c1.checkbox("Drop rows with missing values")
drop_dupes = c2.checkbox("Drop duplicate rows")
reset_idx = c3.checkbox("Reset index")

if st.button("Apply Row Cleaning"):
    if drop_null:
        df = drop_null_rows(df)
    if drop_dupes:
        df = drop_duplicate_rows(df)
    if reset_idx:
        df = reset_index(df)

    st.success("✅ Row-level cleaning applied")

st.markdown("---")

# --------------------------------------------------
# PREVIEW
# --------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Rows (After)", df.shape[0])
c2.metric("Columns (After)", df.shape[1])
c3.metric("Remaining Missing Values", df.isnull().sum().sum())

st.dataframe(df.head(), use_container_width=True)

# --------------------------------------------------
# SAVE DATA
# --------------------------------------------------
st.session_state["df"] = df

st.info("ℹ️ Cleaning is user-controlled and column-specific.")
