import streamlit as st
import pandas as pd

# ------------------ IMPORT EDA MODULES ------------------
from modules.eda.overview import *
from modules.eda.descriptive import *
from modules.eda.distribution import *
from modules.eda.missing import *
from modules.eda.outliers import *
from modules.eda.correlation import *
from modules.eda.categorical import *
from modules.eda.target import *
from modules.eda.timeseries import *
from modules.eda.advanced import *
from modules.eda.summary import *


def load_css():
    with open("assets/theme.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# --------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------
st.set_page_config(
    page_title="EDA – Exploratory Data Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Exploratory Data Analysis (EDA)")
st.caption("Filter-based, on-demand exploratory analysis")

st.markdown("---")

# --------------------------------------------------------
# CHECK DATA
# --------------------------------------------------------
if "df" not in st.session_state:
    st.warning("⚠️ Please upload and clean a dataset first.")
    st.stop()

df = st.session_state["df"]

# --------------------------------------------------------
# SIDEBAR – EDA CONTROLS
# --------------------------------------------------------
st.sidebar.header("🧠 EDA Controls")

show_overview = st.sidebar.checkbox("🅰 Dataset Overview")
show_desc = st.sidebar.checkbox("🅱 Descriptive Statistics")
show_dist = st.sidebar.checkbox("🅲 Distribution Analysis")
show_missing = st.sidebar.checkbox("🅳 Missing Data Analysis")
show_outliers = st.sidebar.checkbox("🅴 Outlier Analysis")
show_corr = st.sidebar.checkbox("🅵 Correlation Analysis")
show_cat = st.sidebar.checkbox("🅶 Categorical Analysis")
show_target = st.sidebar.checkbox("🅷 Target Analysis")
show_ts = st.sidebar.checkbox("🅸 Time-Series Analysis")
show_adv = st.sidebar.checkbox("🅹 Advanced EDA")
show_summary = st.sidebar.checkbox("🅺 EDA Summary")

# --------------------------------------------------------
# 🅰 DATASET OVERVIEW
# --------------------------------------------------------
if show_overview:
    st.subheader("🅰 Dataset Overview")

    shape = dataset_shape(df)
    c1, c2 = st.columns(2)
    c1.metric("Rows", shape["rows"])
    c2.metric("Columns", shape["columns"])

    st.markdown("**Column Data Types**")
    st.dataframe(column_dtypes(df), use_container_width=True)

    st.markdown("**Feature Type Count**")
    ft = feature_type_count(df)
    st.write(ft)

    mem_df, total_mem = memory_usage(df)
    st.markdown(f"**Memory Usage:** {total_mem} MB")
    st.dataframe(mem_df, use_container_width=True)

    st.markdown("**Constant Columns**")
    st.write(constant_columns(df))

    st.markdown("**High Cardinality Columns**")
    st.dataframe(high_cardinality_columns(df), use_container_width=True)

    st.markdown("---")

# --------------------------------------------------------
# 🅱 DESCRIPTIVE STATISTICS
# --------------------------------------------------------
if show_desc:
    st.subheader("🅱 Descriptive Statistics")

    num_cols = numeric_columns(df)
    selected_cols = st.multiselect("Select numeric columns", num_cols)

    if selected_cols:
        st.dataframe(descriptive_summary(df, selected_cols), use_container_width=True)

    st.markdown("---")

# --------------------------------------------------------
# 🅲 DISTRIBUTION ANALYSIS
# --------------------------------------------------------
if show_dist:
    st.subheader("🅲 Distribution Analysis")

    col = st.selectbox("Select numeric column", numeric_columns(df))
    plot_type = st.selectbox(
        "Select plot type",
        ["Histogram", "Box Plot", "Violin", "CDF", "Q-Q"]
    )

    if plot_type == "Histogram":
        counts, bins = histogram_data(df, col)
        st.bar_chart(counts)

    elif plot_type == "Box Plot":
        st.write(boxplot_stats(df, col))

    elif plot_type == "Violin":
        st.line_chart(violin_data(df, col))

    elif plot_type == "CDF":
        x, y = cdf_data(df, col)
        st.line_chart(pd.DataFrame({"CDF": y}, index=x))

    elif plot_type == "Q-Q":
        theo, sample = qq_plot_data(df, col)
        st.line_chart(pd.DataFrame({"Sample": sample}, index=theo))

    st.markdown("---")

# --------------------------------------------------------
# 🅳 MISSING DATA ANALYSIS
# --------------------------------------------------------
if show_missing:
    st.subheader("🅳 Missing Data Analysis")

    st.dataframe(missing_summary(df), use_container_width=True)
    st.metric("Overall Data Quality Score", overall_data_quality_score(df))

    st.markdown("---")

# --------------------------------------------------------
# 🅴 OUTLIER ANALYSIS
# --------------------------------------------------------
if show_outliers:
    st.subheader("🅴 Outlier Analysis")

    col = st.selectbox("Select numeric column", numeric_columns(df))
    method = st.radio("Method", ["IQR", "Z-Score"])

    if method == "IQR":
        outliers = iqr_outliers(df, col)
    else:
        outliers = zscore_outliers(df, col)

    st.write(f"Outlier count: {len(outliers)}")
    st.dataframe(outliers, use_container_width=True)

    st.markdown("---")

# --------------------------------------------------------
# 🅵 CORRELATION ANALYSIS
# --------------------------------------------------------
if show_corr:
    st.subheader("🅵 Correlation Analysis")

    method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
    corr = correlation_matrix(df, method)

    st.dataframe(corr, use_container_width=True)

    st.markdown("**Highly Correlated Features**")
    st.dataframe(highly_correlated_features(df), use_container_width=True)

    st.markdown("---")

# --------------------------------------------------------
# 🅶 CATEGORICAL ANALYSIS
# --------------------------------------------------------
if show_cat:
    st.subheader("🅶 Categorical Data Analysis")

    cat_cols = categorical_columns(df)
    col = st.selectbox("Select categorical column", cat_cols)

    st.dataframe(frequency_table(df, col), use_container_width=True)
    st.write("Dominance:", category_dominance(df, col))
    st.dataframe(rare_categories(df, col), use_container_width=True)

    st.markdown("---")

# --------------------------------------------------------
# 🅷 TARGET ANALYSIS
# --------------------------------------------------------
if show_target:
    st.subheader("🅷 Target Variable Analysis")

    target_col = st.selectbox("Select target column", df.columns)

    st.write("Target Distribution")
    st.dataframe(target_distribution(df, target_col))

    if pd.api.types.is_numeric_dtype(df[target_col]):
        st.write("Target Variance:", target_variance(df, target_col))

    st.markdown("---")

# --------------------------------------------------------
# 🅸 TIME-SERIES ANALYSIS
# --------------------------------------------------------
if show_ts and has_datetime_column(df):
    st.subheader("🅸 Time-Series Analysis")

    date_col = st.selectbox(
        "Select date column",
        [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    )

    target_col = st.selectbox("Select target column", numeric_columns(df))

    ts_df = ensure_datetime(df, date_col)

    st.line_chart(trend_data(ts_df, target_col))
    st.line_chart(rolling_mean(ts_df, target_col))

    st.markdown("---")

# --------------------------------------------------------
# 🅹 ADVANCED EDA
# --------------------------------------------------------
if show_adv:
    st.subheader("🅹 Advanced / ML-Ready EDA")

    target_col = st.selectbox("Select target (numeric)", numeric_columns(df))
    st.dataframe(feature_importance(df, target_col), use_container_width=True)
    st.dataframe(mutual_information(df, target_col), use_container_width=True)

    st.markdown("---")

# --------------------------------------------------------
# 🅺 EDA SUMMARY
# --------------------------------------------------------
if show_summary:
    st.subheader("🅺 EDA Summary & Report")

    st.dataframe(eda_summary_table(df), use_container_width=True)
    st.text(generate_eda_report(df))
